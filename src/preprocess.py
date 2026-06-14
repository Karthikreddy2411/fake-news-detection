"""
src/preprocess.py
─────────────────────────────────────────────────────────────────────────────
Data loading, cleaning and vectorisation as described in Section VI-A / VI-B.

Steps:
  1. Load True.csv + Fake.csv, assign labels (1=real, 0=fake)
  2. Basic text cleaning (URLs, HTML, special chars, whitespace)
  3. Build Keras Tokenizer on the full corpus (top VOCAB_SIZE words + OOV)
  4. Convert every article to a fixed-length integer sequence (MAX_LEN=750)
  5. Stratified split: 63 % train / 7 % val / 30 % test (seed=42)
  6. Persist the tokenizer to TOKENIZER_PATH for inference
"""

import os
import re
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config as C


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Light cleaning: remove URLs, HTML tags, extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+", " ", text)      # URLs
    text = re.sub(r"<[^>]+>", " ", text)              # HTML tags
    text = re.sub(r"[^a-zA-Z0-9\s'.,!?]", " ", text) # special chars
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def load_dataset() -> pd.DataFrame:
    """
    Load and merge True.csv + Fake.csv.
    Keeps 'text' and 'title' columns; concatenates them for richer context.
    Labels: 1 = real news, 0 = fake news.
    """
    if not os.path.exists(C.TRUE_CSV):
        raise FileNotFoundError(
            f"\n\n  ✗ '{C.TRUE_CSV}' not found.\n"
            "  Please download the ISOT dataset from Kaggle:\n"
            "    https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset\n"
            "  and place True.csv + Fake.csv inside the 'data/' folder.\n"
        )

    print("Loading dataset …")
    real = pd.read_csv(C.TRUE_CSV)
    fake = pd.read_csv(C.FAKE_CSV)

    real["label"] = 1
    fake["label"] = 0

    df = pd.concat([real, fake], ignore_index=True)

    # Combine title + text for richer signal (common in NLP literature)
    for col in ["title", "text", "subject", "date"]:
        if col not in df.columns:
            df[col] = ""

    df["content"] = (df["title"].fillna("") + " " + df["text"].fillna("")).apply(clean_text)

    # Drop empty or very short articles
    df = df[df["content"].str.split().str.len() > 10].reset_index(drop=True)

    print(f"  ✓ {len(df):,} articles loaded  "
          f"(real={df['label'].sum():,}  fake={(df['label']==0).sum():,})")
    return df[["content", "label"]]


# ─────────────────────────────────────────────────────────────────────────────
# Tokenisation
# ─────────────────────────────────────────────────────────────────────────────

def build_tokenizer(texts, save=True):
    """Fit Keras Tokenizer on training texts and optionally persist it."""
    print(f"Building tokenizer (vocab={C.VOCAB_SIZE}, OOV='{C.OOV_TOKEN}') …")
    tok = Tokenizer(num_words=C.VOCAB_SIZE, oov_token=C.OOV_TOKEN)
    tok.fit_on_texts(texts)
    if save:
        os.makedirs(C.MODELS_DIR, exist_ok=True)
        with open(C.TOKENIZER_PATH, "w") as f:
            json.dump(tok.to_json(), f)
        print(f"  ✓ Tokenizer saved → {C.TOKENIZER_PATH}")
    return tok


def load_tokenizer():
    """Load a previously saved Keras tokenizer from JSON."""
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    with open(C.TOKENIZER_PATH) as f:
        tok_json = json.load(f)
    return tokenizer_from_json(tok_json)


def texts_to_sequences(tok: Tokenizer, texts) -> np.ndarray:
    """Convert list of strings to padded integer sequences (shape [N, MAX_LEN])."""
    seqs = tok.texts_to_sequences(texts)
    return pad_sequences(
        seqs,
        maxlen=C.MAX_LEN,
        padding=C.PADDING,
        truncating=C.TRUNCATING,
        value=C.PAD_IDX,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main split function
# ─────────────────────────────────────────────────────────────────────────────

def prepare_data():
    """
    Full pipeline:
      load → clean → split → tokenize → vectorise
    Returns:
      (X_train, y_train), (X_val, y_val), (X_test, y_test), tokenizer, df_test
    """
    df = load_dataset()

    texts  = df["content"].tolist()
    labels = df["label"].values.astype("float32")

    # ── Stratified split: train+val vs test (70% vs 30%) ──────────────────────
    X_tv, X_test, y_tv, y_test = train_test_split(
        texts, labels,
        test_size=C.TEST_RATIO,
        random_state=C.RANDOM_STATE,
        stratify=labels,
    )

    # ── Within train+val: train vs val (90% vs 10%, so global 63% vs 7%) ──────
    val_ratio_local = C.VAL_RATIO / (C.TRAIN_RATIO + C.VAL_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size=val_ratio_local,
        random_state=C.RANDOM_STATE,
        stratify=y_tv,
    )

    print(f"  Split → train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}")

    # ── Tokenizer fitted ONLY on training texts ────────────────────────────────
    tok = build_tokenizer(X_train, save=True)

    # ── Vectorise all splits ───────────────────────────────────────────────────
    print("Vectorising …")
    X_train_seq = texts_to_sequences(tok, X_train)
    X_val_seq   = texts_to_sequences(tok, X_val)
    X_test_seq  = texts_to_sequences(tok, X_test)
    print("  ✓ Done.")

    # Keep raw test texts for XAI (SHAP / LIME need strings)
    df_test = pd.DataFrame({"text": X_test, "label": y_test})

    return (
        (X_train_seq, y_train),
        (X_val_seq,   y_val),
        (X_test_seq,  y_test),
        tok,
        df_test,
    )


if __name__ == "__main__":
    prepare_data()
