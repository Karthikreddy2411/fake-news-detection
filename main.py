"""
main.py — CLI entry point
─────────────────────────────────────────────────────────────────────────────
Usage:
  python main.py train              Preprocess + train LSTM & CNN
  python main.py evaluate           Compute XAI metrics (Tables II & III)
  python main.py demo               Interactive demo with a built-in sample
  python main.py demo --text "..."  Explain a custom article
"""

import argparse
import os
import sys

# ── Ensure Keras uses TensorFlow backend ───────────────────────────────────
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf

# ── Force eager execution to avoid model.fit() deadlock on macOS ───────────
tf.config.run_functions_eagerly(True)

import config as C


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────────────────────

def cmd_train(model_choice="both"):
    from src.preprocess import prepare_data
    from src.train import train_model, evaluate_model
    from src.models import build_lstm_model, build_cnn_model

    (X_train, y_train), (X_val, y_val), (X_test, y_test), tok, _ = prepare_data()

    if model_choice in ("cnn", "both"):
        print("\n" + "="*60)
        print("  Training CNN model  [fast — ~3 min]")
        print("="*60)
        cnn = build_cnn_model()
        cnn.summary(line_length=70)
        train_model(cnn, X_train, y_train, X_val, y_val, C.CNN_MODEL_PATH)
        evaluate_model(cnn, X_test, y_test, "CNN")
        print(f"\n  ✓ CNN saved → {C.CNN_MODEL_PATH}")
        print("  ✓ You can now run: streamlit run app.py\n")

    if model_choice in ("lstm", "both"):
        print("\n" + "="*60)
        print("  Training LSTM model  [slow — ~20-40 min on CPU]")
        print("="*60)
        lstm = build_lstm_model()
        lstm.summary(line_length=70)
        train_model(lstm, X_train, y_train, X_val, y_val, C.LSTM_MODEL_PATH)
        evaluate_model(lstm, X_test, y_test, "LSTM")
        print(f"\n  ✓ LSTM saved → {C.LSTM_MODEL_PATH}\n")

    print("✓ Done. Run  streamlit run app.py  to launch the web app.\n")


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────────────────────────────────────

def cmd_evaluate():
    from src.preprocess import prepare_data, load_tokenizer
    from src.models import load_model
    from src.xai.shap_explainer import SHAPExplainer
    from src.xai.lime_explainer import LIMEExplainer
    from src.xai.ig_explainer   import IGExplainer
    from src.metrics import evaluate_xai_method, print_results_table

    # ── Check models exist ─────────────────────────────────────────────────────
    for path in [C.LSTM_MODEL_PATH, C.CNN_MODEL_PATH, C.TOKENIZER_PATH]:
        if not os.path.exists(path):
            sys.exit(f"✗ '{path}' not found. Run  python main.py train  first.")

    print("Loading saved models and tokenizer …")
    tok  = load_tokenizer()
    lstm = load_model(C.LSTM_MODEL_PATH)
    cnn  = load_model(C.CNN_MODEL_PATH)

    # ── Load test set ──────────────────────────────────────────────────────────
    _, _, (X_test, y_test), _, _ = prepare_data()

    # ── LSTM evaluation ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  Evaluating XAI methods on LSTM  (this may take a while …)")
    print("="*60)
    lstm_results = {}

    print("\n  → SHAP")
    shap_exp = SHAPExplainer(lstm, tok)
    lstm_results["SHAP"] = evaluate_xai_method(lstm, shap_exp, X_test, method_name="SHAP / LSTM")

    print("\n  → LIME")
    lime_exp = LIMEExplainer(lstm, tok)
    lstm_results["LIME"] = evaluate_xai_method(lstm, lime_exp, X_test, method_name="LIME / LSTM")

    print("\n  → Integrated Gradients")
    ig_exp = IGExplainer(lstm, tok)
    lstm_results["IG"] = evaluate_xai_method(lstm, ig_exp, X_test, method_name="IG / LSTM")

    print_results_table(lstm_results, "LSTM (Table II)")

    # ── CNN evaluation ─────────────────────────────────────────────────────────
    print("="*60)
    print("  Evaluating XAI methods on CNN")
    print("="*60)
    cnn_results = {}

    print("\n  → SHAP")
    shap_exp_cnn = SHAPExplainer(cnn, tok)
    cnn_results["SHAP"] = evaluate_xai_method(cnn, shap_exp_cnn, X_test, method_name="SHAP / CNN")

    print("\n  → LIME")
    lime_exp_cnn = LIMEExplainer(cnn, tok)
    cnn_results["LIME"] = evaluate_xai_method(cnn, lime_exp_cnn, X_test, method_name="LIME / CNN")

    print("\n  → Integrated Gradients")
    ig_exp_cnn = IGExplainer(cnn, tok)
    cnn_results["IG"] = evaluate_xai_method(cnn, ig_exp_cnn, X_test, method_name="IG / CNN")

    print_results_table(cnn_results, "CNN (Table III)")

    print("✓ Evaluation complete. See tables above (compare with Tables II & III in paper).\n")


# ─────────────────────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_REAL = (
    "The White House confirmed on Thursday that the administration will introduce "
    "a new policy framework aimed at reducing carbon emissions by 40% over the next "
    "decade. Officials stated the plan includes investment in renewable energy "
    "infrastructure and new international climate agreements."
)

SAMPLE_FAKE = (
    "BREAKING: President SECRETLY signs executive order to hand over US sovereignty "
    "to the United Nations! MASSIVE globalist agenda exposed! Patriots are calling "
    "for immediate resistance as the deep state DESTROYS American freedom! Share "
    "before this is CENSORED and deleted by the mainstream media!"
)


def cmd_demo(text: str | None = None, model_name: str = "cnn"):
    from src.preprocess import load_tokenizer
    from src.models import load_model
    from src.xai.shap_explainer import SHAPExplainer
    from src.xai.lime_explainer import LIMEExplainer
    from src.xai.ig_explainer   import IGExplainer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    for path in [C.CNN_MODEL_PATH if model_name == "cnn" else C.LSTM_MODEL_PATH,
                 C.TOKENIZER_PATH]:
        if not os.path.exists(path):
            sys.exit(f"✗ '{path}' not found. Run  python main.py train  first.")

    print(f"\nLoading {model_name.upper()} model and tokenizer …")
    tok   = load_tokenizer()
    model = load_model(C.CNN_MODEL_PATH if model_name == "cnn" else C.LSTM_MODEL_PATH)

    article = text or SAMPLE_FAKE
    print(f"\n{'─'*70}")
    print("  ARTICLE:")
    print(f"  {article[:300]}{'…' if len(article)>300 else ''}")
    print(f"{'─'*70}")

    # Preprocess
    seqs = tok.texts_to_sequences([article])
    X = pad_sequences(seqs, maxlen=C.MAX_LEN,
                      padding=C.PADDING, truncating=C.TRUNCATING,
                      value=C.PAD_IDX)
    prob = float(model.predict(X, verbose=0)[0, 0])
    label = "REAL" if prob >= 0.5 else "FAKE"
    print(f"\n  Prediction : {label}  (confidence: {max(prob, 1-prob):.2%})")

    # LIME (fastest for a quick demo)
    print("\n  Computing LIME explanation (fastest for demo) …")
    lime_exp = LIMEExplainer(model, tok)
    attr = lime_exp.explain(X[0])

    # Print top-10 influential tokens
    index_word = {v: k for k, v in tok.word_index.items()}
    non_pad = [(i, X[0][i], attr[i]) for i in range(C.MAX_LEN) if X[0][i] != C.PAD_IDX]
    top10 = sorted(non_pad, key=lambda t: -abs(t[2]))[:10]

    print("\n  Top-10 influential tokens (LIME):")
    print(f"  {'Word':<20} {'Attribution':>12}  Direction")
    print(f"  {'─'*48}")
    for pos, idx, score in top10:
        word = index_word.get(int(idx), "<OOV>")
        direction = "→ REAL" if score > 0 else "→ FAKE"
        print(f"  {word:<20} {score:>12.4f}  {direction}")

    print(f"\n  Tip: Run  streamlit run app.py  for the full interactive UI.\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fake News Detection with XAI — CLI"
    )
    sub = parser.add_subparsers(dest="command")

    train_p = sub.add_parser("train", help="Preprocess data and train models")
    train_p.add_argument(
        "--model", type=str, default="both",
        choices=["cnn", "lstm", "both"],
        help="Which model to train: cnn (fast ~3min), lstm (slow ~30min), both (default)"
    )

    sub.add_parser("evaluate", help="Compute XAI fidelity metrics (Tables II & III)")

    demo_p = sub.add_parser("demo", help="Run explanation demo on a single article")
    demo_p.add_argument("--text",  type=str, default=None, help="Article text to explain")
    demo_p.add_argument("--model", type=str, default="cnn", choices=["cnn","lstm"])

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(model_choice=getattr(args, "model", "both"))
    elif args.command == "evaluate":
        cmd_evaluate()
    elif args.command == "demo":
        cmd_demo(text=getattr(args, "text", None),
                 model_name=getattr(args, "model", "cnn"))
    else:
        parser.print_help()
