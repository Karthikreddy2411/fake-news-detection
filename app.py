"""
app.py — Streamlit Web App for Fake News Detection
Run: streamlit run app.py
"""

import os
import sys
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
import config as C

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered",
)

# ─────────────────────────────────────────────────────────────────────────────
# Simple styling
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .result-box {
    text-align: center;
    padding: 2rem;
    border-radius: 12px;
    margin: 1.5rem 0;
  }
  .result-real {
    background: #d1fae5;
    border: 2px solid #10b981;
  }
  .result-fake {
    background: #fee2e2;
    border: 2px solid #ef4444;
  }
  .result-label {
    font-size: 2.5rem;
    font-weight: 800;
    margin: 0;
  }
  .label-real { color: #059669; }
  .label-fake { color: #dc2626; }
  .confidence {
    font-size: 1.2rem;
    color: #555;
    margin-top: 0.5rem;
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load model & tokenizer
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model …")
def load_resources(model_name):
    from src.preprocess import load_tokenizer
    from src.models import load_model
    tok = load_tokenizer()
    path = C.LSTM_MODEL_PATH if model_name == "LSTM" else C.CNN_MODEL_PATH
    model = load_model(path)
    return tok, model

# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

st.title("📰 Fake News Detector")
st.write("Paste a news article below and click **Analyse** to check if it's real or fake.")

# Check if models exist
models_ready = (
    os.path.exists(C.LSTM_MODEL_PATH) and
    os.path.exists(C.CNN_MODEL_PATH) and
    os.path.exists(C.TOKENIZER_PATH)
)

if not models_ready:
    st.error("⚠️ Models not found. Run `python main.py train` first.")
    st.stop()

# Model selector
model_choice = st.selectbox("Select Model", ["CNN", "LSTM"])

# Text input
article_text = st.text_area(
    "Article Text",
    height=200,
    placeholder="Paste your news article here …",
)

# Analyse button
if st.button("🔍 Analyse", use_container_width=True):
    if not article_text.strip():
        st.warning("Please paste an article first.")
    else:
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from src.preprocess import clean_text

        tok, model = load_resources(model_choice)

        # Preprocess
        cleaned = clean_text(article_text)
        seqs = tok.texts_to_sequences([cleaned])
        X = pad_sequences(seqs, maxlen=C.MAX_LEN,
                          padding=C.PADDING, truncating=C.TRUNCATING,
                          value=C.PAD_IDX)

        # Predict
        prob = float(model.predict(X, verbose=0)[0, 0])
        label = "REAL" if prob >= 0.5 else "FAKE"
        confidence = max(prob, 1 - prob)

        # Display result
        if label == "REAL":
            st.markdown(f"""
            <div class="result-box result-real">
              <p class="result-label label-real">✅ REAL</p>
              <p class="confidence">Confidence: {confidence:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box result-fake">
              <p class="result-label label-fake">❌ FAKE</p>
              <p class="confidence">Confidence: {confidence:.1%}</p>
            </div>
            """, unsafe_allow_html=True)

        # ── GNews Cross-Reference ─────────────────────────────────────────
        st.markdown("---")
        st.subheader("🌐 Online Verification (Google News)")

        with st.spinner("Searching Google News …"):
            from src.news_checker import NewsChecker
            checker = NewsChecker(max_results=10)
            verification = checker.check(article_text)

        st.write(f"**Verdict:** {verification.verdict_emoji} {verification.verdict}")
        st.write(f"**Articles found:** {verification.total_found} ({verification.reputable_count} from reputable sources)")

        if verification.articles:
            for art in verification.articles[:5]:
                tag = " ✅ Reputable" if art.is_reputable else ""
                st.write(f"- [{art.title}]({art.url}) — *{art.source}*{tag}")
        elif verification.error:
            st.warning(f"Could not search: {verification.error}")
        else:
            st.info("No matching articles found online. This story may not be covered by any major outlet.")
