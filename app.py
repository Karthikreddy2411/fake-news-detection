"""
app.py — Streamlit Interactive Web App
─────────────────────────────────────────────────────────────────────────────
Provides an interactive UI to:
  1. Paste any news article
  2. Choose LSTM or CNN classifier
  3. See prediction + confidence
  4. Explore SHAP / LIME / IG explanations as a token heatmap
  5. View Δcomp, Δsuff, AOPC, Flip@k for the current instance

Run: streamlit run app.py
"""

import os
import sys
import time
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(__file__))
import config as C

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="XAI Fake News Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .stApp { background: #050810; color: #f1f5f9; }

  /* Hero banner */
  .hero-banner {
    background: linear-gradient(135deg, rgba(139,92,246,0.18), rgba(6,182,212,0.12));
    border: 1px solid rgba(139,92,246,0.3);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
  }
  .hero-title { font-size: 2rem; font-weight: 800; color: #f1f5f9; margin: 0 0 0.25rem; }
  .hero-sub   { color: #94a3b8; font-size: 0.9375rem; margin: 0; }
  .arxiv-badge {
    display: inline-block;
    background: rgba(6,182,212,0.15);
    border: 1px solid rgba(6,182,212,0.4);
    color: #22d3ee;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    padding: 0.2rem 0.7rem;
    border-radius: 99px;
    margin-top: 0.75rem;
  }

  /* Prediction badge */
  .pred-real {
    display:inline-block; padding:0.4rem 1.2rem;
    background:rgba(16,185,129,0.2); border:1px solid rgba(16,185,129,0.5);
    color:#34d399; border-radius:99px; font-weight:700; font-size:1.1rem;
  }
  .pred-fake {
    display:inline-block; padding:0.4rem 1.2rem;
    background:rgba(239,68,68,0.2); border:1px solid rgba(239,68,68,0.5);
    color:#f87171; border-radius:99px; font-weight:700; font-size:1.1rem;
  }

  /* Metric card */
  .metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    text-align: center;
  }
  .metric-val { font-family:'JetBrains Mono',monospace; font-size:1.4rem; font-weight:700; }
  .metric-lbl { font-size:0.75rem; color:#64748b; margin-top:0.2rem; }
  .mv-comp { color:#a78bfa; }
  .mv-suff { color:#22d3ee; }
  .mv-aopc { color:#fb923c; }
  .mv-flip { color:#f472b6; }

  /* Heatmap token */
  .hm-wrap { font-size:0.95rem; line-height:2.4; font-family:'Inter',sans-serif; }
  .hm-token { border-radius:3px; padding:2px 4px; cursor:default; }

  /* Section headers */
  .sec-hdr {
    font-size:0.7rem; font-weight:700; letter-spacing:0.1em;
    text-transform:uppercase; color:#64748b; margin-bottom:0.5rem;
  }

  /* Info boxes */
  .info-box {
    background:rgba(139,92,246,0.1); border:1px solid rgba(139,92,246,0.25);
    border-radius:8px; padding:0.75rem 1rem; font-size:0.875rem; color:#94a3b8;
    margin-bottom:1rem;
  }
  .warn-box {
    background:rgba(249,115,22,0.1); border:1px solid rgba(249,115,22,0.25);
    border-radius:8px; padding:0.75rem 1rem; font-size:0.875rem; color:#fb923c;
    margin-bottom:1rem;
  }

  /* Verification cards */
  .verif-card {
    background: linear-gradient(135deg, rgba(6,182,212,0.08), rgba(139,92,246,0.06));
    border: 1px solid rgba(6,182,212,0.2);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.75rem;
  }
  .verif-title {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #22d3ee; margin-bottom: 0.75rem;
  }
  .verif-verdict {
    font-size: 1.1rem; font-weight: 700; margin-bottom: 0.5rem;
  }
  .verif-explanation {
    font-size: 0.875rem; color: #94a3b8; line-height: 1.5;
  }
  .source-item {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    margin-bottom: 0.4rem;
    transition: border-color 0.2s;
  }
  .source-item:hover {
    border-color: rgba(139,92,246,0.4);
  }
  .source-name {
    font-size: 0.75rem; font-weight: 600;
    color: #a78bfa; text-transform: uppercase; letter-spacing: 0.05em;
  }
  .source-title {
    font-size: 0.875rem; color: #e2e8f0;
    margin-top: 0.15rem;
  }
  .source-title a { color: #e2e8f0; text-decoration: none; }
  .source-title a:hover { color: #22d3ee; text-decoration: underline; }
  .source-badge-rep {
    display: inline-block; font-size: 0.65rem; font-weight: 600;
    background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.3);
    color: #34d399; padding: 1px 8px; border-radius: 99px; margin-left: 0.5rem;
  }
  .trust-bar-bg {
    width: 100%; height: 8px; background: rgba(255,255,255,0.06);
    border-radius: 99px; overflow: hidden; margin-top: 0.5rem;
  }
  .trust-bar-fill {
    height: 100%; border-radius: 99px;
    transition: width 0.6s ease;
  }

  /* Sidebar */
  [data-testid="stSidebar"] { background: #080d1a; border-right: 1px solid rgba(255,255,255,0.06); }
  [data-testid="stSidebar"] .stMarkdown { color: #94a3b8; }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #8b5cf6, #06b6d4);
    color: white; border: none; border-radius: 8px;
    font-weight: 600; padding: 0.5rem 1.5rem;
    transition: opacity 0.2s;
  }
  .stButton > button:hover { opacity: 0.88; }

  /* Hide Streamlit branding */
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Cached resource loaders
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading tokenizer …")
def get_tokenizer():
    from src.preprocess import load_tokenizer
    return load_tokenizer()


@st.cache_resource(show_spinner="Loading model …")
def get_model(name: str):
    from src.models import load_model
    path = C.LSTM_MODEL_PATH if name == "LSTM" else C.CNN_MODEL_PATH
    return load_model(path)


@st.cache_resource(show_spinner="Initialising SHAP explainer …")
def get_shap(_model, _tok):
    from src.xai.shap_explainer import SHAPExplainer
    return SHAPExplainer(_model, _tok)


@st.cache_resource(show_spinner="Initialising LIME explainer …")
def get_lime(_model, _tok):
    from src.xai.lime_explainer import LIMEExplainer
    return LIMEExplainer(_model, _tok)


@st.cache_resource(show_spinner="Initialising IG explainer …")
def get_ig(_model, _tok):
    from src.xai.ig_explainer import IGExplainer
    return IGExplainer(_model, _tok)


@st.cache_resource(show_spinner="Initialising News Checker …")
def get_news_checker():
    from src.news_checker import NewsChecker
    return NewsChecker(max_results=10)


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap rendering
# ─────────────────────────────────────────────────────────────────────────────

def render_heatmap(token_ids: np.ndarray, attributions: np.ndarray,
                   index_word: dict, max_tokens: int = 120) -> str:
    """
    Returns HTML string of colour-coded tokens.
    Red → pushes toward Real, Blue → pushes toward Fake.
    """
    non_pad = [(i, token_ids[i], attributions[i])
               for i in range(C.MAX_LEN) if token_ids[i] != C.PAD_IDX][:max_tokens]

    if not non_pad:
        return "<em>No tokens found.</em>"

    max_abs = max(abs(a) for _, _, a in non_pad) or 1.0

    html_parts = []
    for pos, idx, score in non_pad:
        word = index_word.get(int(idx), "<OOV>")
        norm = score / max_abs  # [-1, 1]

        if norm > 0:   # → Real: red
            r, g, b = 239, 68, 68
            alpha = 0.15 + abs(norm) * 0.65
        else:          # → Fake: blue
            r, g, b = 59, 130, 246
            alpha = 0.15 + abs(norm) * 0.65

        text_col = "#ffffff" if abs(norm) > 0.4 else "#94a3b8"
        tooltip  = f"score: {score:.4f}"
        style = (f"background:rgba({r},{g},{b},{alpha:.2f});"
                 f"color:{text_col};padding:2px 4px;border-radius:3px;")
        html_parts.append(f'<span style="{style}" title="{tooltip}">{word}</span>')

    return '<div class="hm-wrap">' + " ".join(html_parts) + "</div>"


# ─────────────────────────────────────────────────────────────────────────────
# Instance metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_display_metrics(model, token_ids, attributions):
    from src.metrics import compute_instance_metrics
    return compute_instance_metrics(model, token_ids, attributions)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 XAI Fake News Detector")
    st.markdown('<div class="arxiv-badge">arXiv:2603.11778v1</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### ⚙️ Settings")

    model_choice = st.selectbox("Classifier", ["CNN", "LSTM"],
                                help="CNN: best with IG · LSTM: best with SHAP")

    xai_method = st.selectbox("XAI Method", ["LIME", "SHAP", "Integrated Gradients"],
                              help="LIME is fastest for live demo")

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
This app implements the complete fake news detection pipeline from:

**Trust Oriented Explainable AI for Fake News Detection**
*Siwek, Stankowski, Stodolski — 2026*

**Models**: LSTM · CNN  
**XAI**: SHAP · LIME · IG  
**Dataset**: ISOT Fake News
    """)
    st.markdown("---")

    # Check if models are trained
    models_ready = (
        os.path.exists(C.LSTM_MODEL_PATH) and
        os.path.exists(C.CNN_MODEL_PATH) and
        os.path.exists(C.TOKENIZER_PATH)
    )

    if not models_ready:
        st.warning("⚠️ Models not trained yet.")
        st.markdown("""
**To train:**
```bash
python main.py train
```
        """)
    else:
        st.success("✅ Models ready")


# ─────────────────────────────────────────────────────────────────────────────
# Hero banner
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-banner">
  <div class="hero-title">🧠 XAI Fake News Detector</div>
  <p class="hero-sub">
    Paste any news article below. The classifier predicts Real or Fake,
    then SHAP / LIME / Integrated Gradients explain <em>which words</em> drove the decision —
    via a colour-coded token heatmap.
  </p>
  <span class="arxiv-badge">arXiv:2603.11778v1 · cs.CL · 2026</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_REAL = (
    "The White House confirmed on Thursday that the administration will introduce "
    "a new policy framework aimed at reducing carbon emissions by 40% over the next "
    "decade. Officials stated the plan includes investment in renewable energy "
    "infrastructure and new international climate agreements. The president signed "
    "the bill in a formal ceremony attended by senior members of Congress."
)

SAMPLE_FAKE = (
    "BREAKING: President SECRETLY signs executive order to hand over US sovereignty "
    "to the United Nations! MASSIVE globalist agenda exposed! Patriots are calling "
    "for immediate resistance as the deep state DESTROYS American freedom! Share "
    "before this is CENSORED and deleted by the mainstream media! Wake up America!"
)

col_input, col_sample = st.columns([3, 1])
with col_input:
    article_text = st.text_area(
        "📰 Article Text",
        height=180,
        placeholder="Paste a news article here …",
        key="article_input",
    )

with col_sample:
    st.markdown("<div class='sec-hdr'>Quick samples</div>", unsafe_allow_html=True)
    
    def set_real():
        st.session_state.article_input = SAMPLE_REAL
        
    def set_fake():
        st.session_state.article_input = SAMPLE_FAKE
        
    st.button("Load Real sample", on_click=set_real)
    st.button("Load Fake sample", on_click=set_fake)
    st.markdown("""
<div class="info-box">
ℹ️ Red tokens → Real<br>Blue tokens → Fake
</div>
""", unsafe_allow_html=True)

run_btn = st.button("🔍 Analyse Article", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Check prerequisites
# ─────────────────────────────────────────────────────────────────────────────

if not models_ready:
    st.markdown("""
<div class="warn-box">
⚠️ <strong>Models not found.</strong>
First download the ISOT dataset (see README), then run:<br>
<code style="background:rgba(0,0,0,0.3);padding:2px 6px;border-radius:4px;">
python main.py train
</code>
</div>
""", unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Inference + explanation
# ─────────────────────────────────────────────────────────────────────────────

if run_btn and article_text.strip():
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from src.preprocess import clean_text

    # Load resources
    tok   = get_tokenizer()
    model = get_model(model_choice)
    index_word = {v: k for k, v in tok.word_index.items()}

    # Preprocess text
    cleaned = clean_text(article_text)
    seqs    = tok.texts_to_sequences([cleaned])
    X       = pad_sequences(seqs, maxlen=C.MAX_LEN,
                            padding=C.PADDING, truncating=C.TRUNCATING,
                            value=C.PAD_IDX)
    token_ids = X[0]

    # ── Prediction ─────────────────────────────────────────────────────────────
    prob  = float(model.predict(X, verbose=0)[0, 0])
    label = "REAL" if prob >= 0.5 else "FAKE"
    conf  = max(prob, 1 - prob)

    st.markdown("---")
    st.markdown("### 🎯 Prediction")
    pcol1, pcol2, pcol3 = st.columns([1, 1, 2])
    with pcol1:
        badge_cls = "pred-real" if label == "REAL" else "pred-fake"
        st.markdown(f'<div class="{badge_cls}">{label}</div>', unsafe_allow_html=True)
    with pcol2:
        st.metric("Confidence", f"{conf:.1%}")
    with pcol3:
        st.metric("Model", model_choice)

    st.markdown("---")

    # ── External Verification ──────────────────────────────────────────────────
    st.markdown("### 🌐 Cross-Reference Verification")

    with st.spinner("Searching Google News for matching articles …"):
        checker = get_news_checker()
        verification = checker.check(article_text)
        assessment = checker.get_combined_assessment(conf, label, verification)

    # ── Combined trust indicator ──────────────────────────────────────────────
    trust_color = assessment["trust_color"]
    trust_level = assessment["trust_level"]
    cross_score = assessment["cross_ref_score"]
    bar_width = max(5, int(cross_score * 100))

    vcol1, vcol2 = st.columns([1, 2])
    with vcol1:
        st.markdown(f"""
<div class="verif-card">
  <div class="verif-title">Combined Trust Assessment</div>
  <div class="verif-verdict" style="color:{trust_color};">
    {verification.verdict_emoji} {trust_level} TRUST
  </div>
  <div class="verif-explanation">{assessment["explanation"]}</div>
  <div style="margin-top:0.75rem;">
    <div style="display:flex;justify-content:space-between;font-size:0.7rem;color:#64748b;">
      <span>Cross-ref score</span>
      <span style="color:{trust_color};font-weight:600;">{cross_score:.0%}</span>
    </div>
    <div class="trust-bar-bg">
      <div class="trust-bar-fill" style="width:{bar_width}%;background:{trust_color};"></div>
    </div>
  </div>
  <div style="margin-top:0.75rem;font-size:0.75rem;color:#64748b;">
    <span style="font-weight:600;color:#e2e8f0;">{verification.total_found}</span> articles found &middot;
    <span style="font-weight:600;color:#34d399;">{verification.reputable_count}</span> from reputable sources
  </div>
</div>
""", unsafe_allow_html=True)

    with vcol2:
        if verification.error:
            st.markdown(f"""
<div class="verif-card">
  <div class="verif-title">Matching Sources</div>
  <div class="verif-explanation">❌ {verification.error}</div>
</div>
""", unsafe_allow_html=True)
        elif not verification.articles:
            st.markdown(f"""
<div class="verif-card">
  <div class="verif-title">Matching Sources</div>
  <div class="verif-explanation">
    No matching articles found on Google News.<br>
    This could mean the story is not being covered by any major outlet,
    or the search terms didn't match. Consider verifying manually.
  </div>
</div>
""", unsafe_allow_html=True)
        else:
            sources_html = '<div class="verif-card">\n<div class="verif-title">Matching Sources</div>\n'
            for art in verification.articles[:6]:
                rep_badge = '<span class="source-badge-rep">✓ Reputable</span>' if art.is_reputable else ''
                sources_html += f"""
<div class="source-item">
  <div class="source-name">{art.source}{rep_badge}</div>
  <div class="source-title"><a href="{art.url}" target="_blank">{art.title}</a></div>
</div>
"""
            if verification.total_found > 6:
                sources_html += f'<div style="font-size:0.75rem;color:#64748b;margin-top:0.3rem;">+ {verification.total_found - 6} more articles</div>'
            sources_html += '</div>'
            st.markdown(sources_html, unsafe_allow_html=True)

    # Show the search query used
    with st.expander("🔎 Search query used", expanded=False):
        st.code(verification.query, language=None)

    st.markdown("---")

    # ── XAI Explanation ────────────────────────────────────────────────────────
    st.markdown(f"### 🔬 Explanation — {xai_method}")

    with st.spinner(f"Computing {xai_method} attributions …"):
        t0 = time.time()

        if xai_method == "SHAP":
            exp = get_shap(model, tok)
            attributions = exp.explain(token_ids)

        elif xai_method == "LIME":
            exp = get_lime(model, tok)
            attributions = exp.explain(token_ids)

        else:  # Integrated Gradients
            exp = get_ig(model, tok)
            attributions = exp.explain(token_ids)

        elapsed = time.time() - t0

    st.caption(f"⏱ Computed in {elapsed:.2f}s")

    # ── Token Heatmap ──────────────────────────────────────────────────────────
    st.markdown("<div class='sec-hdr'>Token Saliency Heatmap</div>", unsafe_allow_html=True)
    heatmap_html = render_heatmap(token_ids, attributions, index_word)
    st.markdown(heatmap_html, unsafe_allow_html=True)

    # Legend
    st.markdown("""
<div style="display:flex;align-items:center;gap:1rem;margin-top:0.5rem;font-size:0.8rem;color:#64748b;">
  <span style="color:#f87171;font-weight:600;">■ Red</span> → pushes toward Real news &nbsp;&nbsp;
  <span style="color:#60a5fa;font-weight:600;">■ Blue</span> → pushes toward Fake news
  <span style="margin-left:auto;">Hover a token to see its exact score</span>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Top tokens table ───────────────────────────────────────────────────────
    st.markdown("### 📊 Top Influential Tokens")
    non_pad = [(i, token_ids[i], attributions[i])
               for i in range(C.MAX_LEN) if token_ids[i] != C.PAD_IDX]
    top_tokens = sorted(non_pad, key=lambda t: -abs(t[2]))[:15]

    import pandas as pd
    rows = []
    for pos, idx, score in top_tokens:
        word = index_word.get(int(idx), "<OOV>")
        rows.append({
            "Rank": len(rows) + 1,
            "Token": word,
            "Attribution": round(float(score), 5),
            "Direction": "→ Real" if score > 0 else "→ Fake",
        })
    df_top = pd.DataFrame(rows)
    st.dataframe(df_top, use_container_width=True, hide_index=True)

    # ── Instance fidelity metrics ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📐 Fidelity Metrics (this instance)")

    with st.spinner("Computing fidelity metrics …"):
        metrics = compute_display_metrics(model, token_ids, attributions)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
<div class="metric-card">
  <div class="metric-val mv-comp">{metrics['delta_comp']:.4f}</div>
  <div class="metric-lbl">Δ<sub>comp</sub> (↑ better)</div>
</div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
<div class="metric-card">
  <div class="metric-val mv-suff">{metrics['delta_suff']:.4f}</div>
  <div class="metric-lbl">Δ<sub>suff</sub> (↓ better)</div>
</div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
<div class="metric-card">
  <div class="metric-val mv-aopc">{metrics['aopc']:.4f}</div>
  <div class="metric-lbl">AOPC (↑ better)</div>
</div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
<div class="metric-card">
  <div class="metric-val mv-flip">{metrics['flip_at_k']}</div>
  <div class="metric-lbl">Flip@k (↓ better)</div>
</div>""", unsafe_allow_html=True)

    # ── Attribution bar chart ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📈 Attribution Bar Chart — Top 20 tokens")

    top20 = sorted(non_pad, key=lambda t: -abs(t[2]))[:20]
    words  = [index_word.get(int(idx), f"T{pos}") for pos, idx, _ in top20]
    scores = [s for _, _, s in top20]
    colors = ["#ef4444" if s > 0 else "#3b82f6" for s in scores]

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#080d1a")
    ax.set_facecolor("#080d1a")
    bars = ax.barh(words[::-1], scores[::-1], color=colors[::-1], height=0.6)
    ax.axvline(0, color="#475569", linewidth=0.8)
    ax.tick_params(colors="#94a3b8", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e293b")
    ax.set_xlabel("Attribution score", color="#94a3b8", fontsize=9)
    ax.set_title(f"{xai_method} · {model_choice}", color="#f1f5f9", fontsize=10, pad=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

elif run_btn and not article_text.strip():
    st.warning("Please enter or paste an article first.")

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#475569;font-size:0.8rem;">
  Trust Oriented Explainable AI for Fake News Detection ·
  Siwek, Stankowski, Stodolski · arXiv:2603.11778 · CC BY 4.0
</div>
""", unsafe_allow_html=True)
