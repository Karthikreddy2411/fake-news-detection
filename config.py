"""
config.py — All hyperparameters from the paper
"Trust Oriented Explainable AI for Fake News Detection"
arXiv:2603.11778v1
"""

# ── Data paths ─────────────────────────────────────────────────────────────────
DATA_DIR   = "data"
MODELS_DIR = "models"
TRUE_CSV   = "data/True.csv"
FAKE_CSV   = "data/Fake.csv"
TOKENIZER_PATH = "models/tokenizer.json"
LSTM_MODEL_PATH = "models/lstm_model.keras"
CNN_MODEL_PATH  = "models/cnn_model.keras"

# ── Text preprocessing (Section VI-B) ─────────────────────────────────────────
VOCAB_SIZE  = 30_000   # increased: covers new special tokens + richer vocab
MAX_LEN     = 300      # increased: was 150, short articles were mostly padding
OOV_TOKEN   = "<OOV>"
PADDING     = "post"
TRUNCATING  = "post"

# ── Embeddings ─────────────────────────────────────────────────────────────────
EMBED_DIM   = 128

# ── LSTM model (Section VI-C) ──────────────────────────────────────────────────
LSTM_UNITS          = 128
LSTM_DROPOUT        = 0.2
LSTM_RECURRENT_DROP = 0.2

# ── CNN model (Section VI-C) ───────────────────────────────────────────────────
CNN_FILTERS     = 128
CNN_KERNEL_SIZE = 5
CNN_DROPOUT     = 0.5

# ── Training (Section VI-D) ────────────────────────────────────────────────────
BATCH_SIZE    = 1024   # increased for M4 Metal GPU — more work per step = faster
EPOCHS        = 10     # increased from 3; EarlyStopping (patience=2) prevents overfitting
RANDOM_STATE  = 42

# ── Train / val / test split (Section VI-B) ────────────────────────────────────
TRAIN_RATIO = 0.63
VAL_RATIO   = 0.07
TEST_RATIO  = 0.30

# ── XAI settings (Section VI-E) ────────────────────────────────────────────────
SHAP_NUM_SAMPLES    = 100      # masked variants per instance
LIME_NUM_SAMPLES    = 1000     # perturbations per instance
LIME_NUM_FEATURES   = 20       # top influential words to expose
IG_STEPS            = 50       # integration steps (32–50 in paper)

# ── Metrics evaluation (Section VI-G) ──────────────────────────────────────────
TOP_K           = 20           # number of top tokens used for Δcomp / Δsuff / Flip@k
AOPC_STEPS      = 20           # m in AOPC formula
EVAL_SAMPLE_N   = 60           # articles to evaluate (per model per method)
PAD_IDX         = 0            # padding token index
