# Trust Oriented XAI for Fake News Detection

Implementation of the complete pipeline from the research paper:

> **Trust Oriented Explainable AI for Fake News Detection**  
> Siwek, Stankowski, Stodolski — arXiv:2603.11778v1 · cs.CL · March 2026

---

## Results

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| **LSTM** | 95.6% | 0.96 | 0.96 | 0.96 |
| **CNN**  | 99.5% | 0.99 | 0.99 | 0.99 |

> Models trained on the ISOT Fake News dataset (44,842 articles).

---

## ⚠️ Model Files Are Not in This Repo

The trained model files (`lstm_model.keras`, `cnn_model.keras`, `tokenizer.json`)
are **not committed to git** because they are 75–80 MB each — too large for GitHub.

**You must train the models yourself after cloning.** This takes ~10 minutes on CPU
(see Step 3 below). Everything is automated with a single command.

---

## Quick Start (for friends cloning this repo)

### Step 1 — Clone & set up the environment

> **Requires Python 3.12** — TensorFlow does not support Python 3.13+ yet.

```bash
git clone https://github.com/Karthikreddy2411/fake-news-detection.git
cd fake-news-detection

# Create virtual environment with Python 3.12
python3.12 -m venv venv

# Activate it
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

# Install all dependencies
pip install -r requirements.txt
```

---

### Step 2 — Download the ISOT Dataset

1. Go to: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
2. Click **Download** (free Kaggle account required)
3. Unzip and place **`True.csv`** and **`Fake.csv`** inside the `data/` folder:

```
fake-news-detection/
└── data/
    ├── True.csv    ← real news articles (Reuters)
    └── Fake.csv    ← fake news (PolitiFact / Wikipedia)
```

**Or use the Kaggle CLI:**
```bash
pip install kaggle
# Put your API token in ~/.kaggle/kaggle.json first
kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset -p data/ --unzip
```

---

### Step 3 — Train the Models

```bash
# Train both LSTM and CNN (recommended)
python main.py train

# Or train individually (CNN is much faster ~3 min)
python main.py train --model cnn
python main.py train --model lstm
```

This will automatically:
- Load and clean 44,842 articles with sensationalism-aware preprocessing
- Build tokenizer (vocab=30,000 with special fake-news tokens)
- Pad/truncate sequences to 300 tokens
- Train LSTM and CNN with early stopping (up to 10 epochs)
- Save models to `models/` (created automatically)

**Expected time:**
| Model | CPU (Apple M-series) | CPU (Intel) |
|-------|---------------------|-------------|
| CNN   | ~3–4 min | ~8–10 min |
| LSTM  | ~8–10 min | ~25–40 min |

---

### Step 4 — Launch the Web App

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**

**Features:**
- Paste any news article → get REAL / FAKE prediction with confidence %
- Choose LSTM or CNN model
- Choose SHAP / LIME / Integrated Gradients explainability method
- See **token heatmap** (which words drove the decision)
- See **top influential tokens** table
- See **Δcomp, Δsuff, AOPC, Flip@k** metrics for this instance

---

### Optional — CLI Demo

```bash
# Run a quick demo with built-in fake news sample
python main.py demo

# Test your own article text
python main.py demo --text "Your article text here" --model lstm

# Compute full XAI metrics (Tables II & III) — takes 1-3 hours
python main.py evaluate
```

---

## What Makes This Preprocessing Different

Previous versions of this project stripped ALL-CAPS, URLs, and exclamation marks
from article text — accidentally erasing the strongest fake-news signals.

The current `clean_text()` in `src/preprocess.py` converts them to **learnable tokens**:

| Input | Token produced |
|-------|---------------|
| `BREAKING`, `SECRETLY`, `MASSIVE` | `<allcaps>` |
| `http://fakesite.com` | `<url>` |
| `!!`, `!?!` | `<exclam>` |
| `"allegedly"`, `"sources claim"` | `<allegedly>` |
| `"anonymous"` | `<anonymous_source>` |
| `"No official confirmed"` | `<no_official_confirm>` |
| `"share before deleted/censored"` | `<share_before_deleted>` |

This lets the model learn that sensationalist language patterns are predictive of fake news.

---

## Project Structure

```
fake-news-detection/
├── data/                       ← Place True.csv + Fake.csv here (not in git)
├── models/                     ← Auto-created when you run train (not in git)
│   ├── lstm_model.keras
│   ├── cnn_model.keras
│   └── tokenizer.json
├── src/
│   ├── preprocess.py           ← Data loading, cleaning, tokenization
│   ├── models.py               ← LSTM + CNN architectures
│   ├── train.py                ← Training + evaluation
│   ├── metrics.py              ← Δcomp, Δsuff, AOPC, Flip@k
│   └── xai/
│       ├── shap_explainer.py
│       ├── lime_explainer.py
│       └── ig_explainer.py
├── app.py                      ← Streamlit web app
├── main.py                     ← CLI: train / evaluate / demo
├── config.py                   ← All hyperparameters
└── requirements.txt
```

---

## Model Architectures

### LSTM
```
Embedding(30001, 128, input_length=300)
→ LSTM(128, dropout=0.2)
→ Dropout(0.2)
→ Dense(1, sigmoid, dtype=float32)
```

### CNN
```
Embedding(30001, 128, input_length=300)
→ Conv1D(128, kernel_size=5, relu)
→ GlobalAveragePooling1D()
→ Dropout(0.5)
→ Dense(1, sigmoid, dtype=float32)
```

Both: `Adam` · `binary_crossentropy` · `batch_size=1024` · `epochs=10 (early stopping)`

---

## XAI Metrics (paper Tables II & III)

### LSTM
| Method | Δcomp ↑ | Δsuff ↓ | AOPC ↑ | Flip@k ↓ |
|--------|---------|---------|--------|----------|
| **SHAP** 🏆 | **0.0862** | **0.4907** | **0.4725** | **9.47** |
| LIME | 0.0067 | 0.5082 | 0.4400 | 9.65 |
| IG | 0.0137 | 0.5030 | 0.4411 | 9.85 |

### CNN
| Method | Δcomp ↑ | Δsuff ↓ | AOPC ↑ | Flip@k ↓ |
|--------|---------|---------|--------|----------|
| **IG** 🏆 | **0.2866** | **0.2008** | **0.6498** | **4.78** |
| LIME | 0.1734 | 0.2256 | 0.6160 | 4.95 |
| SHAP | 0.1274 | 0.3993 | 0.5284 | 4.93 |

---

## License

CC BY 4.0 — following the original paper's license.
