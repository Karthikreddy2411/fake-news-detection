# Trust Oriented XAI for Fake News Detection

Implementation of the complete pipeline from the research paper:

> **Trust Oriented Explainable AI for Fake News Detection**  
> Siwek, Stankowski, Stodolski — arXiv:2603.11778v1 · cs.CL · March 2026

---

## What This Project Does

| Component | Details |
|-----------|---------|
| **Models** | LSTM (128 units) and CNN (128 filters, kernel=5) |
| **XAI methods** | SHAP, LIME, Integrated Gradients |
| **Metrics** | Δcomp, Δsuff, AOPC, Flip@k |
| **Dataset** | ISOT Fake News (44,898 articles) |
| **Interface** | Streamlit web app + CLI |

---

## Setup

### 1. Install dependencies

> **Requires Python 3.12** — TensorFlow does not support Python 3.13+ yet.
> A `venv/` folder has already been created for you with Python 3.12.

**Activate the virtual environment first:**
```bash
cd "mini project"
source venv/bin/activate        # macOS / Linux
# or on Windows: venv\Scripts\activate
```

**Then install (already done if you see the `venv/` folder is populated):**
```bash
pip install -r requirements.txt
```

### 2. Download the ISOT Dataset

1. Go to: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
2. Click **Download** (requires free Kaggle account)
3. Unzip and copy **`True.csv`** and **`Fake.csv`** into the `data/` folder:

```
mini project/
└── data/
    ├── True.csv    ← real news articles from Reuters
    └── Fake.csv    ← fake news flagged by PolitiFact / Wikipedia
```

**Alternative (Kaggle CLI):**
```bash
pip install kaggle
# Set up ~/.kaggle/kaggle.json with your API token
kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset -p data/ --unzip
```

---

## Usage

### Step 1 — Train Models

```bash
python main.py train
```

This will:
- Load and clean the dataset (~44k articles)
- Build tokenizer (vocab=20,000, OOV token)
- Pad/truncate all sequences to 750 tokens
- Train **LSTM** and **CNN** with early stopping
- Save models to `models/lstm_model.keras` and `models/cnn_model.keras`

Expected time: ~5–20 min (GPU) / 30–60 min (CPU)

---

### Step 2 — Launch the Web App

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**

**Features:**
- Paste any news article
- Choose LSTM or CNN
- Choose SHAP / LIME / Integrated Gradients
- See **prediction** + **confidence**
- See **token heatmap** (red=Real, blue=Fake)
- See **top influential tokens** table
- See **Δcomp, Δsuff, AOPC, Flip@k** for this instance

---

### Step 3 (Optional) — Compute Full Metrics (Tables II & III)

```bash
python main.py evaluate
```

Evaluates all 3 XAI methods × 2 models over 60 test articles.
Prints the same metrics as Tables II and III in the paper.

> ⚠️ This can take **1–3 hours** (SHAP and LIME are slow on CPU).

---

### Quick Demo (terminal)

```bash
# Demo with built-in fake news sample (CNN + LIME)
python main.py demo

# Demo with custom text
python main.py demo --text "Your article text here" --model lstm
```

---

## Project Structure

```
mini project/
├── data/                       ← Place True.csv + Fake.csv here
│   └── .gitkeep
├── models/                     ← Auto-created, stores trained models
├── src/
│   ├── __init__.py
│   ├── preprocess.py           ← Data loading, cleaning, tokenization
│   ├── models.py               ← LSTM + CNN architectures
│   ├── train.py                ← Training + evaluation
│   ├── metrics.py              ← Δcomp, Δsuff, AOPC, Flip@k
│   └── xai/
│       ├── __init__.py
│       ├── shap_explainer.py   ← SHAP (TextMasker, 100 samples)
│       ├── lime_explainer.py   ← LIME (1000 perturbations, top-20)
│       └── ig_explainer.py     ← Integrated Gradients (50 steps)
├── app.py                      ← Streamlit web app
├── main.py                     ← CLI: train / evaluate / demo
├── config.py                   ← All hyperparameters from paper
├── requirements.txt
└── README.md
```

---

## Model Architectures (Section VI-C)

### LSTM
```
Embedding(20000, 128, input_length=750)
→ LSTM(128, dropout=0.2, recurrent_dropout=0.2)
→ Dense(1, sigmoid)
```

### CNN
```
Embedding(20000, 128, input_length=750)
→ Conv1D(128, kernel_size=5, activation='relu')
→ GlobalAveragePooling1D()
→ Dropout(0.5)
→ Dense(1, sigmoid)
```

Both: `Adam` optimizer · `binary_crossentropy` loss · `batch_size=64`

---

## Expected Results (paper Tables II & III)

### LSTM
| Method | Δcomp ↑ | Δsuff ↓ | AOPC ↑ | Flip@k ↓ | Time |
|--------|---------|---------|--------|----------|------|
| **SHAP** 🏆 | **0.0862** | **0.4907** | **0.4725** | **9.47** | 4.69s |
| LIME | 0.0067 | 0.5082 | 0.4400 | 9.65 | 4.45s |
| IG | 0.0137 | 0.5030 | 0.4411 | 9.85 | 24.34s |

### CNN
| Method | Δcomp ↑ | Δsuff ↓ | AOPC ↑ | Flip@k ↓ | Time |
|--------|---------|---------|--------|----------|------|
| **IG** 🏆 | **0.2866** | **0.2008** | **0.6498** | **4.78** | **0.16s** |
| LIME | 0.1734 | 0.2256 | 0.6160 | 4.95 | 0.74s |
| SHAP | 0.1274 | 0.3993 | 0.5284 | 4.93 | 1.25s |

---

## License

CC BY 4.0 — following the original paper's license.
