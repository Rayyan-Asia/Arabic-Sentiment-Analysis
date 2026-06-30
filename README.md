# Arabic Sentiment Analysis — Assignment 2

Birzeit University, ECE Department — Second Semester 2025-2026

## Contents

| File | Purpose |
|---|---|
| `arabic_sentiment_analysis.ipynb` | All project code: preprocessing, feature representation, model training, hyperparameter tuning, and evaluation |
| `Assignment 2 Report.md` | Written report |
| `real data.txt` | Raw dataset (tab-separated: `text\tlabel`, no header) |
| `images/` | Plots exported from the notebook and embedded in the report |
| `requirements.txt` | Pinned Python dependencies |

## Setup

Requires Python 3.12.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running

1. Activate the venv (above).
2. Launch Jupyter and open `arabic_sentiment_analysis.ipynb`:
   ```bash
   jupyter notebook arabic_sentiment_analysis.ipynb
   ```
3. Run all cells top to bottom. The notebook expects `real data.txt` in the same directory as the notebook (already included in this repo).
4. Plots are saved into `images/` as the notebook runs; these are the same images referenced in `Assignment 2 Report.md`.

The notebook is organized to mirror the report and the assignment's required pipeline stages, in order:

1. **Imports & dataset loading**
2. **EDA** — class distribution, word count distribution
3. **Preprocessing** — Arabic normalization, stop-word removal, cleaning
4. **Dataset splitting** — stratified 60/20/20 train/val/test
5. **Feature representation** — TF-IDF, Word2Vec (CBOW), FastText
6. **Traditional ML models** — Naïve Bayes, Random Forest (GridSearchCV tuning)
7. **Deep learning models** — CNN, RNN, LSTM (manual grid search, early stopping)
8. **Comparative evaluation** — metrics table, confusion matrices, representation/model comparison plots

## Notes on reproducibility

- All vectorizers, tokenizers, and embedding models are fit on the training split only.
- Random seeds (`random_state=42` / `seed=42`) are set wherever supported, but exact TensorFlow/Keras results may vary slightly across hardware (GPU/Metal vs CPU).
- Random Forest grid search over TF-IDF's 10,000-feature sparse matrix is the slowest step and may take several minutes.
- Deep learning models use early stopping (`patience=5`, `restore_best_weights=True`), so actual training epochs will be fewer than `MAX_EPOCHS=30`.
