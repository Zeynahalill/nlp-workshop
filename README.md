# NLP Workshop

A hands-on project that shows, step by step, how a computer turns raw
human text into a prediction. Built for a workshop setting: every NLP
stage is a real, separate, runnable piece of code -- not a single hidden
`preprocess()` call.

By the end of this workshop you should be able to explain:

> A computer does not understand human language directly. We first clean
> the raw text, split it into words, remove low-information words,
> normalize each word to its base form, turn the text into numbers, and
> only then does a machine learning model learn from those numbers and
> predict a category for text it has never seen before.

---

## NLP Nedir? / What is NLP?

Natural Language Processing (NLP) is the field of turning human language
(text or speech) into a form a computer can work with, and back again.
Computers only understand numbers, so every NLP system eventually has to
answer the same question: **"how do we turn this sentence into numbers
without losing its meaning?"** This project answers that question one
concrete step at a time.

## Bu Workshop'ta Ne Öğreneceğiz? / What Will We Learn?

- The difference between text **cleaning** and text **normalization**
- What **tokenization** actually does to a sentence
- Why **stopwords** are removed, and what gets lost/gained by doing so
- The difference between **stemming** and **lemmatization**
- How **TF-IDF** turns words into numbers (and what Bag-of-Words, TF and
  IDF mean on their own)
- Why we split data into **train / validation / test**, and what "data
  leakage" means in this context
- How to train and fairly **compare multiple classic ML models**
- How to **evaluate** a classifier properly (accuracy vs. macro F1)
- How to run **live predictions** on text the model has never seen

## NLP Pipeline

```
RAW TEXT
   │
   ▼
TEXT CLEANING            (preprocessing.py -> clean_text)
   │
   ▼
NORMALIZATION             (preprocessing.py -> normalize_text)
   │
   ▼
TOKENIZATION              (preprocessing.py -> tokenize_text)
   │
   ▼
STOPWORD REMOVAL          (preprocessing.py -> remove_stopwords)
   │
   ▼
LEMMATIZATION              (preprocessing.py -> lemmatize_tokens)
   │
   ▼
FEATURE EXTRACTION (TF-IDF)  (feature_extraction.py)
   │
   ▼
TRAIN / VALIDATION / TEST   (modeling.py -> split_dataset)
   │
   ▼
MODEL TRAINING              (modeling.py, scripts/train.py)
   │
   ▼
MODEL EVALUATION            (evaluation.py)
   │
   ▼
FINAL TEST                  (scripts/train.py, step 6)
   │
   ▼
NEW TEXT -> LIVE PREDICTION  (prediction.py, scripts/predict.py, app.py)
```

Run `python scripts/train.py` and read the terminal output top to bottom
-- it prints every one of these stages, in this order, with real values.

## Dataset

**File:** `data/raw/dataset.csv` (700 rows, `text,category` columns, 5
balanced classes: `Billing`, `Delivery`, `Technical Support`, `Account`,
`Product Feedback`)

**Important note on where this dataset came from:** this project was
built in an offline environment with no internet access, so a real
dataset could not be downloaded at build time. `scripts/generate_dataset.py`
instead assembles ~700 realistic customer-support-ticket sentences from a
large bank of opener/detail/closer fragments (the same domain as this
brief's own examples, e.g. *"I was charged an unexpected fee on my credit
card!"*). It is a synthetic teaching dataset, not a scraped real-world
one -- which is fine here, since (per the brief) **the goal of this
project is to demonstrate the NLP pipeline, not to chase high accuracy.**

If you have internet access and want to swap in a fully "real" dataset of
the same shape (`text,category`), good options include:
- Twitter US Airline Sentiment / Complaints (Kaggle)
- Consumer Complaint Database (consumerfinance.gov / data.gov)
- E-commerce Customer Support Tickets (Kaggle / Hugging Face Datasets)

Just replace `data/raw/dataset.csv` with your own two-column CSV and
re-run `python scripts/train.py` -- nothing else needs to change. The
dataset file itself is **not committed to GitHub** (see `.gitignore`);
regenerate it locally with:

```bash
python scripts/generate_dataset.py
```

## 1. Text Cleaning

Removes *structural noise* that carries no meaning: HTML tags, URLs,
punctuation, and extra whitespace. Implemented in `clean_text()`.

```
"<p>I was charged an unexpected fee!</p>" -> "I was charged an unexpected fee"
```

## 2. Normalization

Where cleaning removes noise, normalization makes *different surface
forms of the same meaning* look identical: lowercasing and expanding
contractions ("don't" -> "do not"). Implemented in `normalize_text()`.

```
"I don't like this!" -> "i do not like this"
```

**Cleaning vs. normalization, in one line:** cleaning throws away things
that were never meaningful (HTML, URLs, punctuation); normalization
standardizes things that *are* meaningful but written inconsistently
(case, contractions).

## 3. Tokenization

Splits normalized text into a list of individual word tokens.
Implemented in `tokenize_text()` with a simple regex (`[a-zA-Z]+`) --
transparent and dependency-free, which matters for a workshop where
people should be able to read the whole rule in one line.

```
"this is a good product" -> ["this", "is", "a", "good", "product"]
```

## 4. Stopword Removal

Very common words ("the", "is", "a", "on"...) appear in almost every
sentence and carry little information for *classification*. We filter
them using a small built-in list (`config.py -> STOPWORDS`) rather than
downloading one, so the project works fully offline.

```
Before: ["this", "is", "a", "good", "product"]
After:  ["good", "product"]
```

## 5. Stemming vs. Lemmatization

Both reduce a word to a common base form so that "run", "running", and
"ran" can be treated as related.

- **Stemming** chops off suffixes using crude rules, without caring
  whether the result is a real word (e.g. Porter's stemmer turns
  "studies" into "studi").
- **Lemmatization** aims for the actual dictionary base form ("studies"
  -> "study"), which is more interpretable in a live demo -- so this
  project uses lemmatization.

This project's lemmatizer is **rule-based**, not `nltk`/`spaCy`-based.
Both of those need to download corpora/models on first use, which would
make the workshop depend on internet access mid-session. The rule-based
version in `preprocessing.py -> lemmatize_tokens()` is intentionally
simple and documented in code; swap in `nltk.WordNetLemmatizer` or spaCy
any time you want higher linguistic accuracy.

```
studies  -> study
running  -> run
charged  -> charge
products -> product
```

## 6. Feature Extraction

This is the core question in NLP: **how does a computer turn text into
numbers?** Two building blocks lead up to TF-IDF:

- **Bag of Words (BoW):** represent a text as a vector of raw word
  counts, ignoring order and grammar entirely.
- **TF (Term Frequency):** how often a word appears *in one document*,
  usually normalized by document length.
- **IDF (Inverse Document Frequency):** down-weights words that appear in
  *many* documents (they're less useful for telling documents apart) and
  up-weights rarer, more distinctive words.
- **TF-IDF = TF × IDF:** a word's score is high only when it is frequent
  in this document but rare across the whole dataset -- which is exactly
  the kind of word that's useful for classification.

## 7. TF-IDF (applied)

`feature_extraction.py` fits an `sklearn.TfidfVectorizer` on the
lemmatized text. The Streamlit app shows the **top TF-IDF features**
(word + score) for whatever text you type in, so you can see directly
which words drove the numbers the model actually sees.

## 8. Train / Validation / Test

`modeling.py -> split_dataset()` performs a **stratified** split:

- 60% Train
- 20% Validation
- 20% Test

**Data leakage rule (critical):** the TF-IDF vectorizer is `fit` only on
the TRAIN split (`feature_extraction.py -> fit_tfidf_vectorizer`).
Validation and test texts are only ever `transform`-ed with that already-
fitted vectorizer -- their vocabulary never influences the fit. Model
selection uses the validation set; the test set is touched exactly once,
after the winning model has already been chosen.

## 9. Machine Learning Models

Three classic, interpretable models are trained and compared -- no deep
learning, no LLMs, since the point of this workshop is the classic NLP
pipeline itself:

1. Multinomial Naive Bayes
2. Logistic Regression
3. Linear SVM

`scripts/train.py` prints a comparison table (Accuracy / Precision /
Recall / F1, all computed on the **validation** set) and picks the model
with the best macro F1.

## 10. Evaluation Metrics

Computed once on the held-out **test** set, after model selection:

- **Accuracy** -- fraction of correct predictions overall.
- **Precision (macro)** -- of everything predicted as class X, how much
  actually was X, averaged equally across classes.
- **Recall (macro)** -- of everything that actually was class X, how
  much was found, averaged equally across classes.
- **F1 (macro)** -- harmonic mean of precision and recall, averaged
  equally across classes. Macro averaging matters when classes are
  imbalanced: a model that only ever predicts the majority class can
  still score well on plain accuracy but will score poorly on macro F1.
- **Confusion Matrix** -- saved as an image at
  `outputs/evaluation/confusion_matrix.png`, showing exactly which
  categories get confused with which.

## 11. Live Prediction

After training, `scripts/predict.py` (CLI) and `app.py` (Streamlit) run
brand-new text through the exact same 5 preprocessing stages, then
`transform` (never `fit`) it with the saved vectorizer, then `predict`
(never `fit`/`train`) with the saved model.

**The new text is never added to the dataset, never written to a CSV,
never used to retrain anything.** It is used for exactly one prediction
and then discarded.

---

## Project Structure

```
nlp-workshop/
├── data/
│   └── raw/
│       └── dataset.csv                 # generated, not committed to git
├── src/
│   └── nlp_project/
│       ├── __init__.py
│       ├── config.py                   # paths, stopwords, split ratios, model list
│       ├── preprocessing.py            # cleaning, normalization, tokenization, stopwords, lemmatization
│       ├── feature_extraction.py       # TF-IDF (fit on train only)
│       ├── modeling.py                 # train/val/test split, candidate models
│       ├── evaluation.py               # metrics, comparison table, confusion matrix
│       └── prediction.py               # live prediction from a saved artifact
├── scripts/
│   ├── generate_dataset.py             # builds data/raw/dataset.csv
│   ├── train.py                        # full pipeline, prints every stage, saves artifact
│   └── predict.py                      # CLI live prediction
├── models/
│   └── model_artifacts.joblib          # saved vectorizer + model + metadata (generated)
├── outputs/
│   ├── evaluation/                     # confusion_matrix.png, metrics CSVs (generated)
│   └── predictions/
├── tests/                               # pytest suite (11+ test files)
├── app.py                               # Streamlit demo
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── README.md
```

## Installation

Requires Python 3.10+.

```bash
cd nlp-workshop
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                 # installs nlp_project as an importable package
```

## Dataset Setup

The dataset is not committed to git. Generate it once:

```bash
python scripts/generate_dataset.py
```

This writes 700 rows to `data/raw/dataset.csv`.

## Training

```bash
python scripts/train.py
```

This single command runs steps 1-14 of the pipeline above, prints the
step-by-step demo, the model comparison table, and the final test
metrics, and saves `models/model_artifacts.joblib`.

## Testing

```bash
pytest
```

Covers: cleaning, normalization, tokenization, stopword removal,
lemmatization, TF-IDF (including a no-leakage check), model training,
prediction, artifact loading (including a missing-file case), empty
input, and a full end-to-end run on synthetic data.

## Streamlit Demo

```bash
streamlit run app.py
```

Opens a browser tab where you can type any text and see:
1. Every preprocessing stage's output side by side
2. The top TF-IDF features for that text
3. The predicted category and (when available) a top-3 breakdown
4. Basic model info (which model, how many classes, dataset size)

## NLP Pipeline'ın Özeti / Summary

| Stage | Where |
|---|---|
| Cleaning | `src/nlp_project/preprocessing.py -> clean_text` |
| Normalization | `src/nlp_project/preprocessing.py -> normalize_text` |
| Tokenization | `src/nlp_project/preprocessing.py -> tokenize_text` |
| Stopword removal | `src/nlp_project/preprocessing.py -> remove_stopwords` |
| Lemmatization | `src/nlp_project/preprocessing.py -> lemmatize_tokens` |
| TF-IDF | `src/nlp_project/feature_extraction.py` |
| Train/Val/Test split | `src/nlp_project/modeling.py -> split_dataset` |
| Model training | `src/nlp_project/modeling.py`, `scripts/train.py` |
| Evaluation | `src/nlp_project/evaluation.py` |
| Live prediction (CLI) | `src/nlp_project/prediction.py`, `scripts/predict.py` |
| Live prediction (UI) | `app.py` |

The end goal isn't "we trained a model" -- it's seeing, on one real piece
of text, the whole journey:

```
RAW TEXT -> CLEANING -> NORMALIZATION -> TOKENIZATION -> STOPWORD REMOVAL
-> LEMMATIZATION -> TF-IDF -> NUMERICAL FEATURES -> MACHINE LEARNING
-> TRAINING -> TEST -> PREDICTION
```
