# Amharic_Language_Model
**Unsupervised Pre-trained Subword Embeddings & Language Representation for Amharic**

Part of the research project:
> **"ANLI: A Multi-Domain Benchmark Corpus, Automated Quality Filtering, and Subword Methodologies for Amharic Natural Language Inference"**
> *Tesfaye Bekele Kassa, Dr. Tesfa Tegegne Asfaw, Beyene Feleke Mekonnen*
> *Woldia University & Bahir Dar University, Ethiopia*

---

## 🌟 Overview

This repository houses the **unsupervised pre-training pipelines**, **Ge'ez canonical normalizers**, and **FastText subword vector engines** trained on a curated corpus of **2,342,340 Amharic sentences** ($44,003,990$ tokens).

### Key Features
1. **Deterministic Ge'ez Canonical Normalization (`corpus_preprocessor.py`)**:
   - Reduces vocabulary entropy by **$23.87\%$** ($14.82 \to 11.28$ bits)
   - Collapses 4 homophonous equivalence classes ($C_1-C_4$) with $0.04\%$ False Homonym Error Rate ($\text{FHER}$)
2. **FastText Subword CBOW ($d=300$) (`pretrain_fasttext_embeddings.py`)**:
   - Multi-scale character $n$-grams ($3 \le n \le 6$) with $2,000,000$ hash buckets
   - Guarantees **$0.00\%$ out-of-vocabulary (OOV) rate** on unseen Semitic morphological inflections
3. **Geometric SVD Effective Rank Evaluator (`evaluate_embeddings_svd_rank.py`)**:
   - Measures representation isotropy: $R_{\text{eff}} = \exp\left(-\sum p_i \ln p_i\right)$
   - Demonstrates a **$4.88\times$ isotropy expansion** over standard Word2Vec ($24.3 \to 118.6$)

---

## 📂 Repository Structure

```
Amharic_Language_Model/
├── README.md
├── requirements.txt
├── corpus_preprocessor.py          # Ge'ez orthographic normalization & sentence stream
├── pretrain_fasttext_embeddings.py # FastText CBOW pre-training (d=300, n in [3,6])
├── evaluate_embeddings_svd_rank.py # SVD Effective Rank & Vector space isotropy
├── data/
│   └── raw_corpus_sample.txt       # Sample Amharic raw texts for testing
└── models/
    └── .gitkeep                    # Target directory for exported .bin and .vec models
```

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Preprocess Raw Corpus
```bash
python corpus_preprocessor.py
```

### 3. Pre-train FastText Subword Embeddings ($d=300$)
```bash
# Uses root training.txt by default
python pretrain_fasttext_embeddings.py \
    --corpus training.txt \
    --output models/amharic_fasttext_d300_subword.bin \
    --dim 300 \
    --epochs 10
```

### 4. Evaluate SVD Effective Rank & Isotropy (Table 6)
```bash
python evaluate_embeddings_svd_rank.py
```
