# 🛍️ Emiasd - Recommender Systems

![EMIASD Dauphine](https://img.shields.io/badge/-EMIASD%20Dauphine-000000?style=flat) ![Python](https://img.shields.io/badge/-Python-3776AB?style=flat&logo=python&logoColor=white) ![LightFM](https://img.shields.io/badge/-LightFM-blue?style=flat) ![Collaborative Filtering](https://img.shields.io/badge/-Collaborative%20Filtering-blue?style=flat) ![Matrix Factorization](https://img.shields.io/badge/-Matrix%20Factorization-blue?style=flat) ![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

---

This repository contains the course material, labs, and projects for the **Recommender Systems** module of the **[EMIASD Executive Master](https://executive-education.dauphine.psl.eu/formations/executive-master-diplome-universite/ia-science-donnees)** (Artificial Intelligence & Data Science, Université Paris-Dauphine \| PSL).

## 📚 Repository Structure

```
├── 1.Cours/              # Course slides (PDFs)
├── 2.Articles/           # Research papers
├── 3.Notes/              # Course notes and mind maps
├── 4.Td/                 # Labs (Jupyter notebooks)
├── 5.Documentations/     # Reference books and syllabus
└── 6.Projects/           # Final project
```

## 🎯 Learning Objectives

The course covers the fundamental and advanced concepts of recommender systems:

1. **Introduction to Recommender Systems**
2. **Reinforcement Learning for RecSys**
3. **Deep Learning for RecSys**
4. **Point-of-Interest (POI) Recommendation**
5. **Recommender System Architecture**
6. **LLM-based Recommender Systems**

## 📊 Datasets Used

### MovieLens 100K (Labs)
- **943** users
- **1,682** movies
- **100,000** ratings
- Used for learning and hands-on exercises

### H&M Fashion Dataset (Final Project)
- **1.3M** customers
- **105,000** articles
- **31M** transactions
- Production-scale dataset for the final project

## 🛠️ Technologies & Libraries

The notebooks mainly use:

```python
# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score
from scipy.sparse.linalg import svds
```

## 📓 Labs (4.Td/)

The notebooks implement and compare **four recommendation approaches**:

### 1. Popularity-Based Recommendation
- Weighted score: `0.7 × average_rating + 0.3 × log(num_ratings)`
- Good baseline, handles cold start

### 2. Content-Based Filtering
- Uses movie genres (18 binary features) + normalized release year
- Cosine similarity between feature vectors
- Explainable recommendations

### 3. Collaborative Filtering (User-Based)
- User-item matrix
- Cosine similarity between users
- Recommends items liked by similar users (rating ≥ 4)

### 4. Matrix Factorization (SVD)
- `scipy.svds()` with 50 latent factors
- Normalized by user mean
- Better personalization

## 📈 Evaluation Metrics

The implementations include:
- **Precision@K, Recall@K, F1@K**
- **Coverage** (catalog diversity)
- Train/test split (80/20)
- 100-user test sample for speed

## 🚀 Quick Start

### Install dependencies

```bash
pip install jupyter pandas numpy matplotlib seaborn scikit-learn scipy
```

### Run the notebooks

```bash
# Option 1: Jupyter Notebook
jupyter notebook

# Option 2: Jupyter Lab
jupyter lab
```

### Run a lab

1. Navigate to `4.Td/`
2. Open `Lab_using_MovieLens.ipynb`
3. The MovieLens dataset downloads automatically on first run

## 📁 Key Files

### Labs
- `Lab_using_MovieLens.ipynb` - Main working version
- `Lab_using_MovieLens_solution.ipynb` - Full solution with implementations
- `Lab_using_MovieLens_original.ipynb` - Original starter template

### ⭐ Final Project — H&M Fashion Recommender System (LightFM)

*With Henri Balamou.* A hybrid recommender system (Collaborative Filtering + item/user features) on the H&M dataset (31.7M transactions, 99.98% sparsity). Key result: the hybrid model doesn't move the global metrics, but improves Precision@5 by **+109%** on cold-start users (496 users with no purchase history) thanks to demographic features.

See **[`6.Projects/README.md`](6.Projects/README.md)** for the full write-up: the 9-step pipeline, detailed results, a matrix-alignment bug we hit and fixed, limitations, and improvement ideas.

## ⚠️ Important Notes

- **MovieLens**: the dataset downloads automatically on first run
- **H&M Dataset**: large files (3.7GB total) — use sampling for development
- **SVD**: can take 30-60 seconds to train on the full MovieLens dataset
- The implementations prioritize **pedagogical clarity** over production performance

## 📖 Sampling Strategy

To work with the H&M dataset, a sampling strategy is recommended:

```python
def sample(df, column_name, k):
    """
    Keep only items/users with >= k transactions
    """
    column_counts = df[column_name].value_counts()
    columns_valid = column_counts[column_counts >= k].index
    return df[df[column_name].isin(columns_valid)]

# Recommended thresholds:
# k_article = 1000 (reduces to ~17M transactions)
# k_customer = 100 (reduces to ~9.4M transactions)
```

## 🎓 Resources

- **Course**: see the `1.Cours/` folder for the PDF slides
- **Documentation**: reference books in `5.Documentations/`
- **Articles**: research papers in `2.Articles/`

## 📝 License

Educational material for the EMIASD program.

---

**Author**: Malik Chettih (final project co-authored with Henri Balamou)
