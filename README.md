# Systèmes de Recommandation - EMIASD

Ce repository contient les supports de cours, exercices pratiques et projets pour le module **Systèmes de Recommandation** du **[Master Exécutif EMIASD](https://executive-education.dauphine.psl.eu/formations/executive-master-diplome-universite/ia-science-donnees) (Intelligence Artificielle et Sciences des Données) de l'Université Paris-Dauphine \| PSL**.

## 📚 Structure du Repository

```
├── 1.Cours/              # Supports de cours (PDFs)
├── 2.Articles/           # Articles de recherche
├── 3.Notes/              # Notes de cours et mind maps
├── 4.Td/                 # Travaux dirigés (notebooks Jupyter)
├── 5.Documentations/     # Livres de référence et syllabus
└── 6.Projects/           # Projets finaux
```

## 🎯 Objectifs Pédagogiques

Ce cours couvre les concepts fondamentaux et avancés des systèmes de recommandation :

1. **Introduction aux Systèmes de Recommandation**
2. **Apprentissage par Renforcement pour RecSys**
3. **Deep Learning pour RecSys**
4. **Recommandation de Points d'Intérêt (POI)**
5. **Architecture des Systèmes de Recommandation**
6. **Systèmes de Recommandation basés sur les LLM**

## 📊 Datasets Utilisés

### MovieLens 100K (Travaux Dirigés)
- **943** utilisateurs
- **1,682** films
- **100,000** évaluations
- Utilisé pour l'apprentissage et les exercices pratiques

### H&M Fashion Dataset (Projet Final)
- **1.3M** clients
- **105,000** articles
- **31M** transactions
- Dataset de production pour le projet final

## 🛠️ Technologies et Bibliothèques

Les notebooks utilisent principalement :

```python
# Manipulation de données
import pandas as pd
import numpy as np

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score
from scipy.sparse.linalg import svds
```

## 📓 Travaux Dirigés (4.Td/)

Les notebooks implémentent et comparent **quatre approches** de recommandation :

### 1. Recommandation par Popularité
- Score pondéré : `0.7 × note_moyenne + 0.3 × log(nb_évaluations)`
- Bon baseline, gère le cold start

### 2. Filtrage Basé sur le Contenu
- Utilise les genres de films (18 features binaires) + année de sortie normalisée
- Similarité cosinus entre vecteurs de features
- Recommandations explicables

### 3. Filtrage Collaboratif (User-Based)
- Matrice utilisateurs-items
- Similarité cosinus entre utilisateurs
- Recommande les items appréciés par des utilisateurs similaires (note ≥ 4)

### 4. Factorisation Matricielle (SVD)
- `scipy.svds()` avec 50 facteurs latents
- Normalisation par moyenne utilisateur
- Meilleure personnalisation

## 📈 Métriques d'Évaluation

Les implémentations incluent :
- **Precision@K, Recall@K, F1@K**
- **Coverage** (diversité du catalogue)
- Split train/test (80/20)
- Échantillonnage de 100 utilisateurs test pour la rapidité

## 🚀 Démarrage Rapide

### Installation des dépendances

```bash
pip install jupyter pandas numpy matplotlib seaborn scikit-learn scipy
```

### Lancer les notebooks

```bash
# Option 1 : Jupyter Notebook
jupyter notebook

# Option 2 : Jupyter Lab
jupyter lab
```

### Exécuter un exercice

1. Naviguer vers `4.Td/`
2. Ouvrir `Lab_using_MovieLens.ipynb`
3. Le dataset MovieLens se télécharge automatiquement au premier lancement

## 📁 Fichiers Importants

### Travaux Dirigés
- `Lab_using_MovieLens.ipynb` - Version de travail principale
- `Lab_using_MovieLens_solution.ipynb` - Solution complète avec implémentations
- `Lab_using_MovieLens_original.ipynb` - Template de départ original

### Projet Final
- `exploratory_data_analysis.ipynb` - Analyse exploratoire du dataset H&M
- `helpers.py` - Fonctions utilitaires
- `data/` - Fichiers CSV volumineux (>3GB, exclus de git)

## ⚠️ Notes Importantes

- **MovieLens** : Le dataset se télécharge automatiquement lors de la première exécution
- **H&M Dataset** : Fichiers volumineux (3.7GB total) - utiliser l'échantillonnage pour le développement
- **SVD** : Peut prendre 30-60 secondes pour s'entraîner sur le dataset MovieLens complet
- Les implémentations privilégient la **clarté pédagogique** plutôt que la performance en production

## 📖 Stratégie d'Échantillonnage

Pour travailler avec le dataset H&M, une stratégie d'échantillonnage est recommandée :

```python
def enchantillonner(df, column_name, k):
    """
    Garde uniquement les items/utilisateurs avec >= k transactions
    """
    column_counts = df[column_name].value_counts()
    columns_valid = column_counts[column_counts >= k].index
    return df[df[column_name].isin(columns_valid)]

# Seuils recommandés :
# k_article = 1000 (réduit à ~17M transactions)
# k_customer = 100 (réduit à ~9.4M transactions)
```

## 🎓 Ressources

- **Cours** : Voir le répertoire `1.Cours/` pour les supports PDF
- **Documentations** : Livres de référence dans `5.Documentations/`
- **Articles** : Papers de recherche dans `2.Articles/`

## 📝 Licence

Matériel pédagogique pour le programme EMIASD.

---

**Auteur** : Programme EMIASD
**Dernière mise à jour** : Octobre 2024
