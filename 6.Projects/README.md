# 🛍️ H&M Fashion Recommendation Pipeline

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LightFM](https://img.shields.io/badge/LightFM-compatible-green.svg)](https://making.lyst.com/lightfm/docs/home.html)

Pipeline complet de système de recommandation pour H&M utilisant LightFM avec optimisation des hyperparamètres et modèle hybride collaboratif + contenu.

## 📋 Vue d'ensemble

Ce projet consiste à construire un système de recommandation complet en utilisant le **dataset H&M** (issu d'une compétition Kaggle) et la bibliothèque **LightFM**. Le projet couvre l'ensemble du pipeline de développement d'un système de recommandation, de l'analyse exploratoire jusqu'à l'optimisation d'un modèle hybride.

## 🎯 Objectifs du Projet

- Maîtriser le pipeline complet de construction d'un système de recommandation
- Analyser et comprendre un dataset de recommandation réel (31M transactions)
- Implémenter des modèles de filtrage collaboratif et hybride avec LightFM
- Optimiser les hyperparamètres pour maximiser les performances
- Évaluer les modèles avec des métriques appropriées

## 📊 Dataset H&M

### Fichiers Requis

Les données sont disponibles sur Kaggle ou dans le dossier gdrive partagé :

```
data/
├── transactions_train.csv  # 31.7M transactions
├── articles.csv           # 105K articles avec métadonnées
└── customers.csv          # 1.3M clients avec profils
```

### Caractéristiques du Dataset

- **Transactions** : Historique d'achats avec timestamps
- **Articles** : Métadonnées produits (genres, couleurs, départements, etc.)
- **Clients** : Profils démographiques (âge, statut, etc.)

## 🔧 Technologies

### Bibliothèque Principale : LightFM

**LightFM** est une bibliothèque Python conçue pour construire et évaluer des systèmes de recommandation hybrides.

#### Fonctionnalités Clés

✅ **Modèles Hybrides Flexibles** : Combine filtrage collaboratif et content-based
✅ **Multiples Fonctions de Perte** :
   - **WARP** (Weighted Approximate-Rank Pairwise) : Optimise le ranking, adapté aux feedbacks implicites
   - **BPR** (Bayesian Personalized Ranking) : Optimise le ranking par paires
   - **Logistic** : Pour feedbacks explicites
   - **WARP-kos** : Variante de WARP pour datasets très sparse

✅ **Scalabilité** : Gère efficacement de larges datasets
✅ **API Intuitive** : Simple à utiliser

#### Composants d'un Modèle LightFM

1. **Matrice d'Interactions** : Matrice sparse users × items (achats)
2. **Features Utilisateurs/Items** : Matrices optionnelles de métadonnées
3. **Fonction de Perte** : Définit comment le modèle s'entraîne

### Installation

```bash
pip install lightfm
pip install pandas numpy scikit-learn matplotlib seaborn
```

## 📝 Étapes du Projet

### Étape 1 : Exploration & Compréhension des Données

**Objectif** : Se familiariser avec la structure et les caractéristiques du dataset H&M.

**Questions Clés à Explorer :**

- Combien d'utilisateurs et d'items uniques ?
- Quelle est la sparsité du dataset ? (interactions possibles vs réelles)
- Comment les interactions sont-elles distribuées ? Y a-t-il des power users ou blockbusters ?
- Quelle période couvre les données ? Y a-t-il des patterns saisonniers ?
- Quelles métadonnées sont disponibles ?

**Analyses Suggérées :**

- Tracer la distribution des interactions par user/item (histogrammes, boxplots)
- Identifier la longue traîne : quel % d'items/users représentent 80% des interactions ?
- Examiner les items les plus/moins populaires
- Réfléchir : Comment ces patterns affectent votre stratégie de recommandation ?

### Étape 2 : Stratégie d'Échantillonnage

**Objectif** : Créer un dataset gérable pour l'expérimentation tout en préservant les caractéristiques importantes.

**Pourquoi échantillonner ?** Les datasets complets peuvent être coûteux computationnellement. Un bon échantillonnage permet d'itérer rapidement.

**Considérations :**

- Échantillonner users, items ou interactions ? Quels sont les trade-offs ?
- Comment maintenir les distributions du dataset original ?
- Stratégies : aléatoire, stratifiée, ou basée sur les niveaux d'activité
- **Recommandé** : Commencer avec utilisateurs actifs (5+ interactions) et items populaires

**Expérimentation** : Tester différentes tailles (1K, 10K, 50K interactions) et observer l'impact sur les performances.

### Étape 3 : Prétraitement & Construction des Matrices

**Objectif** : Transformer les données brutes en formats adaptés à LightFM.

**Tâches Clés :**

- **Mapping d'IDs** : Créer des mappings entiers pour user_id et item_id (LightFM requiert des indices entiers)
- **Matrice d'Interactions** : Construire une matrice sparse user-item
  - [Documentation LightFM](https://lyst.github.io/lightfm/docs/home.html) - sections "Building the ID mappings" et "Building the interactions matrix"
- **Nettoyage** : Gérer les duplicates, outliers, entrées invalides

### Étape 4 : Stratégie de Split Train/Test

**Objectif** : Créer une configuration d'évaluation robuste qui simule des scénarios réels.

**Documentation** : [LightFM Cross-validation](https://lyst.github.io/lightfm/docs/examples/cross_validation.html)

**Stratégies de Split à Considérer :**

- **Split Temporel** : Découpage basé sur le temps (plus réaliste pour RecSys)
- **Split Aléatoire** : Découpage aléatoire des interactions par utilisateur
- **Split par Utilisateur** : Garder certains utilisateurs entièrement pour le test

### Étape 5 : Entraînement du Modèle

**Objectif** : Construire et comprendre un modèle de filtrage collaboratif de base.

**Documentation** : [LightFM Model Class](https://lyst.github.io/lightfm/docs/lightfm.html)

**Configuration Expérimentale :**

- Commencer avec un modèle simple utilisant uniquement les interactions (sans features)
- Tester différentes fonctions de perte : WARP, BPR, logistic
- Expérimenter avec différents nombres de facteurs latents

**Paramètres à Explorer :**

```python
from lightfm import LightFM

model = LightFM(
    no_components=30,      # Commencer avec 30-50, expérimenter
    loss='warp',           # 'warp' pour feedback implicite
    learning_rate=0.05,    # Tester entre 0.01-0.1
    random_state=42
)

model.fit(
    interactions=train_matrix,
    epochs=20,             # Commencer avec 10-20, surveiller convergence
    num_threads=4
)
```

### Étape 6 : Optimisation des Hyperparamètres

**Objectif** : Améliorer systématiquement les performances du modèle.

**Approches Possibles :**

- **Grid Search Manuel** : Tester des combinaisons de paramètres clés
- **Random Search** : Plus efficace que grid search pour beaucoup de paramètres
- **Validation-based** : Utiliser un set de validation séparé ou cross-validation

**Paramètres Clés à Tuner :**

- Nombre de facteurs latents (no_components: 30, 50, 100, 200)
- Learning rate et régularisation (0.01, 0.05, 0.1)
- Choix de fonction de perte ('warp', 'bpr', 'logistic')
- Nombre d'epochs (surveiller l'overfitting)

### Étape 7 : Évaluation & Interprétation

**Objectif** : Évaluer la qualité du modèle avec des métriques appropriées.

**Documentation** : [LightFM Evaluation](https://lyst.github.io/lightfm/docs/examples/evaluation.html)

**Métriques à Considérer :**

```python
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score

# Precision@K & Recall@K
precision = precision_at_k(model, test_matrix, k=10)
recall = recall_at_k(model, test_matrix, k=10)

# AUC (qualité du ranking global)
auc = auc_score(model, test_matrix)
```

**Questions d'Évaluation :**

- Comment la performance varie avec K (top-5 vs top-20) ?
- Y a-t-il des différences entre segments d'utilisateurs (actifs vs occasionnels) ?
- Qu'est-ce qu'une "bonne" performance pour votre cas d'usage ?

**Au-delà des Chiffres :**

- Inspecter manuellement les recommandations pour quelques utilisateurs
- Vérifier la diversité : recommandez-vous uniquement des items populaires ?

### Étape 8 : Modèle Hybride avec Features Items

**Objectif** : Incorporer les métadonnées des items pour améliorer les recommandations, notamment pour le cold-start.

**Documentation** : [LightFM Dataset Class](https://lyst.github.io/lightfm/docs/lightfm.data.html)

**Implémentation :**

```python
from lightfm.data import Dataset

# Construire le dataset avec features
dataset = Dataset()
dataset.fit(users, items, item_features=item_features)

# Construire les matrices avec features
(interactions, weights) = dataset.build_interactions(interactions_data)
item_features_matrix = dataset.build_item_features(item_features_data)

# Entraîner modèle hybride
hybrid_model = LightFM(loss='warp')
hybrid_model.fit(
    interactions=interactions,
    item_features=item_features_matrix,
    epochs=20
)
```

**Expérimentations Hybrides :**

- Comparer filtrage collaboratif pur vs modèle hybride
- Tester scénarios de cold-start : recommandation pour nouveaux items ?
- Feature ablation : quelles features contribuent le plus ?

## 📈 Métriques de Succès

| Métrique | Description | Interprétation |
|----------|-------------|----------------|
| **Precision@K** | Proportion d'items pertinents dans top-K | Plus élevé = meilleur |
| **Recall@K** | Proportion d'items pertinents retrouvés | Couverture des items pertinents |
| **AUC** | Qualité du ranking global | 0.5 = aléatoire, 1.0 = parfait |
| **NDCG** | Considère l'ordre des items pertinents | Récompense les items pertinents en tête |

## 🗂️ Structure du Projet

```
6.Projects/
├── README.md                          # Ce fichier
├── docs/
│   └── RecSys Project.docx           # Description détaillée du projet
├── data/
│   ├── articles.csv                  # Métadonnées des articles
│   ├── customers.csv                 # Profils clients
│   ├── transactions_train.csv        # Historique des transactions
│   └── zip/                          # Archives des datasets
├── notebooks/
│   ├── 01_data_exploration.ipynb     # À créer - Exploration
│   ├── 02_data_preprocessing.ipynb   # À créer - Prétraitement
│   ├── 03_baseline_model.ipynb       # À créer - Modèle de base
│   ├── 04_hyperparameter_tuning.ipynb # À créer - Optimisation
│   └── 05_hybrid_model.ipynb         # À créer - Modèle hybride
└── src/
    ├── data_utils.py                 # À créer - Utilitaires data
    ├── model_utils.py                # À créer - Utilitaires modèle
    └── evaluation.py                 # À créer - Fonctions d'évaluation
```

## 🚀 Pour Commencer

### 1. Télécharger les Données

```bash
# Depuis Kaggle (nécessite kaggle CLI)
kaggle competitions download -c h-and-m-personalized-fashion-recommendations

# Ou utiliser les fichiers du dossier gdrive partagé
```

### 2. Installer les Dépendances

```bash
pip install lightfm pandas numpy scikit-learn matplotlib seaborn jupyter
```

### 3. Lancer l'Exploration

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

## 💡 Conseils & Bonnes Pratiques

1. **Commencer Petit** : Utilisez un échantillon pour itérer rapidement
2. **Surveiller la Sparsité** : Datasets très sparse nécessitent des approches spécifiques
3. **Validation Temporelle** : Pour RecSys, le split temporel est plus réaliste
4. **Cold Start** : Les modèles hybrides avec features aident pour nouveaux items
5. **Interprétabilité** : Toujours inspecter manuellement quelques recommandations
6. **Diversité** : Éviter de recommander uniquement les items populaires

## 📚 Ressources

### Documentation LightFM
- [Documentation officielle](https://lyst.github.io/lightfm/docs/home.html)
- [Exemples](https://lyst.github.io/lightfm/docs/examples/quickstart.html)
- [API Reference](https://lyst.github.io/lightfm/docs/lightfm.html)

### Articles de Référence
- [LightFM Paper](https://arxiv.org/abs/1507.08439)
- Voir dossier `2.Articles/` pour plus de papers sur RecSys

### Dataset
- [H&M Competition sur Kaggle](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)

## ✅ Livrables Attendus

1. **Notebooks Jupyter** documentés pour chaque étape
2. **Rapport d'analyse exploratoire** avec visualisations
3. **Modèle baseline** (filtrage collaboratif pur)
4. **Modèle optimisé** avec hyperparamètres tunés
5. **Modèle hybride** intégrant les features items
6. **Analyse comparative** des performances
7. **Recommandations** pour quelques utilisateurs tests avec interprétation

## ⚠️ Notes Importantes

- **Taille des Données** : Le dataset complet est volumineux (3.7GB). Utilisez l'échantillonnage pour le développement.
- **Temps de Calcul** : L'entraînement sur le dataset complet peut prendre plusieurs heures. Optimisez d'abord sur un échantillon.
- **Mémoire** : Les matrices sparse peuvent consommer beaucoup de RAM. Surveillez l'utilisation mémoire.
- **Reproductibilité** : Utilisez `random_state` pour des résultats reproductibles.

---

**Bon courage pour le projet !** 🚀
