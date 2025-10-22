# Questions de Réflexion - Système de Recommandation H&M

Ce document répond aux questions stratégiques soulevées après l'analyse exploratoire (Step 1) et l'échantillonnage (Step 2) du dataset H&M.

---

## 1. Sparsité : Impact et Techniques Adaptées

### 🔍 Problématique

**Constat (Step 1)** :
- **Sparsité du dataset : 99.98%**
- Matrice user-item : 1.36M × 104.5K = 142 milliards de cellules
- Seulement 31.7M interactions (0.02% de densité)
- Après sampling (Step 2) : Sparsité réduite à ~99.5-99.7%

### ⚠️ Impact sur les Modèles

#### 1.1 Collaborative Filtering Classique

**Problèmes** :
- **Manque de données** : Beaucoup de users/items ont très peu d'interactions
- **Similarité peu fiable** : Difficile de trouver des users/items similaires avec peu de données partagées
- **Recommandations biaisées** : Tendance à recommander uniquement les items populaires
- **Cold-start sévère** : Impossible de recommander pour nouveaux users/items

**Impact sur K-NN (User-based CF)** :
```
Exemple : User A a acheté 7 articles (médiane observée)
→ Peu de chevauchement avec autres users
→ Similarité cosinus souvent proche de 0
→ Voisins peu pertinents
```

#### 1.2 Matrix Factorization (SVD)

**Problèmes** :
- **Overfitting** : Risque élevé sur données très sparse
- **Convergence difficile** : Beaucoup de cellules vides à estimer
- **Bruit** : Items/users rares génèrent du bruit dans les facteurs latents

**Avantages relatifs** :
- Mieux que K-NN pour gérer la sparsité
- Capture des patterns latents même avec peu de données

### ✅ Techniques Adaptées

#### 1. Filtrage par Activité Minimale (Implémenté Step 2)
```python
# Stratégie S2
MIN_ITEM_TRANSACTIONS = 10
MIN_USER_TRANSACTIONS = 5
```
**Avantages** :
- Réduit la sparsité de 99.98% → 99.5-99.7%
- Élimine le bruit (items/users avec données insuffisantes)
- Améliore la qualité des similarités calculées

#### 2. Matrix Factorization avec Régularisation
```python
# SVD avec régularisation L2
from scipy.sparse.linalg import svds
U, sigma, Vt = svds(user_item_matrix, k=50)

# Ou LightFM avec régularisation intégrée
model = LightFM(
    no_components=50,
    loss='warp',
    item_alpha=0.0001,  # Régularisation items
    user_alpha=0.0001   # Régularisation users
)
```
**Avantages** :
- Prévient l'overfitting
- Généralise mieux sur données sparse

#### 3. Hybrid Models (LightFM) - **SOLUTION OPTIMALE**
```python
# Combiner interactions + métadonnées
model.fit(
    interactions=interactions_matrix,
    item_features=item_features_matrix,
    user_features=user_features_matrix
)
```
**Avantages** :
- **Résout partiellement le problème de sparsité**
- Utilise les métadonnées quand les interactions manquent
- Fonctionne même pour items/users avec peu d'interactions

#### 4. Implicit Feedback Algorithms
```python
# WARP loss (LightFM) optimisé pour feedback implicite
model = LightFM(loss='warp')
```
**Pourquoi ?** :
- Notre dataset = achats binaires (pas de ratings)
- WARP optimise le ranking (pas la prédiction de note)
- Mieux adapté à la sparsité extrême

#### 5. Popularity Baseline + Hybrid
```python
# Stratégie de fallback
if n_interactions_user < 5:
    return popularity_recommendations()
else:
    return hybrid_recommendations()
```
**Avantages** :
- Garantit toujours des recommandations
- Fallback sur popularité pour cas extrêmes

### 📊 Comparaison des Techniques

| Technique | Gestion Sparsité | Complexité | Performance Attendue |
|-----------|------------------|------------|---------------------|
| User-based K-NN | ⚠️ Faible | Moyenne | Faible sur dataset sparse |
| SVD | ✓ Moyenne | Haute | Moyenne |
| LightFM (Hybrid) | ✅ Excellente | Haute | **Élevée** |
| Popularity | ✅ Immunisé | Faible | Baseline acceptable |

### 🎯 Recommandation

**Approche en cascade** :
1. **Baseline** : Popularity-based (pour validation)
2. **Intermédiaire** : SVD avec régularisation
3. **Final** : **LightFM Hybrid** avec item features (product_group, index_group, garment_group)

---

## 2. Longue Traîne : Recommander les Items Peu Populaires

### 🔍 Problématique

**Constat (Step 1)** :
- **Règle 80/20 vérifiée** : ~30% des items génèrent 80% des transactions
- **Distribution très déséquilibrée** :
  - Médiane : 58 transactions/item
  - Moyenne : 304 transactions/item
  - Top item : >10,000 transactions
  - Items de la longue traîne : 1-10 transactions

**Impact Business** :
- Recommander uniquement les blockbusters → Manque de diversité
- Items de niche ignorés → Perte de revenus potentiels
- Satisfaction client réduite → Recommandations trop génériques

### ✅ Stratégies de Recommandation

#### 2.1 Content-Based Filtering pour la Longue Traîne

**Principe** :
```python
# Similarité basée sur métadonnées, pas sur popularité
item_features = [
    'product_group_name',
    'index_group_name',
    'garment_group_name',
    'colour_group_name'
]
# → Items rares mais similaires aux achats passés
```

**Avantages** :
- **Indépendant de la popularité**
- Recommande items rares si métadonnées similaires
- Exploite les 25 colonnes de métadonnées du dataset

**Exemple** :
```
User a acheté: "Robe Ladieswear Jersey Basic Black"
Content-Based peut recommander: "Jupe Ladieswear Jersey Basic Black"
→ Même si la jupe a peu de ventes (longue traîne)
```

#### 2.2 Hybrid Model avec Boost pour Diversité

**Stratégie LightFM** :
```python
# Entraîner avec métadonnées
model.fit(
    interactions=interactions,
    item_features=item_features_matrix
)

# Post-traitement : diversification
def diversified_recommendations(user_id, n=10):
    # Générer 50 candidats
    candidates = model.predict(user_id, item_ids, n=50)

    # Sélectionner top 10 avec diversité
    selected = []
    for item in candidates:
        if item.popularity_score < POPULARITY_THRESHOLD:
            # Boost pour items longue traîne
            item.score *= 1.2
        selected.append(item)

    return sorted(selected, key=lambda x: x.score)[:n]
```

**Paramètre de contrôle** :
```python
POPULARITY_THRESHOLD = median_item_popularity
# Items sous la médiane = longue traîne
```

#### 2.3 Exploration vs Exploitation

**Stratégie ε-greedy** :
```python
import random

def recommend_with_exploration(user_id, n=10, epsilon=0.2):
    if random.random() < epsilon:
        # 20% du temps : recommander items aléatoires de la longue traîne
        return sample_from_long_tail(n)
    else:
        # 80% du temps : recommandations optimales
        return hybrid_recommendations(user_id, n)
```

**Avantages** :
- Collecte données sur items peu populaires
- Améliore progressivement les recommandations
- Équilibre entre pertinence et découverte

#### 2.4 Segmentation par Catégorie

**Insight Step 1** : 5 index groups (Ladieswear, Menswear, Baby/Children, Divided, Sport)

**Stratégie** :
```python
# Assurer diversité par catégorie
def diverse_recommendations(user_id, n=10):
    recommendations = []

    # 60% Ladieswear (dominant)
    recommendations += hybrid_recommend(user_id,
                                       category='Ladieswear', n=6)

    # 20% Menswear
    recommendations += hybrid_recommend(user_id,
                                       category='Menswear', n=2)

    # 20% Autres (Baby, Divided, Sport)
    recommendations += hybrid_recommend(user_id,
                                       category='Other', n=2)

    return recommendations
```

**Avantages** :
- Garantit exposition de toutes les catégories
- Découvre préférences latentes
- Augmente la couverture du catalogue

#### 2.5 Métriques de Couverture

**Mesures à suivre** :
```python
# Catalog Coverage
def calculate_coverage(recommendations, total_items):
    recommended_items = set(recommendations)
    return len(recommended_items) / total_items

# Long-tail Coverage (items avec <médiane transactions)
def long_tail_coverage(recommendations, long_tail_items):
    recommended_long_tail = set(recommendations) & set(long_tail_items)
    return len(recommended_long_tail) / len(long_tail_items)
```

**Objectif** :
- Catalog Coverage > 10% (vs ~1-2% pour popularité seule)
- Long-tail Coverage > 5%

### 📊 Comparaison des Approches

| Approche | Longue Traîne | Pertinence | Diversité |
|----------|---------------|------------|-----------|
| Collaborative pur | ❌ Très faible | ✅ Élevée | ❌ Faible |
| Content-Based | ✅ Excellente | ⚠️ Moyenne | ✅ Élevée |
| Hybrid (LightFM) | ✅ Bonne | ✅ Élevée | ✓ Moyenne |
| + Diversification | ✅ **Excellente** | ✓ Bonne | ✅ **Élevée** |

### 🎯 Recommandation

**Pipeline recommandé** :
1. **Hybrid Model (LightFM)** : Base solide avec métadonnées
2. **Post-processing** : Boost items longue traîne (+20% score)
3. **Diversification** : Garantir mix catégories
4. **Exploration** : 10-20% d'aléatoire sur longue traîne
5. **Monitoring** : Suivre métriques de couverture

---

## 3. Cold Start : Nouveaux Users et Nouveaux Items

### 🔍 Problématique

**Définitions** :
- **Cold-start User** : Client sans historique d'achat (0 transactions)
- **Cold-start Item** : Article jamais acheté (0 transactions)
- **Warm-start** : User/Item avec 1-4 transactions (données insuffisantes)

**Prévalence** :
- Dans notre dataset : Tous les users ont ≥1 transaction (par définition)
- Après sampling Step 2 : Users ≥5 trans, Items ≥10 trans
- Mais en production : Nouveaux users/items arrivent constamment

### ✅ Stratégies Cold-Start

#### 3.1 Cold-Start USER (Nouveau Client)

##### **Approche 1 : Popularity-Based Recommendations**

```python
def recommend_new_user(n=10):
    """Recommander les items les plus populaires globalement"""

    # Weighted popularity score (Step 1)
    popular_items = articles.join(
        transactions.groupby('article_id').agg({
            'customer_id': 'count',
            'price': 'mean'
        })
    )

    # Score = 0.7 * avg_rating + 0.3 * log(count)
    popular_items['score'] = (
        0.7 * popular_items['avg_price'] +
        0.3 * np.log(popular_items['count'] + 1)
    )

    return popular_items.nlargest(n, 'score')
```

**Avantages** :
- Simple et rapide
- Toujours disponible
- Performance baseline acceptable

**Limites** :
- Pas personnalisé
- Ignore les préférences potentielles

##### **Approche 2 : Demographic-Based (si données disponibles)**

```python
def recommend_by_demographics(user_age, user_gender, n=10):
    """Recommander basé sur démographie similaire"""

    # Filtrer transactions de users similaires
    similar_users = customers[
        (customers['age'].between(user_age - 5, user_age + 5))
    ]['customer_id']

    # Récupérer leurs achats populaires
    popular_for_segment = transactions[
        transactions['customer_id'].isin(similar_users)
    ].groupby('article_id').size().nlargest(n)

    return popular_for_segment
```

**Données disponibles H&M** :
- Age (85% de couverture)
- Club member status
- Fashion news frequency

**Limites** :
- Données démographiques souvent manquantes (65% NaN pour certaines colonnes)

##### **Approche 3 : Onboarding Questionnaire**

```python
# Demander préférences initiales
user_preferences = {
    'index_group': 'Ladieswear',
    'product_groups': ['Garment Upper body', 'Accessories'],
    'color_preferences': ['Black', 'White', 'Blue']
}

def recommend_from_preferences(preferences, n=10):
    """Recommander basé sur préférences explicites"""

    # Filtrer catalogue
    filtered = articles[
        (articles['index_group_name'] == preferences['index_group']) &
        (articles['product_group_name'].isin(preferences['product_groups'])) &
        (articles['colour_group_name'].isin(preferences['color_preferences']))
    ]

    # Combiner avec popularité dans segment
    return filtered.nlargest(n, 'popularity_score')
```

**Avantages** :
- Personnalisation immédiate
- Collecte données pour amélioration future

##### **Approche 4 : Hybrid Cold-Start avec Features**

```python
# LightFM avec user features
model = LightFM(loss='warp')

# Créer profil user basé sur métadonnées disponibles
user_features = [
    'age_group:25-35',
    'club_member:ACTIVE',
    'news_frequency:Monthly'
]

# Recommander avec features uniquement
predictions = model.predict(
    user_ids=[NEW_USER_ID],
    item_ids=all_items,
    user_features=user_features_matrix
)
```

**Avantages** :
- Utilise métadonnées user disponibles
- Meilleur que popularité pure
- S'améliore avec premières interactions

#### 3.2 Cold-Start ITEM (Nouvel Article)

##### **Approche 1 : Content-Based Similarity**

```python
def recommend_new_item_to_users(new_item_id, n_users=100):
    """Recommander nouvel item aux users susceptibles d'aimer"""

    # Features du nouvel item
    new_item_features = articles[articles['article_id'] == new_item_id][[
        'product_group_name',
        'index_group_name',
        'garment_group_name',
        'colour_group_name'
    ]]

    # Trouver items similaires avec historique
    similar_items = calculate_similarity(
        new_item_features,
        existing_items_features
    )

    # Users qui ont aimé items similaires
    target_users = transactions[
        transactions['article_id'].isin(similar_items)
    ]['customer_id'].unique()

    return target_users[:n_users]
```

**Avantages** :
- **Exploite les 25 colonnes de métadonnées**
- Pas besoin d'historique d'achat pour l'item
- Ciblage précis

##### **Approche 2 : LightFM avec Item Features**

```python
# Item features matrix pour nouvel item
new_item_features = encode_item_features(new_item_id)

# Prédictions pour tous les users
for user_id in active_users:
    score = model.predict(
        user_ids=[user_id],
        item_ids=[new_item_id],
        item_features=new_item_features
    )
```

**Avantages** :
- **Solution optimale pour notre projet**
- Combine interaction patterns + métadonnées
- Fonctionne dès ajout de l'item

##### **Approche 3 : Fallback Stratégique**

```python
def recommend_with_fallback(user_id, n=10):
    """Stratégie en cascade"""

    # 1. Essayer Hybrid (nécessite user + item data)
    try:
        recs = hybrid_recommendations(user_id, n)
        if len(recs) >= n:
            return recs
    except InsufficientDataError:
        pass

    # 2. Fallback sur Content-Based (nécessite user OU item data)
    try:
        recs = content_based_recommendations(user_id, n)
        if len(recs) >= n:
            return recs
    except InsufficientDataError:
        pass

    # 3. Fallback sur Popularity (toujours disponible)
    return popularity_recommendations(n)
```

### 📊 Matrice de Décision Cold-Start

| Scénario | Données User | Données Item | Approche Recommandée |
|----------|--------------|--------------|----------------------|
| **New User, Existing Items** | Démographie | Historique complet | **Demographic-Based + Popularity** |
| **New User, No Data** | Aucune | Historique complet | **Popularity-Based** |
| **Existing User, New Item** | Historique complet | Métadonnées | **LightFM avec Item Features** |
| **New User, New Item** | Aucune | Métadonnées | **Content-Based sur métadonnées** |
| **Warm-Start (1-4 trans)** | Peu | Peu | **Hybrid (LightFM)** |

### 🎯 Recommandation pour Projet H&M

**Architecture en 3 niveaux** :

```python
class RecommendationService:
    def recommend(self, user_id, n=10):
        # Niveau 1: Vérifier données disponibles
        user_data = self.get_user_data(user_id)

        if user_data['n_transactions'] >= 5:
            # User chaud → Hybrid/CF
            return self.hybrid_recommend(user_id, n)

        elif user_data['n_transactions'] >= 1:
            # Warm-start → Content-Based + Hybrid
            return self.content_based_recommend(user_id, n)

        else:
            # Cold-start → Popularity + Demographics
            if user_data['demographics']:
                return self.demographic_recommend(user_data, n)
            else:
                return self.popularity_recommend(n)
```

**Pour les items** :
```python
# Toujours utiliser item features (25 colonnes disponibles)
model = LightFM(loss='warp')
model.fit(
    interactions=interactions,
    item_features=item_features_matrix  # ← Solution cold-start items
)
```

---

## 4. Features : Métadonnées Items/Users pour Modèle Hybride

### 🔍 Métadonnées Disponibles

#### 4.1 Features ITEMS (25 colonnes)

**Analyse Step 1** : Distribution et cardinalité

| Feature | Type | Cardinalité | Utilité RecSys | Priorité |
|---------|------|-------------|----------------|----------|
| **index_group_name** | Catégorielle | 5 | ✅ Très discriminant | ⭐⭐⭐ |
| **product_group_name** | Catégorielle | 19 | ✅ Bon équilibre | ⭐⭐⭐ |
| **garment_group_name** | Catégorielle | 21 | ✅ Features détaillées | ⭐⭐⭐ |
| **product_type_name** | Catégorielle | 132 | ⚠️ Trop granulaire | ⭐⭐ |
| **colour_group_name** | Catégorielle | 50 | ✅ Diversification | ⭐⭐ |
| **department_name** | Catégorielle | 250 | ❌ Trop granulaire | ⭐ |
| **section_name** | Catégorielle | 57 | ⚠️ Redondant | ⭐ |
| **graphical_appearance_name** | Catégorielle | 30 | ✓ Complémentaire | ⭐ |
| **perceived_colour_master_name** | Catégorielle | 20 | ✓ Utile | ⭐ |
| **detail_desc** | Texte | ~105K | ⚠️ Nécessite NLP | - |

#### 4.2 Features USERS (7 colonnes)

| Feature | Type | Couverture | Utilité RecSys | Priorité |
|---------|------|------------|----------------|----------|
| **age** | Numérique | 98.8% | ✅ Segmentation | ⭐⭐⭐ |
| **club_member_status** | Catégorielle | 99.6% | ✅ Comportement | ⭐⭐⭐ |
| **fashion_news_frequency** | Catégorielle | 98.8% | ✅ Engagement | ⭐⭐ |
| **FN** | Float | 34.8% | ❌ Trop sparse | ⭐ |
| **Active** | Float | 33.8% | ❌ Trop sparse | ⭐ |
| **postal_code** | Hash | 100% | ❌ Anonymisé | - |

### ✅ Sélection des Features

#### Features ITEMS Recommandées (One-Hot Encoding)

**Tier 1 : OBLIGATOIRES**
```python
primary_features = [
    'index_group_name',      # 5 catégories  → 5 features binaires
    'product_group_name',    # 19 catégories → 19 features binaires
    'garment_group_name',    # 21 catégories → 21 features binaires
]
# Total: 45 features binaires
```

**Rationale (Step 1)** :
- **index_group_name** : Très discriminant (Ladieswear 72%, Menswear 6%, etc.)
- **product_group_name** : Équilibre granularité/généralisation
- **garment_group_name** : Permet similarité fine (Jersey Basic, Knitwear, etc.)

**Tier 2 : OPTIONNELLES**
```python
secondary_features = [
    'colour_group_name',              # 50 catégories → 50 features
    'perceived_colour_master_name',   # 20 catégories → 20 features
    'graphical_appearance_name',      # 30 catégories → 30 features
]
# Total additionnel: 100 features binaires
```

**Trade-off** :
- ✅ Améliore diversité (couleurs différentes)
- ⚠️ Augmente dimensionnalité (risque overfitting)
- 💡 **Recommandation** : Inclure colour_group seulement

#### Features USERS Recommandées

**Configuration recommandée** :
```python
user_features = [
    'age_group',              # Discrétisation en bins
    'club_member_status',     # 3 catégories (ACTIVE, PRE-CREATE, LEFT CLUB)
    'fashion_news_frequency', # 3 catégories (Regularly, Monthly, NONE)
]
```

**Preprocessing nécessaire** :
```python
# Discrétiser age en groupes
customers['age_group'] = pd.cut(
    customers['age'],
    bins=[0, 25, 35, 45, 55, 100],
    labels=['<25', '25-35', '35-45', '45-55', '55+']
)

# Imputer valeurs manquantes
customers['club_member_status'].fillna('ACTIVE', inplace=True)
customers['fashion_news_frequency'].fillna('NONE', inplace=True)
```

**Résultat** :
- 5 age groups → 5 features
- 3 club statuses → 3 features
- 3 news frequencies → 3 features
- **Total : 11 user features**

### 📊 Encodage des Features

#### One-Hot Encoding (Articles)

```python
# Créer matrice de features items
from sklearn.preprocessing import MultiLabelBinarizer

# Sélectionner colonnes
articles_features = articles[[
    'article_id',
    'product_group_name',
    'index_group_name',
    'garment_group_name',
    'colour_group_name'  # Optionnel
]]

# One-Hot Encoding
articles_encoded = pd.get_dummies(
    articles_features,
    columns=['product_group_name', 'index_group_name',
             'garment_group_name', 'colour_group_name']
)

# Résultat: article_id + 95 features binaires (45 + 50 couleurs)
```

#### LightFM Dataset Format

```python
from lightfm.data import Dataset

# Créer dataset
dataset = Dataset()

# Fit avec features
dataset.fit(
    users=users_list,
    items=items_list,
    item_features=item_features_list,
    user_features=user_features_list
)

# Build matrices
(interactions, weights) = dataset.build_interactions(
    transactions_data
)

item_features_matrix = dataset.build_item_features(
    [(item_id, [f'product_group:{group}',
                f'index_group:{index}',
                f'garment_group:{garment}',
                f'colour:{color}'])
     for item_id, group, index, garment, color in items_data]
)

user_features_matrix = dataset.build_user_features(
    [(user_id, [f'age_group:{age}',
                f'club:{status}',
                f'news:{freq}'])
     for user_id, age, status, freq in users_data]
)
```

### 🎯 Configuration Recommandée Finale

**Pour le projet H&M** :

```python
# ITEM FEATURES (Tier 1 + couleur)
item_feature_columns = [
    'product_group_name',    # 19 features
    'index_group_name',      # 5 features
    'garment_group_name',    # 21 features
    'colour_group_name',     # 50 features
]
# Total: 95 item features binaires

# USER FEATURES (si disponibles)
user_feature_columns = [
    'age_group',              # 5 features
    'club_member_status',     # 3 features
    'fashion_news_frequency', # 3 features
]
# Total: 11 user features binaires

# TOTAL FEATURES: 106 features binaires
```

**Dimensionnalité manageable** :
- LightFM peut gérer 100+ features facilement
- Avec 50 composantes latentes : Réduction 106 → 50
- Trade-off optimal entre expressivité et complexité

### 💡 Validation des Features (A/B Testing)

**Expérimentations à mener** :

| Configuration | Item Features | User Features | Baseline |
|---------------|---------------|---------------|----------|
| **Config A** | Tier 1 (45) | Aucun | Oui |
| **Config B** | Tier 1 + couleur (95) | Aucun | Non |
| **Config C** | Tier 1 (45) | Age + Club (8) | Non |
| **Config D** | Tier 1 + couleur (95) | Tous (11) | **Recommandé** |

**Métriques de validation** :
- Precision@10, Recall@10
- Coverage (diversité)
- NDCG@10
- Cold-start performance

---

## 5. Sampling : Stratégie d'Échantillonnage

### 🔍 Analyse des Options (Step 2)

**4 stratégies testées** :

| Stratégie | Réduction | Sparsité | Avantages | Limites |
|-----------|-----------|----------|-----------|---------|
| **S1: Temporel (6 mois)** | ~65% | 99.7% | Données récentes | Sparsité encore élevée |
| **S2: Activité (≥10/≥5)** | ~75% | 99.5% | Qualité élevée | Perd données récentes |
| **S3: RFM Segments** | ~55% | 99.75% | Focus clients valeur | Moins de diversité |
| **S4: Combinée (S1+S2)** | ~90% | 99.5-99.7% | **Optimal** | - |

### ✅ Stratégie Retenue : **S4 - Combinée**

**Pipeline de sampling** :

```python
# 1. Nettoyage
transactions_clean = transactions.drop_duplicates()

# 2. Filtrage temporel
cutoff_date = max_date - timedelta(days=180)  # 6 mois
sample = transactions_clean[transactions_clean['t_dat'] >= cutoff_date]

# 3. Filtrage itératif par activité
MIN_ITEM_TRANSACTIONS = 10
MIN_USER_TRANSACTIONS = 5

def filter_by_activity(df, min_user, min_item, max_iter=10):
    for i in range(max_iter):
        initial_size = len(df)

        # Filtrer items
        item_counts = df['article_id'].value_counts()
        valid_items = item_counts[item_counts >= min_item].index
        df = df[df['article_id'].isin(valid_items)]

        # Filtrer users
        user_counts = df['customer_id'].value_counts()
        valid_users = user_counts[user_counts >= min_user].index
        df = df[df['customer_id'].isin(valid_users)]

        # Convergence
        if len(df) == initial_size:
            break

    return df

sample_final = filter_by_activity(sample, MIN_USER_TRANSACTIONS, MIN_ITEM_TRANSACTIONS)
```

### 📊 Résultats Attendus (Step 2)

**Dataset Final** :
- **Transactions** : 2-3M (vs 28.8M) → **-90% ✓**
- **Users** : 300-500K (vs 1.36M) → **-65% ✓**
- **Items** : 30-50K (vs 104.5K) → **-55% ✓**
- **Sparsité** : 99.5-99.7% (vs 99.98%) → **Amélioration 0.3-0.5 pts ✓**

**Distributions préservées** :
- Index Group : Écarts <2%
- Product Group : Écarts <3%
- Garment Group : Écarts <3%

### 🎯 Critères de Sélection Justifiés

#### Critère 1 : Temporalité (6 mois)

**Justification** :
- **Concept drift** : Les préférences mode évoluent rapidement
- **Saisonnalité** : Capturer la dernière saison complète
- **Pertinence** : Données récentes = prédictions plus précises
- **Split réaliste** : Permet validation temporelle (dernier mois = test)

**Validation Step 1** :
- 2020 représente 34.5% du dataset (2018-2020)
- Derniers 6 mois ≈ 20-25% du dataset
- Balance entre taille et récence

#### Critère 2 : Activité Minimale

**Items ≥ 10 transactions** :
- **Step 1** : Médiane item = 58 transactions
- Seuil 10 = ~17% de la médiane
- Élimine ~60% des items mais <5% des transactions
- **Rationale** : Items <10 = bruit (données insuffisantes pour CF)

**Users ≥ 5 transactions** :
- **Step 1** : Médiane user = 7 transactions
- Seuil 5 = ~70% de la médiane
- Permet calcul de similarités user-user
- **Rationale** : Minimum pour identifier préférences

#### Critère 3 : Filtrage Itératif

**Pourquoi itératif ?**
```
Itération 1: Supprimer items <10 trans → Certains users tombent <5 trans
Itération 2: Supprimer users <5 trans → Certains items tombent <10 trans
Itération 3: ...
Convergence après 3-5 itérations généralement
```

**Avantage** :
- Garantit cohérence du sample
- Tous les users ont ≥5 trans
- Tous les items ont ≥10 trans
- Pas d'incohérences

### ⚠️ Limites et Mitigations

**Limite 1 : Biais temporel**
- ❌ Perdre historique long-terme
- ✅ Mitigation : 6 mois suffisent pour capturer patterns

**Limite 2 : Perte de diversité**
- ❌ Éliminer items/users de niche
- ✅ Mitigation : Seuils bas (10/5), préserve 40-45% du catalogue

**Limite 3 : Généralisation**
- ❌ Modèle entraîné sur sample = biais
- ✅ Mitigation : Validation sur distributions (Step 2)

### 🔧 Paramètres Tunables

**Pour ajustement futur** :

```python
# Configuration de sampling
SAMPLING_CONFIG = {
    'temporal': {
        'n_months': 6,           # Ajustable : 3, 6, 9, 12
        'cutoff_mode': 'last'    # 'last' ou 'random'
    },
    'activity': {
        'min_item_trans': 10,    # Ajustable : 5, 10, 20, 50
        'min_user_trans': 5,     # Ajustable : 3, 5, 10
        'max_iterations': 10
    },
    'target': {
        'max_transactions': 5_000_000,  # Stop si taille atteinte
        'min_sparsity_reduction': 0.003  # Réduction minimale souhaitée
    }
}
```

**Trade-offs** :

| Paramètre ↑ | Taille ↓ | Sparsité ↓ | Qualité ↑ | Diversité ↓ |
|-------------|----------|------------|-----------|-------------|
| n_months | ✓ | ✓ | ⚠️ | ⚠️ |
| min_item_trans | ✓ | ✓ | ✓ | ✓ |
| min_user_trans | ✓ | ✓ | ✓ | ⚠️ |

### 📝 Validation Post-Sampling

**Checklist** :
- ✅ Taille manageable (<5M transactions)
- ✅ Sparsité réduite (<99.8%)
- ✅ Distributions préservées (écarts <5%)
- ✅ Coverage catalogue (>30% items)
- ✅ Coverage users (>30% users)
- ✅ Moyennes cohérentes (trans/user >20, trans/item >200)

**Si validation échoue** :
→ Ajuster paramètres et ré-exécuter Step 2

---

## 6. Évaluation : Métriques et Mesure du Succès

### 🔍 Défis d'Évaluation

**Spécificités du dataset H&M** :
- **Feedback implicite** : Achats binaires (pas de ratings 1-5)
- **Top-N recommendations** : Recommander liste d'items (pas prédire score)
- **Contexte temporel** : Prédire achats futurs
- **Business metrics** : Ventes, diversité, coverage

### ✅ Métriques Recommandées

#### 6.1 Métriques de Ranking (Priorité 1)

##### **Precision@K**

**Définition** :
```
Precision@K = (Nombre d'items pertinents dans top-K) / K
```

**Implémentation** :
```python
def precision_at_k(predictions, actuals, k=10):
    """
    predictions: Liste ordonnée de item_ids recommandés
    actuals: Set de item_ids réellement achetés
    """
    top_k = predictions[:k]
    hits = len(set(top_k) & set(actuals))
    return hits / k

# Exemple
predictions = [101, 205, 330, 442, ...]  # Top 10 recommandés
actuals = {205, 442, 891}                 # Achats réels
precision = precision_at_k(predictions, actuals, k=10)
# → 2/10 = 0.20 (20%)
```

**Interprétation** :
- Precision@10 = 0.10 → 1 item pertinent sur 10 recommandés
- Precision@10 = 0.20 → 2 items pertinents sur 10 (bon)
- Precision@10 = 0.30+ → Excellent

**Pourquoi important ?** :
- Mesure la **pertinence immédiate**
- Business impact direct (CTR, conversion)
- Facile à interpréter

##### **Recall@K**

**Définition** :
```
Recall@K = (Nombre d'items pertinents dans top-K) / (Total items pertinents)
```

**Implémentation** :
```python
def recall_at_k(predictions, actuals, k=10):
    top_k = predictions[:k]
    hits = len(set(top_k) & set(actuals))
    return hits / len(actuals) if len(actuals) > 0 else 0

# Exemple
predictions = [101, 205, 330, 442, ...]
actuals = {205, 442, 891, 102, 553}  # 5 achats réels
recall = recall_at_k(predictions, actuals, k=10)
# → 2/5 = 0.40 (40%)
```

**Interprétation** :
- Recall@10 = 0.20 → Capturé 20% des achats futurs
- Recall@10 = 0.50 → Capturé 50% (très bon)

**Trade-off avec Precision** :
- K ↑ → Recall ↑, Precision ↓
- Trouver équilibre optimal

##### **NDCG@K (Normalized Discounted Cumulative Gain)**

**Définition** :
```
DCG@K = Σ(i=1 to K) [rel_i / log2(i + 1)]
NDCG@K = DCG@K / IDCG@K  (normalisé par ranking idéal)
```

**Implémentation** :
```python
import numpy as np

def ndcg_at_k(predictions, actuals, k=10):
    """
    predictions: Liste ordonnée de item_ids
    actuals: Dict {item_id: relevance_score} ou Set (relevance=1)
    """
    dcg = 0.0
    for i, item_id in enumerate(predictions[:k]):
        if item_id in actuals:
            relevance = actuals.get(item_id, 1) if isinstance(actuals, dict) else 1
            dcg += relevance / np.log2(i + 2)  # i+2 car index starts at 0

    # Ideal DCG (si on avait ranking parfait)
    ideal_relevances = sorted(
        [actuals.get(item, 0) for item in predictions[:k]],
        reverse=True
    )
    idcg = sum([rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances)])

    return dcg / idcg if idcg > 0 else 0.0
```

**Avantage sur Precision/Recall** :
- **Prend en compte la position** : Item en position 1 > Item en position 10
- **Plus réaliste** : Utilisateurs regardent d'abord les premiers items
- **Norme industrielle** : Utilisé par Netflix, Spotify, YouTube

**Interprétation** :
- NDCG@10 = 0.5 → Bon
- NDCG@10 = 0.7 → Très bon
- NDCG@10 = 0.9+ → Excellent

##### **Mean Average Precision (MAP@K)**

**Définition** :
```
AP@K = (1/min(K, |actuals|)) × Σ(k=1 to K) [Precision@k × rel(k)]
MAP@K = Moyenne de AP@K sur tous les users
```

**Implémentation** :
```python
def average_precision_at_k(predictions, actuals, k=10):
    if len(actuals) == 0:
        return 0.0

    score = 0.0
    num_hits = 0.0

    for i, item_id in enumerate(predictions[:k]):
        if item_id in actuals:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actuals), k)

def mean_average_precision(all_predictions, all_actuals, k=10):
    return np.mean([
        average_precision_at_k(pred, actual, k)
        for pred, actual in zip(all_predictions, all_actuals)
    ])
```

**Avantage** :
- Combine precision et ranking
- Récompense avoir items pertinents tôt dans la liste

#### 6.2 Métriques de Diversité (Priorité 2)

##### **Catalog Coverage**

**Définition** :
```
Coverage = (Nombre d'items uniques recommandés) / (Total items catalogue)
```

**Implémentation** :
```python
def catalog_coverage(all_recommendations, total_items):
    """
    all_recommendations: Liste de listes [[item1, item2], [item3, ...]]
    total_items: Nombre total d'items dans catalogue
    """
    unique_recommended = set()
    for recs in all_recommendations:
        unique_recommended.update(recs)

    return len(unique_recommended) / total_items

# Objectif: Coverage > 10-20%
# (vs 1-2% pour popularity-based)
```

**Insight Step 1** :
- 104.5K items au total
- Après sampling : 30-50K items
- Coverage cible : >15% (4,500-7,500 items uniques recommandés)

##### **Intra-List Diversity**

**Définition** :
```
ILD = Moyenne des distances entre paires d'items dans chaque liste recommandée
```

**Implémentation** :
```python
from sklearn.metrics.pairwise import cosine_distances

def intra_list_diversity(recommendation_list, item_features_matrix):
    """
    recommendation_list: [item1, item2, ..., itemK]
    item_features_matrix: Matrice de features items (one-hot encodé)
    """
    if len(recommendation_list) < 2:
        return 0.0

    # Récupérer features des items recommandés
    features = item_features_matrix[recommendation_list]

    # Calculer distances moyennes
    distances = cosine_distances(features)

    # Moyenne des distances (hors diagonale)
    n = len(recommendation_list)
    diversity = distances.sum() / (n * (n - 1))

    return diversity

# Objectif: ILD > 0.3 (similarité cosinus)
```

**Pourquoi important ?** :
- Éviter recommander 10 t-shirts noirs identiques
- Augmenter chances de satisfaire l'utilisateur
- Exploration du catalogue

##### **Gini Index (Concentration)**

**Définition** :
```
Mesure l'inégalité de distribution des recommandations
Gini = 0 → Parfaite égalité (tous items recommandés également)
Gini = 1 → Parfaite inégalité (1 seul item recommandé)
```

**Implémentation** :
```python
def gini_index(recommendation_counts):
    """
    recommendation_counts: Dict {item_id: count}
    """
    counts = np.array(sorted(recommendation_counts.values()))
    n = len(counts)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * counts)) / (n * np.sum(counts)) - (n + 1) / n

# Objectif: Gini < 0.7
# (Popularity-based ≈ 0.9)
```

#### 6.3 Métriques Business (Priorité 3)

##### **Revenue Impact**

```python
def revenue_lift(recommended_items, actual_purchases, item_prices):
    """
    Calculer le revenu généré par les recommandations
    """
    recommended_revenue = sum([
        item_prices[item]
        for item in recommended_items
        if item in actual_purchases
    ])

    baseline_revenue = calculate_baseline_revenue()

    return (recommended_revenue - baseline_revenue) / baseline_revenue
```

##### **Click-Through Rate (CTR)**

```python
def ctr(recommendations_shown, items_clicked):
    """
    CTR = Nombre de clics / Nombre de recommandations affichées
    """
    return len(items_clicked) / len(recommendations_shown)
```

##### **Conversion Rate**

```python
def conversion_rate(items_recommended, items_purchased):
    """
    Conversion = Achats / Recommandations
    """
    return len(set(items_recommended) & set(items_purchased)) / len(items_recommended)
```

#### 6.4 Métriques LightFM (Built-in)

```python
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score

# Precision & Recall
train_precision = precision_at_k(model, train, k=10).mean()
test_precision = precision_at_k(model, test, k=10).mean()

train_recall = recall_at_k(model, train, k=10).mean()
test_recall = recall_at_k(model, test, k=10).mean()

# AUC (Area Under ROC Curve)
train_auc = auc_score(model, train).mean()
test_auc = auc_score(model, test).mean()

print(f"Train Precision@10: {train_precision:.3f}")
print(f"Test Precision@10: {test_precision:.3f}")
print(f"Test AUC: {test_auc:.3f}")
```

### 📊 Dashboard de Métriques

**Configuration recommandée** :

```python
METRICS_CONFIG = {
    'ranking': {
        'precision@k': [5, 10, 20],
        'recall@k': [5, 10, 20],
        'ndcg@k': [5, 10, 20],
        'map@k': [10]
    },
    'diversity': {
        'catalog_coverage': True,
        'intra_list_diversity': True,
        'gini_index': True
    },
    'business': {
        'revenue_lift': True,
        'ctr': True,
        'conversion_rate': True
    }
}

def evaluate_model(model, test_data, config=METRICS_CONFIG):
    results = {}

    # Ranking metrics
    for k in config['ranking']['precision@k']:
        results[f'precision@{k}'] = calculate_precision(model, test_data, k)

    for k in config['ranking']['recall@k']:
        results[f'recall@{k}'] = calculate_recall(model, test_data, k)

    for k in config['ranking']['ndcg@k']:
        results[f'ndcg@{k}'] = calculate_ndcg(model, test_data, k)

    # Diversity metrics
    if config['diversity']['catalog_coverage']:
        results['coverage'] = calculate_coverage(model, test_data)

    # ... etc

    return results
```

### 🎯 Objectifs de Performance

**Baselines attendues** :

| Modèle | Precision@10 | Recall@10 | NDCG@10 | Coverage |
|--------|--------------|-----------|---------|----------|
| **Random** | 0.001 | 0.01 | 0.05 | 100% |
| **Popularity** | 0.05-0.08 | 0.10-0.15 | 0.15-0.20 | 1-2% |
| **Content-Based** | 0.08-0.12 | 0.15-0.25 | 0.20-0.30 | 15-25% |
| **CF (SVD)** | 0.10-0.15 | 0.20-0.30 | 0.25-0.35 | 5-10% |
| **LightFM (Hybrid)** | **0.15-0.20** | **0.25-0.35** | **0.30-0.40** | **10-20%** |

**Objectifs minimaux** :
- ✅ Precision@10 > 0.10 (10% des recommandations pertinentes)
- ✅ Recall@10 > 0.15 (15% des achats futurs capturés)
- ✅ NDCG@10 > 0.25
- ✅ Coverage > 10% (vs 1-2% popularity)

**Objectifs optimaux** :
- 🎯 Precision@10 > 0.15
- 🎯 Recall@10 > 0.25
- 🎯 NDCG@10 > 0.35
- 🎯 Coverage > 15%

### 📈 Stratégie de Validation

**Split Train/Test** :

```python
# Split temporel (réaliste)
cutoff_date = transactions['t_dat'].quantile(0.9)  # 90% train, 10% test

train = transactions[transactions['t_dat'] < cutoff_date]
test = transactions[transactions['t_dat'] >= cutoff_date]

# Validation additionnelle
# Garder seulement users/items présents dans train ET test
common_users = set(train['customer_id']) & set(test['customer_id'])
common_items = set(train['article_id']) & set(test['article_id'])

test_filtered = test[
    (test['customer_id'].isin(common_users)) &
    (test['article_id'].isin(common_items))
]
```

**Cross-Validation Temporelle** :

```python
# 3-Fold temporal CV
def temporal_cv_split(transactions, n_folds=3):
    transactions_sorted = transactions.sort_values('t_dat')
    fold_size = len(transactions_sorted) // n_folds

    folds = []
    for i in range(n_folds):
        train_end = (i + 1) * fold_size
        test_end = (i + 2) * fold_size if i < n_folds - 1 else len(transactions_sorted)

        train = transactions_sorted.iloc[:train_end]
        test = transactions_sorted.iloc[train_end:test_end]

        folds.append((train, test))

    return folds

# Évaluer sur chaque fold
for i, (train, test) in enumerate(temporal_cv_split(transactions)):
    model.fit(train)
    metrics = evaluate_model(model, test)
    print(f"Fold {i+1}: Precision@10 = {metrics['precision@10']:.3f}")
```

### 🔧 Monitoring en Production

**KPIs à suivre** :

```python
# Logging des recommandations
recommendation_log = {
    'user_id': user_id,
    'recommendations': [item1, item2, ...],
    'timestamp': datetime.now(),
    'model_version': 'lightfm_v1.2',
    'features_used': ['product_group', 'index_group'],
    'prediction_scores': [0.85, 0.72, ...]
}

# Feedback tracking
feedback_log = {
    'user_id': user_id,
    'recommended_items': [...],
    'clicked_items': [...],
    'purchased_items': [...],
    'session_duration': 450,  # secondes
    'timestamp': datetime.now()
}

# Calcul métriques hebdomadaires
weekly_metrics = {
    'precision@10': 0.145,
    'ctr': 0.082,
    'conversion_rate': 0.034,
    'revenue_generated': 45_000,
    'unique_items_recommended': 8_500,
    'avg_recommendation_latency': 45  # ms
}
```

---

## 🎯 Conclusion et Recommandations Finales

### Synthèse des Décisions

| Question | Décision | Justification |
|----------|----------|---------------|
| **1. Sparsité** | LightFM Hybrid + Filtrage activité | Combine CF + métadonnées, réduit sparsité |
| **2. Longue traîne** | Content-Based + Diversification | Métadonnées permettent recommandations items rares |
| **3. Cold-Start** | LightFM avec features + Fallback | Item features (25 cols), user features (11), popularity backup |
| **4. Features** | Items: 95 (Tier1 + couleur), Users: 11 | Optimal balance expressivité/complexité |
| **5. Sampling** | S4 Combinée (Temporel + Activité) | 6 mois + ≥10/≥5, réduit 90%, préserve patterns |
| **6. Évaluation** | Precision@10, NDCG@10, Coverage | Ranking + Diversité, objectif P@10>0.15 |

### Pipeline Recommandé

```
Step 1: EDA ✅
  ↓
Step 2: Sampling (S4) ✅
  ↓
Step 3: Preprocessing
  ├─ One-Hot Encoding (items: 95 features, users: 11)
  ├─ Split temporel (90% train, 10% test)
  └─ LightFM Dataset construction
  ↓
Step 4: Baseline Models
  ├─ Popularity-based (baseline)
  └─ Content-Based (métadonnées)
  ↓
Step 5: Collaborative Filtering
  ├─ SVD (Matrix Factorization)
  └─ Évaluation (P@10, R@10, NDCG@10)
  ↓
Step 6: LightFM Hybrid (FINAL)
  ├─ Training (WARP loss, 50 components)
  ├─ Hyperparameter tuning (GridSearch)
  ├─ Évaluation complète
  └─ Comparaison vs baselines
  ↓
Step 7: Deployment
  ├─ Stratégie fallback (3 niveaux)
  ├─ Monitoring (KPIs hebdomadaires)
  └─ A/B Testing
```

### Prochaines Actions

**Immédiat** :
1. ✅ Exécuter step2-DataSampling.ipynb
2. ✅ Valider préservation distributions
3. ✅ Sauvegarder dataset échantillonné

**Court terme** :
4. 📝 Step 3: Data Preprocessing
5. 🎯 Step 4: Baseline Models
6. 🚀 Step 5: Collaborative Filtering

**Moyen terme** :
7. 🏆 Step 6: LightFM Hybrid avec hyperparameter tuning
8. 📊 Step 7: Évaluation comparative complète
9. 🎨 Step 8: Visualisation et interprétation

---

**Document créé le** : Octobre 2024
**Dernière mise à jour** : Basé sur analyses Step 1 & Step 2
