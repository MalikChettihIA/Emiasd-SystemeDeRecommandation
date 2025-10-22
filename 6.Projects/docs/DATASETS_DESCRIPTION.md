# Description des Datasets H&M

Ce document décrit en détail les trois fichiers de données du dataset H&M utilisés pour construire notre système de recommandation.

---

## Vue d'ensemble

Le dataset H&M est composé de trois fichiers CSV principaux :

| Fichier | Lignes | Colonnes | Taille | Description |
|---------|---------|----------|--------|-------------|
| **transactions_train.csv** | 31,788,324 | 5 | ~3.7 GB | Historique des achats |
| **articles.csv** | 105,542 | 25 | ~30 MB | Catalogue produits avec métadonnées |
| **customers.csv** | 1,371,980 | 7 | ~85 MB | Profils clients |

**Période couverte** : 20 septembre 2018 → 22 septembre 2020 (~2 ans)

---

## 1. Dataset Transactions (transactions_train.csv)

### Description
Ce fichier contient l'historique complet des transactions d'achat des clients H&M sur une période de 2 ans. Chaque ligne représente un article acheté par un client à une date donnée.

### Dimensions
- **31,788,324 lignes** (transactions)
- **5 colonnes**
- **1,362,281 clients uniques**
- **104,547 articles uniques**

### Structure des Colonnes

| Colonne | Type | Description | Exemple |
|---------|------|-------------|---------|
| **t_dat** | date (YYYY-MM-DD) | Date de la transaction | `2020-09-22` |
| **customer_id** | string (hash) | Identifiant unique du client (anonymisé, 64 caractères hexadécimaux) | `000058a12d5b43e67d...` |
| **article_id** | integer | Identifiant unique de l'article acheté | `663713001` |
| **price** | float | Prix de l'article acheté (normalisé entre 0 et 1) | `0.050831` |
| **sales_channel_id** | integer | Canal de vente : `1` = magasin physique, `2` = en ligne | `2` |

### Caractéristiques Importantes

#### Interactions Binaires
- Les données représentent uniquement les **achats effectués** (valeur implicite = 1)
- **Pas de ratings explicites** (comme des notes 1-5)
- Impossible de distinguer entre "article vu mais non acheté" et "article jamais vu"

#### Distribution Temporelle
- **2018** : 4,411,262 transactions (13.9%)
- **2019** : 16,396,930 transactions (51.6%)
- **2020** : 10,980,132 transactions (34.5%)

#### Sparsité
- **Sparsité : 99.98%**
- Densité : 0.02%
- Chaque utilisateur n'achète qu'une infime fraction du catalogue

#### Duplicates
- Le dataset contient des **duplicates** (environ 2,974,905 lignes, ~9.4%)
- Duplicates identifiés sur : même date, client, article, prix et canal
- **Recommandation** : Supprimer les duplicates lors du preprocessing
- Après déduplication : environ **28,813,419 transactions**

#### Distribution des Interactions

**Par Utilisateur :**
- Moyenne : ~23 transactions/utilisateur
- Médiane : ~7 transactions/utilisateur
- Min : 1 | Max : >1000 (power users)
- **Effet longue traîne** : ~30% des utilisateurs génèrent 80% des transactions

**Par Article :**
- Moyenne : ~304 transactions/article
- Médiane : ~58 transactions/article
- Distribution très déséquilibrée (quelques blockbusters, beaucoup d'articles peu vendus)

#### Patterns Temporels
- **Saisonnalité** : Pics observés en septembre (rentrée) et décembre (fêtes)
- **Jour de la semaine** : Plus d'achats en milieu et fin de semaine
- **Tendance** : Croissance jusqu'à mi-2019, puis stabilisation/légère baisse en 2020 (COVID-19)

---

## 2. Dataset Articles (articles.csv)

### Description
Ce fichier contient le catalogue complet des articles disponibles chez H&M avec leurs métadonnées détaillées (type, couleur, département, description, etc.). C'est la source principale pour le **Content-Based Filtering**.

### Dimensions
- **105,542 lignes** (articles)
- **25 colonnes** (features)
- **416 valeurs manquantes** (uniquement dans `detail_desc`)

### Structure des Colonnes

#### Identifiants

| Colonne | Type | Description | Cardinalité | Exemple |
|---------|------|-------------|-------------|---------|
| **article_id** | int64 | Identifiant unique de l'article | 105,542 | `108775015` |
| **product_code** | int64 | Code produit (plusieurs articles peuvent partager le même code = variantes) | ~47,000 | `108775` |

#### Caractéristiques Produit

| Colonne | Type | Description | Cardinalité | Exemple |
|---------|------|-------------|-------------|---------|
| **prod_name** | string | Nom commercial du produit | ~45,000 | `Strap top` |
| **product_type_no** | int64 | Numéro du type de produit | 132 | `253` |
| **product_type_name** | string | Nom du type de produit | 132 | `Vest top` |
| **product_group_name** | string | **Catégorie principale** | 19 | `Garment Upper body` |

**Valeurs de `product_group_name`** (19 catégories) :
- Garment Upper body
- Garment Lower body
- Garment Full body
- Accessories
- Shoes
- Underwear
- Swimwear
- Socks & Tights
- Nightwear
- Bags
- Cosmetic
- Furniture
- Items
- Stationery
- Interior textile
- Fun
- Garment and Shoe care
- Underwear/nightwear
- Unknown

#### Caractéristiques Visuelles

| Colonne | Type | Description | Cardinalité | Exemple |
|---------|------|-------------|-------------|---------|
| **graphical_appearance_no** | int64 | Numéro de l'apparence graphique | 30 | `1010016` |
| **graphical_appearance_name** | string | Type d'apparence (motif) | 30 | `Solid`, `Stripe`, `Print` |
| **colour_group_code** | int64 | Code de la couleur | 50 | `9` |
| **colour_group_name** | string | **Nom de la couleur** | 50 | `Black`, `White`, `Blue` |

**Top 10 Couleurs** :
1. Black (11,849 articles)
2. White (7,770)
3. Light Beige (6,618)
4. Dark Blue (5,515)
5. Grey (4,581)
6. Blue (4,016)
7. Beige (3,759)
8. Red (2,793)
9. Pink (2,528)
10. Light Pink (2,359)

#### Perception des Couleurs

| Colonne | Type | Description | Cardinalité | Exemple |
|---------|------|-------------|-------------|---------|
| **perceived_colour_value_id** | int64 | ID de la valeur perçue | 8 | `4` |
| **perceived_colour_value_name** | string | Valeur perçue (clair/foncé) | 8 | `Dark`, `Light`, `Medium` |
| **perceived_colour_master_id** | int64 | ID couleur maître | 20 | `5` |
| **perceived_colour_master_name** | string | Couleur maître perçue | 20 | `Black`, `White`, `Blue` |

#### Organisation Hiérarchique

| Colonne | Type | Description | Cardinalité | Exemple |
|---------|------|-------------|-------------|---------|
| **department_no** | int64 | Numéro du département | 299 | `1676` |
| **department_name** | string | Nom du département | 250 | `Jersey Basic` |
| **index_code** | string | Code de l'index (catégorie large) | 10 | `A`, `B`, `C` |
| **index_name** | string | Nom de l'index | 10 | `Ladieswear`, `Lingeries/Tights` |
| **index_group_no** | int64 | Numéro du groupe d'index | 5 | `1` |
| **index_group_name** | string | **Groupe principal** | 5 | `Ladieswear`, `Menswear`, `Sport` |

**Valeurs de `index_group_name`** (5 catégories) :
- **Ladieswear** (Vêtements femme) - 76,066 articles (72.1%)
- **Baby/Children** (Bébé/Enfant) - 11,253 articles (10.7%)
- **Divided** (Jeunes femmes) - 10,426 articles (9.9%)
- **Menswear** (Vêtements homme) - 7,205 articles (6.8%)
- **Sport** (Sportswear) - 592 articles (0.6%)

#### Section et Groupe de Vêtements

| Colonne | Type | Description | Cardinalité | Exemple |
|---------|------|-------------|-------------|---------|
| **section_no** | int64 | Numéro de section | 57 | `16` |
| **section_name** | string | Nom de la section | 57 | `Womens Everyday Basics` |
| **garment_group_no** | int64 | Numéro du groupe de vêtement | 21 | `1002` |
| **garment_group_name** | string | **Nom du groupe de vêtement** | 21 | `Jersey Basic`, `Knitwear` |

**Valeurs de `garment_group_name`** (21 catégories) :
- Jersey Basic
- Jersey Fancy
- Knitwear
- Trousers
- Trousers Denim
- Shorts
- Skirts
- Dresses Ladies
- Dresses/Skirts girls
- Blouses
- Shirts
- Outdoor
- Under-, Nightwear
- Accessories
- Shoes
- Socks and Tights
- Swimwear
- Special Offers
- Dressed
- Woven/Jersey/Knitted mix Baby
- Unknown

#### Description Textuelle

| Colonne | Type | Description | Valeurs Manquantes | Exemple |
|---------|------|-------------|-------------------|---------|
| **detail_desc** | string | Description détaillée du produit | 416 (0.39%) | `Jersey top with narrow shoulder straps.` |

---

## 3. Dataset Customers (customers.csv)

### Description
Ce fichier contient les profils des clients H&M avec des informations démographiques et comportementales. Ces features sont utiles pour la personnalisation et l'analyse de segments.

### Dimensions
- **1,371,980 lignes** (clients)
- **7 colonnes**
- **Taux de valeurs manquantes élevé** pour certaines colonnes

### Structure des Colonnes

| Colonne | Type | Description | Valeurs Manquantes | Exemple |
|---------|------|-------------|-------------------|---------|
| **customer_id** | string | Identifiant unique du client (hash 64 caractères) | 0 (0%) | `00000dbacae5abe5e...` |
| **FN** | float64 | Feature non documentée (probablement Fashion News opt-in numérique) | 895,050 (65.2%) | `1.0` |
| **Active** | float64 | Indicateur si le client est actif | 907,576 (66.2%) | `1.0` |
| **club_member_status** | string | Statut de membre du club fidélité | 6,062 (0.4%) | `ACTIVE`, `PRE-CREATE`, `LEFT CLUB` |
| **fashion_news_frequency** | string | Fréquence de réception des newsletters mode | 16,011 (1.2%) | `Regularly`, `Monthly`, `NONE` |
| **age** | float64 | Âge du client | 15,861 (1.2%) | `49.0` |
| **postal_code** | string | Code postal (hashé pour anonymisation) | 0 (0%) | `52043ee2162cf5aa...` |

### Analyse Détaillée des Features

#### customer_id
- **Unique** : 1,371,980 clients
- Format : Hash hexadécimal de 64 caractères
- Aucune valeur manquante

#### FN (Fashion News Opt-in)
- **Valeurs** : 0.0, 1.0, NaN
- **Manquant** : 65.2%
- Probablement : 1 = inscrit aux nouvelles mode, 0 = non inscrit

#### Active
- **Valeurs** : 0.0, 1.0, NaN
- **Manquant** : 66.2%
- Indique si le client est actuellement actif

#### club_member_status
- **Valeurs** : `ACTIVE`, `PRE-CREATE`, `LEFT CLUB`, NaN
- **Manquant** : 0.4% seulement
- Distribution :
  - **ACTIVE** : ~95% des clients avec données
  - **PRE-CREATE** : ~4%
  - **LEFT CLUB** : ~1%

#### fashion_news_frequency
- **Valeurs** : `Regularly`, `Monthly`, `NONE`, NaN
- **Manquant** : 1.2%
- Distribution :
  - **NONE** : Majorité (clients ne souhaitant pas recevoir de newsletters)
  - **Monthly** : Newsletters mensuelles
  - **Regularly** : Newsletters fréquentes

#### age
- **Type** : Numérique continu
- **Manquant** : 1.2%
- **Statistiques** :
  - Moyenne : ~38 ans
  - Médiane : ~36 ans
  - Min : 15 ans
  - Max : 99 ans (possibles outliers)
  - Q1 : 27 ans
  - Q3 : 49 ans
- **Distribution** : Majorité entre 20-50 ans, pic autour de 25-35 ans

#### postal_code
- **Format** : Hash hexadécimal de 64 caractères
- **Aucune valeur manquante**
- **Particularité** : Un code postal a 120,303 clients (probablement code pour adresses manquantes ou centre de distribution)

### Limitations et Considérations

⚠️ **Taux de Valeurs Manquantes Élevé**
- Plus de 65% des clients n'ont pas de données pour `FN` et `Active`
- Ces colonnes sont peu exploitables telles quelles
- Recommandation : Se concentrer sur `age`, `club_member_status`, `fashion_news_frequency`

⚠️ **Anonymisation**
- `customer_id` et `postal_code` sont hashés (pas de géolocalisation possible)
- Impossible de relier à d'autres sources de données externes

⚠️ **Biais Potentiel**
- Les clients dans le dataset sont ceux qui ont acheté entre 2018-2020
- Ne représente pas tous les clients H&M (seulement les actifs)

---

## Utilisation des Datasets pour le Système de Recommandation

### 1. Collaborative Filtering
**Données utilisées** : `transactions_train.csv`
- Matrice User-Item construite à partir de `customer_id` × `article_id`
- Valeurs binaires : 1 si achat, 0 sinon
- Techniques : SVD, Matrix Factorization avec LightFM

### 2. Content-Based Filtering
**Données utilisées** : `articles.csv`
- Features principales pour similarité :
  - `product_group_name` (19 catégories)
  - `index_group_name` (5 catégories)
  - `garment_group_name` (21 catégories)
  - `colour_group_name` (50 couleurs)
- Encodage : One-Hot Encoding → 45+ features binaires
- Similarité : Cosinus entre vecteurs d'articles

### 3. Hybrid Model (LightFM)
**Données utilisées** : `transactions_train.csv` + `articles.csv` + `customers.csv`
- Interactions : matrice user-item
- Item features : métadonnées articles (product_group, index_group, garment_group)
- User features : age, club_member_status, fashion_news_frequency
- Avantage : Résout le cold-start pour nouveaux articles avec features

### 4. Segmentation RFM (Optionnel)
**Données utilisées** : `transactions_train.csv`
- Recency : Jours depuis dernier achat
- Frequency : Nombre d'achats
- Monetary : Montant total dépensé
- Segmentation clients : Champions, Loyaux, À Risque, Dormants, etc.

---

## Stratégie de Preprocessing Recommandée

### 1. Nettoyage Transactions
```python
# Supprimer duplicates
transactions = transactions.drop_duplicates(
    subset=['t_dat', 'customer_id', 'article_id', 'price', 'sales_channel_id']
)

# Convertir date
transactions['t_dat'] = pd.to_datetime(transactions['t_dat'])

# Filtrer période spécifique (ex: 6 derniers mois pour entraînement)
transactions = transactions[transactions['t_dat'] >= '2020-03-01']
```

### 2. Enrichissement Articles
```python
# Features pour Content-Based
articles_features = articles[[
    'article_id',
    'product_group_name',
    'index_group_name',
    'garment_group_name'
]]

# One-Hot Encoding
articles_encoded = pd.get_dummies(
    articles_features,
    columns=['product_group_name', 'index_group_name', 'garment_group_name']
)
```

### 3. Filtrage Customers
```python
# Garder seulement colonnes avec peu de NaN
customers_clean = customers[['customer_id', 'age', 'club_member_status', 'fashion_news_frequency']]

# Imputer age manquant avec médiane
customers_clean['age'].fillna(customers_clean['age'].median(), inplace=True)

# Imputer catégorielles avec mode
customers_clean['club_member_status'].fillna('ACTIVE', inplace=True)
customers_clean['fashion_news_frequency'].fillna('NONE', inplace=True)
```

### 4. Échantillonnage (Datasets volumineux)
```python
# Garder articles avec minimum 10 achats
article_counts = transactions['article_id'].value_counts()
popular_articles = article_counts[article_counts >= 10].index
transactions = transactions[transactions['article_id'].isin(popular_articles)]

# Garder utilisateurs avec minimum 5 achats
user_counts = transactions['customer_id'].value_counts()
active_users = user_counts[user_counts >= 5].index
transactions = transactions[transactions['customer_id'].isin(active_users)]
```

---

## Métriques du Dataset Final (Après Nettoyage Typique)

| Métrique | Valeur |
|----------|--------|
| Transactions uniques | ~28.8M |
| Clients actifs (5+ achats) | ~500K |
| Articles populaires (10+ ventes) | ~50K |
| Sparsité | ~99.9% |
| Période recommandée pour training | Mars - Août 2020 |
| Période recommandée pour test | Septembre 2020 |

---

## Ressources Complémentaires

- **Compétition Kaggle** : [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
- **Documentation LightFM** : [https://lyst.github.io/lightfm/docs/home.html](https://lyst.github.io/lightfm/docs/home.html)
- **Paper de référence** : [LightFM: A Hybrid Recommendation Algorithm](https://arxiv.org/abs/1507.08439)

---

**Dernière mise à jour** : Octobre 2024
