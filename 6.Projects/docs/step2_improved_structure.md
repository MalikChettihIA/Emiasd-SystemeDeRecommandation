# Proposition : Structure Améliorée pour step2-DataSampling.ipynb

## 🎯 Objectif

Créer une structure chronologique, claire et optimale qui :
1. ✅ Compare les stratégies d'échantillonnage
2. ✅ Choisit la meilleure stratégie (décision explicite)
3. ✅ Expérimente avec différentes tailles
4. ✅ Sauvegarde PLUSIEURS tailles pour comparer les modèles plus tard
5. ✅ Évite de re-générer les samples à chaque expérience

---

## 📚 Structure Proposée

### **Section 1-3 : Chargement et Nettoyage** ✅ (Garder tel quel)
- Chargement des données
- Nettoyage (duplicates, dates invalides, etc.)
- Résultat : `transactions_clean`

---

### **Section 4 : Comparaison des 4 Stratégies d'Échantillonnage** ✅ (Améliorer)

**Objectif** : Comparer 4 approches sur UN sample de taille fixe (ex: 100K)

#### Stratégies à comparer :

1. **S1 - User Sampling** : Échantillonner N users, prendre toutes leurs transactions
2. **S2 - Item Sampling** : Échantillonner N items, prendre toutes leurs transactions
3. **S3 - Interaction Sampling** : Échantillonner N transactions aléatoirement
4. **S4 - Stratégie Combinée** : Users actifs (≥5) + Items populaires (≥10) → échantillonner

#### Métriques de comparaison :

Pour chaque stratégie (sur 100K transactions) :
- ✅ Nombre final de transactions (après filtrage min activité)
- ✅ Nombre users, items
- ✅ Avg trans/user, avg trans/item
- ✅ Sparsité
- ✅ Préservation des distributions (vs dataset complet)

#### Résultat attendu :

```
COMPARAISON DES STRATÉGIES (100K transactions cible)
========================================================
Stratégie          Trans_Final  Users  Items  Avg_User  Avg_Item  Sparsity
S1 - User          0            0      0      -         -         100%      ❌
S2 - Item          0            0      0      -         -         100%      ❌
S3 - Interaction   0            0      0      -         -         100%      ❌
S4 - Combinée      100,000      20K    12K    5.0       8.3       99.96%    ✅
```

---

### **Section 5 : DÉCISION de la Stratégie** ⭐ (NOUVELLE - Explicite)

**Contenu** :

```markdown
## 5. Décision : Quelle Stratégie Choisir ?

### 📊 Résultats de la Comparaison

D'après Section 4 :
- S1, S2, S3 : Produisent des résultats VIDES après filtrage → ❌ REJETÉES
- S4 (Combinée) : Préserve les données et la qualité → ✅ CHOISIE

### ✅ Stratégie Retenue : **Combinée (S4)**

**Méthode** :
1. Filtrer TOUT le dataset pour :
   - Users actifs : ≥5 transactions
   - Items populaires : ≥10 transactions
2. Créer `base_sample_combine` (~259K transactions)
3. Échantillonner différentes tailles DEPUIS ce dataset pré-filtré

**Avantages** :
- ✅ Garantit des données denses même pour petits samples
- ✅ Chaque user a un historique significatif (≥5)
- ✅ Chaque item a du signal (≥10)
- ✅ Évite les samples vides

**Justification** : Basé sur le document de référence
`hm-fashion-recommendation-pipeline_notebooks_02_Echantillonnage_Donnees.ipynb`

### 🚀 Suite : Expérimenter avec différentes TAILLES (Section 6)
```

---

### **Section 6 : Création du Dataset Pré-filtré (Stratégie S4)** ⭐ (NOUVELLE)

**Code** :

```python
print("=" * 80)
print("CRÉATION DU DATASET PRÉ-FILTRÉ (STRATÉGIE COMBINÉE)")
print("=" * 80)

# Identifier users actifs (≥5 transactions)
user_activity = transactions_clean['customer_id'].value_counts()
users_actifs = user_activity[user_activity >= 5].index
print(f"\n👥 Users actifs (≥5 transactions): {len(users_actifs):,}")

# Identifier items populaires (≥10 transactions)
item_activity = transactions_clean['article_id'].value_counts()
items_populaires = item_activity[item_activity >= 10].index
print(f"📦 Items populaires (≥10 transactions): {len(items_populaires):,}")

# Filtrer le dataset complet
base_sample_combine = transactions_clean[
    (transactions_clean['customer_id'].isin(users_actifs)) &
    (transactions_clean['article_id'].isin(items_populaires))
].copy()

print(f"\n✅ Dataset pré-filtré créé:")
print(f"   • Transactions: {len(base_sample_combine):,}")
print(f"   • Users: {base_sample_combine['customer_id'].nunique():,}")
print(f"   • Items: {base_sample_combine['article_id'].nunique():,}")
print(f"   • Avg trans/user: {len(base_sample_combine)/base_sample_combine['customer_id'].nunique():.2f}")
print(f"   • Avg trans/item: {len(base_sample_combine)/base_sample_combine['article_id'].nunique():.2f}")

print(f"\n🎯 Ce dataset sera utilisé pour créer les différentes tailles (Section 7)")
```

---

### **Section 7 : Expérimentation des TAILLES** ✅ (Garder mais clarifier)

**Objectif** : Tester différentes tailles d'échantillons DEPUIS `base_sample_combine`

**Tailles à tester** : [1K, 10K, 50K, 100K, 259K (max)]

**Résultat attendu** :

```
COMPARAISON DES TAILLES (depuis Stratégie Combinée)
====================================================
Taille   Trans    Users   Items   Avg_User  Avg_Item  Temps_Estimé
1K       1,000    ~500    ~400    ~2.0      ~2.5      ~20 sec       ⚠️
10K      10,000   ~3,000  ~2,500  ~3.3      ~4.0      ~1 min        ⚠️
50K      50,000   ~12,000 ~8,000  ~4.2      ~6.2      ~2-3 min      ✅
100K     100,000  ~20,000 ~12,000 ~5.0      ~8.3      ~5-6 min      ✅
259K     259,486  47,691  20,051  ~5.4      ~13.0     ~15-20 min    ⚠️
```

**Analyse** :
- 1K-10K : Trop petits, avg trans/user < 4 → Modèle faible
- **50K : OPTIMAL** pour expérimentation (bon compromis qualité/vitesse)
- 100K : Meilleur modèle mais 2x plus lent
- 259K : Meilleur possible mais très lent

---

### **Section 8 : DÉCISION des Tailles à Sauvegarder** ⭐ (NOUVELLE - Explicite)

**Contenu** :

```markdown
## 8. Décision : Quelles Tailles Sauvegarder ?

### 🎯 Stratégie Recommandée : Sauvegarder 3 TAILLES

Plutôt que de sauvegarder qu'un seul sample (50K), on va sauvegarder 3 tailles
pour permettre des comparaisons de performances plus tard :

#### 1️⃣ **10K - "Debug"**
- **Usage** : Tests rapides, debug du code, vérification pipeline
- **Temps** : ~30-60 sec d'entraînement
- **Qualité** : Modèle faible mais suffisant pour validation technique
- **Sauvegarde** : `data/sampled/10K/`

#### 2️⃣ **50K - "Expérimentation"** ⭐ PRINCIPAL
- **Usage** : Hyperparameter tuning, comparaison de modèles, itérations rapides
- **Temps** : ~2-3 min d'entraînement
- **Qualité** : Bon modèle, données suffisantes
- **Sauvegarde** : `data/sampled/50K/`

#### 3️⃣ **100K - "Production"**
- **Usage** : Modèle final avec les meilleurs hyperparamètres trouvés
- **Temps** : ~5-6 min d'entraînement
- **Qualité** : Meilleur modèle réaliste
- **Sauvegarde** : `data/sampled/100K/`

### 📊 Workflow Complet

```
Step 3-5 : Développement rapide
    ↓
Utiliser 10K (debug rapide) et 50K (expérimentation)
    ↓
Trouver les meilleurs hyperparamètres sur 50K
    ↓
Step 6 : Modèle final
    ↓
Ré-entraîner sur 100K avec meilleurs hyperparamètres
    ↓
Comparaison finale : "50K vs 100K : +X% performance, Xmin de plus"
```

### ✅ Avantages de cette approche

1. ✅ **Flexibilité** : On peut choisir la taille selon le besoin
2. ✅ **Comparaison** : On peut mesurer l'impact de la taille des données
3. ✅ **Rapidité** : Pas besoin de re-générer les samples
4. ✅ **Documentation** : Chaque sample a ses métadonnées
5. ✅ **Pédagogique** : Montre la compréhension du trade-off temps/qualité
```

---

### **Section 9 : Sauvegarde des 3 Tailles** ⭐ (NOUVELLE - Remplace Section 8)

**Code** :

```python
import json
import os

print("=" * 80)
print("SAUVEGARDE DE 3 TAILLES POUR COMPARAISONS FUTURES")
print("=" * 80)

# Configuration : Tailles à sauvegarder
SIZES_TO_SAVE = {
    '10K': 10_000,
    '50K': 50_000,
    '100K': 100_000
}

BASE_PATH = 'data/sampled/'

# Pour chaque taille
for size_name, size_value in SIZES_TO_SAVE.items():
    print(f"\n{'='*80}")
    print(f"📦 CRÉATION SAMPLE {size_name}")
    print(f"{'='*80}")

    # Vérifier si assez de données
    if len(base_sample_combine) >= size_value:
        sample = base_sample_combine.sample(n=size_value, random_state=42).copy()
    else:
        print(f"⚠️  Taille demandée ({size_value:,}) > disponible ({len(base_sample_combine):,})")
        print(f"   → Utilisation du maximum disponible")
        sample = base_sample_combine.copy()

    # Statistiques
    n_trans = len(sample)
    n_users = sample['customer_id'].nunique()
    n_items = sample['article_id'].nunique()
    avg_user = n_trans / n_users
    avg_item = n_trans / n_items
    sparsity = 1 - (n_trans / (n_users * n_items))

    print(f"\n📊 Statistiques {size_name}:")
    print(f"   • Transactions: {n_trans:,}")
    print(f"   • Users: {n_users:,}")
    print(f"   • Items: {n_items:,}")
    print(f"   • Avg trans/user: {avg_user:.2f}")
    print(f"   • Avg trans/item: {avg_item:.2f}")
    print(f"   • Sparsité: {sparsity:.4%}")

    # Créer dossier
    sample_path = os.path.join(BASE_PATH, size_name)
    os.makedirs(sample_path, exist_ok=True)

    print(f"\n💾 Sauvegarde dans {sample_path}/")

    # Sauvegarder transactions
    sample.to_csv(f'{sample_path}/transactions_sampled.csv', index=False)
    print(f"   ✓ transactions_sampled.csv")

    # Sauvegarder articles
    articles_in = articles[articles['article_id'].isin(sample['article_id'])].copy()
    articles_in.to_csv(f'{sample_path}/articles_sampled.csv', index=False)
    print(f"   ✓ articles_sampled.csv ({len(articles_in):,} articles)")

    # Sauvegarder customers
    customers_in = customers[customers['customer_id'].isin(sample['customer_id'])].copy()
    customers_in.to_csv(f'{sample_path}/customers_sampled.csv', index=False)
    print(f"   ✓ customers_sampled.csv ({len(customers_in):,} customers)")

    # Métadonnées
    metadata = {
        'sample_size': size_name,
        'creation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'strategy': 'Combinée (users≥5 + items≥10)',
        'target_size': size_value,
        'actual_size': int(n_trans),
        'statistics': {
            'n_transactions': int(n_trans),
            'n_users': int(n_users),
            'n_items': int(n_items),
            'avg_trans_user': float(avg_user),
            'avg_trans_item': float(avg_item),
            'sparsity': float(sparsity)
        },
        'period': {
            'start': sample['t_dat'].min().strftime('%Y-%m-%d'),
            'end': sample['t_dat'].max().strftime('%Y-%m-%d')
        },
        'usage': {
            '10K': 'Debug rapide, validation technique',
            '50K': 'Expérimentation, hyperparameter tuning',
            '100K': 'Modèle final avec meilleurs hyperparamètres'
        }[size_name]
    }

    with open(f'{sample_path}/sampling_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✓ sampling_metadata.json")

print(f"\n{'='*80}")
print("✅ STEP 2 TERMINÉ - 3 SAMPLES CRÉÉS")
print(f"{'='*80}")

print(f"\n📦 Structure créée:")
print(f"""
data/sampled/
├── 10K/
│   ├── transactions_sampled.csv
│   ├── articles_sampled.csv
│   ├── customers_sampled.csv
│   └── sampling_metadata.json
├── 50K/
│   ├── transactions_sampled.csv
│   ├── articles_sampled.csv
│   ├── customers_sampled.csv
│   └── sampling_metadata.json
└── 100K/
    ├── transactions_sampled.csv
    ├── articles_sampled.csv
    ├── customers_sampled.csv
    └── sampling_metadata.json
""")

print(f"\n🎯 Utilisation recommandée:")
print(f"   • Step 3-4 : Utiliser 50K (expérimentation rapide)")
print(f"   • Step 5 : Hyperparameter tuning sur 50K")
print(f"   • Step 6 : Modèle final sur 100K avec meilleurs params")
print(f"   • Comparaison finale : Performance 50K vs 100K")

print(f"\n🚀 Prochaine étape: Step 3 - Data Preprocessing (avec 50K)")
```

---

## 📊 Comparaison : Avant vs Après

### ❌ Structure Actuelle (Confuse)

```
Section 4-5 : Compare 4 stratégies
Section 6   : Valide S4
Section 7   : Expérimente tailles (quelle stratégie ??)
Section 8   : Sauvegarde 50K (pourquoi 50K ??)

Problèmes :
- Décisions implicites
- Un seul sample sauvegardé
- Pas de justifications
- Chronologie confuse
```

### ✅ Structure Proposée (Claire)

```
Section 4   : Compare 4 stratégies (100K chacune)
Section 5   : DÉCISION EXPLICITE → S4 Combinée
Section 6   : Crée base_sample_combine (S4)
Section 7   : Expérimente TAILLES avec S4
Section 8   : DÉCISION EXPLICITE → Sauvegarder 10K, 50K, 100K
Section 9   : Sauvegarde les 3 tailles

Avantages :
- Chronologie linéaire
- Décisions explicites et justifiées
- 3 samples pour flexibilité
- Prêt pour comparaisons de performance
```

---

## 🎯 Bénéfices pour Votre Projet

### 1. **Clarté Pédagogique**
Votre professeur voit que vous :
- ✅ Comprenez les différentes stratégies
- ✅ Prenez des décisions justifiées
- ✅ Comprenez le trade-off temps/qualité

### 2. **Flexibilité Expérimentale**
Vous pouvez :
- ✅ Débugger rapidement avec 10K
- ✅ Expérimenter vite avec 50K
- ✅ Valider le modèle final avec 100K

### 3. **Comparaisons Rigoureuses**
Dans votre rapport final :
```
"Nous avons testé 3 tailles de datasets :
- 10K : Precision@10 = 0.08, Temps = 30s
- 50K : Precision@10 = 0.15, Temps = 2min30
- 100K : Precision@10 = 0.18, Temps = 5min40

Conclusion : 50K offre le meilleur compromis qualité/vitesse
pour l'expérimentation. 100K améliore de +20% mais coûte 2x plus cher."
```

### 4. **Pas de Re-génération**
- ✅ Samples créés UNE FOIS
- ✅ Réutilisables à volonté
- ✅ Métadonnées documentent chaque sample

---

## 🚀 Action Recommandée

Je peux **restructurer complètement step2** avec cette nouvelle approche. Voulez-vous que je :

1. ✅ **Option A - Restructuration complète** : Réorganiser step2 selon cette structure (Sections 4-9)
2. ⚠️ **Option B - Patch rapide** : Garder la structure actuelle mais corriger Section 8 pour sauvegarder 3 tailles

**Ma recommandation** : **Option A** - La restructuration vaut le coup, vous aurez un notebook beaucoup plus clair et professionnel.

Qu'en pensez-vous ? 🤔
