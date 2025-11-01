# Recommandations pour Améliorer la Stratégie de Sampling

## 📊 Problème Identifié

L'échantillon de 50K transactions présente:
- **93.5% des users** avec 1 seule transaction
- **57.3% des items** avec 1 seule transaction
- **Avg trans/user**: 1.07
- **Avg trans/item**: 2.06
- **Sparsité**: 99.996%

Ceci est problématique pour l'entraînement de modèles de recommandation.

## ✅ Solutions Proposées

### Option 1: Filtrer APRÈS échantillonnage (Recommandé pour 50K)

Modifier la stratégie en ajoutant un filtrage POST-échantillonnage:

```python
# Dans Step 2, après échantillonnage des 50K transactions
sample = base_sample_combine.sample(n=target_size, random_state=42).copy()

# NOUVEAU: Filtrer pour garder seulement users/items avec assez d'interactions
MIN_USER_TRANS = 2  # Au moins 2 transactions par user
MIN_ITEM_TRANS = 2  # Au moins 2 transactions par item

# Filtrer itérativement
changed = True
while changed:
    n_before = len(sample)

    # Garder users avec ≥ MIN_USER_TRANS
    user_counts = sample['customer_id'].value_counts()
    valid_users = user_counts[user_counts >= MIN_USER_TRANS].index
    sample = sample[sample['customer_id'].isin(valid_users)]

    # Garder items avec ≥ MIN_ITEM_TRANS
    item_counts = sample['article_id'].value_counts()
    valid_items = item_counts[item_counts >= MIN_ITEM_TRANS].index
    sample = sample[sample['article_id'].isin(valid_items)]

    n_after = len(sample)
    changed = (n_after < n_before)

print(f"Après filtrage: {len(sample):,} transactions conservées")
```

**Avantages:**
- ✅ Garantit au moins 2 transactions par user/item
- ✅ Permet le train/test split
- ✅ Meilleur apprentissage des patterns

**Inconvénients:**
- ❌ Réduit le nombre de transactions (peut-être ~20-30K au lieu de 50K)
- ❌ Réduit couverture users/items

### Option 2: User-Based Sampling (Recommandé pour expérimentation)

Au lieu d'échantillonner des transactions, échantillonner des USERS:

```python
# Échantillonner N users aléatoires
N_USERS = 5000  # À ajuster

sampled_users = base_sample_combine['customer_id'].unique()
sampled_users = np.random.choice(sampled_users, size=N_USERS, replace=False)

# Garder TOUTES les transactions de ces users
sample = base_sample_combine[
    base_sample_combine['customer_id'].isin(sampled_users)
].copy()
```

**Avantages:**
- ✅ Chaque user a plusieurs transactions (sa moyenne complète)
- ✅ Meilleur pour collaborative filtering
- ✅ Permet train/test split par user

**Inconvénients:**
- ❌ Taille d'échantillon variable (pas exactement 50K)
- ❌ Moins de couverture items

### Option 3: Stratified Sampling (Optimal mais complexe)

Combiner échantillonnage stratifié par activité:

```python
# Grouper users par niveau d'activité
def get_activity_level(n_trans):
    if n_trans >= 10: return 'high'
    elif n_trans >= 5: return 'medium'
    else: return 'low'

user_activity = base_sample_combine.groupby('customer_id').size()
user_activity_groups = user_activity.apply(get_activity_level)

# Échantillonner proportionnellement dans chaque groupe
# ...
```

**Avantages:**
- ✅ Préserve la distribution d'activité
- ✅ Garantit mix de users actifs et peu actifs
- ✅ Meilleure représentativité

**Inconvénients:**
- ❌ Plus complexe à implémenter
- ❌ Nécessite tuning des seuils

## 🎯 Recommandation Finale

**Pour le dataset de 50K (expérimentation):**

Utiliser **Option 2 (User-Based Sampling)** avec post-filtrage:

```python
# 1. Échantillonner 5000-8000 users actifs
N_USERS = 6000

# 2. Depuis le base_sample_combine (déjà filtré users≥5, items≥10)
sampled_users = np.random.choice(
    base_sample_combine['customer_id'].unique(),
    size=N_USERS,
    replace=False
)

# 3. Garder toutes leurs transactions
sample = base_sample_combine[
    base_sample_combine['customer_id'].isin(sampled_users)
].copy()

# 4. Filtrer items peu fréquents dans ce sample
item_counts = sample['article_id'].value_counts()
valid_items = item_counts[item_counts >= 2].index
sample = sample[sample['article_id'].isin(valid_items)]

print(f"Sample final: {len(sample):,} transactions, "
      f"{sample['customer_id'].nunique():,} users, "
      f"{sample['article_id'].nunique():,} items")
print(f"Avg trans/user: {len(sample) / sample['customer_id'].nunique():.2f}")
```

**Statistiques attendues:**
- Transactions: ~30-50K (dépend de N_USERS)
- Users: ~6000
- Items: ~8-12K
- **Avg trans/user: ~6-8** (beaucoup mieux!)
- **Avg trans/item: ~4-6** (beaucoup mieux!)
- Permet train/test split 80/20

## 📊 Comparaison des Stratégies

| Stratégie | Trans | Users | Items | Avg User | Avg Item | Train/Test | Notes |
|-----------|-------|-------|-------|----------|----------|------------|-------|
| **Actuelle (Trans sampling)** | 50K | 46K | 24K | 1.07 | 2.06 | ❌ | Trop sparse |
| **Option 1 (Post-filter)** | ~25K | ~12K | ~12K | 2.0 | 2.0 | ✅ | Perte données |
| **Option 2 (User sampling)** | ~40K | 6K | ~10K | ~6.5 | ~4.0 | ✅ | **Recommandé** |
| **Option 3 (Stratified)** | ~45K | ~8K | ~12K | ~5.6 | ~3.7 | ✅ | Complexe |

## 🔧 Implémentation

Pour tester la nouvelle stratégie:

1. Modifier **Step 2, Section 9** (création des samples)
2. Tester d'abord sur un seul sample (ex: 50K)
3. Comparer les métriques avec l'ancienne approche
4. Si satisfaisant, régénérer tous les samples

## ⚠️ Important

Après changement de stratégie:
- **Régénérer Step 2** (sampling)
- **Régénérer Step 3** (preprocessing)
- **Régénérer Step 4** (train/test split)
- Les modèles (Steps 5-7) n'ont pas besoin d'être modifiés
