# Solution pour V1.4 - Fix Évaluation du Modèle Hybride

## 🔴 Problèmes Identifiés

### 1. user_features manquant lors de l'évaluation
Le modèle est entraîné avec `user_features` mais elles ne sont pas passées lors de l'évaluation.

### 2. Désalignement des matrices (CRITIQUE)
Deux systèmes de mapping différents créés :
- Mapping LightFM : `dataset.fit()` → crée `item_features_matrix`, `user_features_matrix`
- Mapping manuel : `user_id_map`, `item_id_map` → crée `train_matrix`, `test_matrix`

**Résultat** : Les indices ne correspondent pas entre les matrices !

---

## ✅ Solution : Remplacer la Cellule de Split Temporel

**REMPLACER** la cellule qui commence par :
```python
# --- Étape 3 : Prétraitement et Construction de la Matrice ---
```

**PAR** :

```python
print("=" * 80)
print("CRÉATION DES MATRICES TRAIN/TEST AVEC L'API LIGHTFM")
print("=" * 80)

# ============================================================================
# APPROCHE CORRECTE : Utiliser l'API LightFM pour créer train/test
# ============================================================================

print(f"\n🔄 Split temporel des données...")

# 1. Trier les transactions par date
transactions_sorted = transactions.sort_values('t_dat')

# 2. Split 80/20
split_idx = int(len(transactions_sorted) * 0.8)
train_data = transactions_sorted.iloc[:split_idx].copy()
test_data = transactions_sorted.iloc[split_idx:].copy()

print(f"   ✓ Train: {len(train_data):,} transactions")
print(f"   ✓ Test: {len(test_data):,} transactions")

# 3. Binariser (garder une seule occurrence par paire user-item)
train_data = train_data.drop_duplicates(subset=['customer_id', 'article_id'])
test_data = test_data.drop_duplicates(subset=['customer_id', 'article_id'])

print(f"\n🔄 Après binarisation:")
print(f"   ✓ Train: {len(train_data):,} interactions uniques")
print(f"   ✓ Test: {len(test_data):,} interactions uniques")

# 4. Filtrer test pour ne garder que les interactions nouvelles
train_pairs = set(zip(train_data['customer_id'], train_data['article_id']))
test_pairs = list(zip(test_data['customer_id'], test_data['article_id']))
mask = [pair not in train_pairs for pair in test_pairs]
test_data = test_data[mask]

print(f"\n🔄 Après filtrage des doublons train/test:")
print(f"   ✓ Test: {len(test_data):,} interactions nouvelles")

# 5. Construire les matrices avec l'API LightFM (utilise le dataset déjà créé)
print(f"\n🔄 Construction des matrices avec LightFM...")

(train_interactions, _) = dataset.build_interactions(
    ((row['customer_id'], row['article_id']) for _, row in train_data.iterrows())
)

(test_interactions, _) = dataset.build_interactions(
    ((row['customer_id'], row['article_id']) for _, row in test_data.iterrows())
)

print(f"   ✓ train_interactions: {train_interactions.shape} - {train_interactions.nnz:,} nnz")
print(f"   ✓ test_interactions: {test_interactions.shape} - {test_interactions.nnz:,} nnz")

# 6. Convertir en CSR pour compatibilité
train_interactions = train_interactions.tocsr()
test_interactions = test_interactions.tocsr()

print(f"\n✅ Matrices créées avec l'API LightFM")
print(f"\n🔍 Vérification d'alignement:")
print(f"   train_interactions.shape[0] (users): {train_interactions.shape[0]:,}")
print(f"   user_features_matrix.shape[0] (users): {user_features_matrix.shape[0]:,}")
print(f"   train_interactions.shape[1] (items): {train_interactions.shape[1]:,}")
print(f"   item_features_matrix.shape[0] (items): {item_features_matrix.shape[0]:,}")

assert train_interactions.shape[0] == user_features_matrix.shape[0], "Désalignement users !"
assert train_interactions.shape[1] == item_features_matrix.shape[0], "Désalignement items !"

print(f"✅ Alignement parfait garanti !")
```

---

## ✅ Solution : Remplacer la Cellule d'Entraînement

**REMPLACER** la cellule qui commence par :
```python
print("=" * 80)
print("ENTRAÎNEMENT MODÈLE HYBRIDE")
```

**PAR** :

```python
print("=" * 80)
print("ENTRAÎNEMENT MODÈLE HYBRIDE")
print("=" * 80)

# Créer le modèle hybride
hybrid_model = LightFM(
    loss='warp',
    no_components=55,
    learning_rate=0.00394802143257261,
    item_alpha=1.93e-08,
    user_alpha=1.89e-08,
    random_state=42
)

print(f"\n🔄 Entraînement en cours...")
print(f"   Avec item_features ET user_features (LightFM)")

import time
start_time = time.time()

# Entraînement avec les features LightFM
hybrid_model.fit(
    interactions=train_interactions,      # ← Matrices LightFM !
    item_features=item_features_matrix,   # ← Features LightFM (alignées) !
    user_features=user_features_matrix,   # ← Features LightFM (alignées) !
    epochs=10,
    num_threads=4,
    verbose=True
)

training_time = time.time() - start_time
print(f"\n✅ Entraînement terminé en {training_time:.1f}s")
```

---

## ✅ Solution : Corriger la Cellule d'Évaluation

**REMPLACER** la boucle d'évaluation dans la cellule qui contient :
```python
for k in K_VALUES:
    prec = precision_at_k(hybrid_model, test_matrix, k=k,
```

**PAR** :

```python
print(f"\n2️⃣  HYBRID MODEL (avec item + user features LightFM):")
print(f"{'K':<6} | {'Precision@K':<13} | {'Recall@K':<13} | {'AUC':<10}")
print("-" * 55)

for k in K_VALUES:
    # ✅ CORRECTION : Passer AUSSI user_features lors de l'évaluation
    prec = precision_at_k(
        hybrid_model,
        test_interactions,              # ← Matrice LightFM
        k=k,
        train_interactions=train_interactions,  # ← Matrice LightFM
        item_features=item_features_matrix,     # ← Features LightFM
        user_features=user_features_matrix,     # ✅ AJOUTÉ !
        num_threads=4
    ).mean()

    rec = recall_at_k(
        hybrid_model,
        test_interactions,
        k=k,
        train_interactions=train_interactions,
        item_features=item_features_matrix,
        user_features=user_features_matrix,     # ✅ AJOUTÉ !
        num_threads=4
    ).mean()

    auc = auc_score(
        hybrid_model,
        test_interactions,
        train_interactions=train_interactions,
        item_features=item_features_matrix,
        user_features=user_features_matrix,     # ✅ AJOUTÉ !
        num_threads=4
    ).mean()

    results_comparison['hybrid'][k] = {
        'precision': prec,
        'recall': rec,
        'auc': auc
    }
    print(f"{k:<6} | {prec:<13.4f} | {rec:<13.4f} | {auc:<10.4f}")

print(f"\n✅ Évaluation terminée")
```

---

## 📋 Résumé des Changements

| Avant | Après |
|-------|-------|
| Mapping manuel (`user_id_map`) | API LightFM (`dataset.build_interactions()`) |
| `train_matrix`, `test_matrix` | `train_interactions`, `test_interactions` |
| Pas de `user_features` dans évaluation | `user_features=user_features_matrix` partout |
| Désalignement des indices | Alignement garanti par LightFM |

---

## ✅ Vérification

Après ces modifications, vous devriez voir :

```
🔍 Vérification d'alignement:
   train_interactions.shape[0] (users): 9,851
   user_features_matrix.shape[0] (users): 9,851  ✓
   train_interactions.shape[1] (items): 7,865
   item_features_matrix.shape[0] (items): 7,865  ✓
✅ Alignement parfait garanti !

[Epoch 0] ...
[Epoch 9] ...
✅ Entraînement terminé en XX.Xs

2️⃣  HYBRID MODEL (avec item + user features LightFM):
K      | Precision@K   | Recall@K      | AUC
-------------------------------------------------------
5      | 0.XXXX        | 0.XXXX        | 0.XXXX
10     | 0.XXXX        | 0.XXXX        | 0.XXXX
20     | 0.XXXX        | 0.XXXX        | 0.XXXX

✅ Évaluation terminée
```

Aucune erreur `ValueError` !

---

## 🎯 Pourquoi ça fonctionne maintenant

1. **Un seul système de mapping** : LightFM gère tout
2. **Alignement garanti** : Les indices correspondent entre toutes les matrices
3. **Cohérence entraînement/évaluation** : Les mêmes features sont passées partout
4. **API LightFM utilisée correctement** : Pas de mélange avec des mappings manuels

---

**Note importante** : Ne créez JAMAIS vos propres mappings `user_id_map` ou `item_id_map` quand vous utilisez l'API LightFM. Laissez `Dataset.fit()` et `dataset.build_interactions()` faire le travail !
