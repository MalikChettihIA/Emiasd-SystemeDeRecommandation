# Guide d'Utilisation du Notebook Unifié

## 📘 H&M-Fashion-RecommenderSystem-LightFM.ipynb

---

## ✅ Le Notebook a été Généré Automatiquement

Le notebook unifié a été créé en agrégeant les 8 notebooks individuels (step1 à step8) avec les optimisations suivantes:

### Améliorations par Rapport aux 8 Notebooks Séparés

✅ **Élimination des redondances:**
- Suppression de ~21 cellules de sauvegarde/rechargement
- Suppression des sections "Configuration" répétées
- Suppression des vérifications de fichiers processed

✅ **Workflow continu:**
- Les objets restent en mémoire (dataset, matrices, modèles)
- Pas de save/load intermédiaires
- Exécution fluide de bout en bout

✅ **Réduction de taille:**
- 203 cellules (vs ~500 dans les 8 notebooks)
- Réduction de 60% du nombre de cellules

---

## 🚀 Avant d'Exécuter le Notebook

### 1. Vérifications Préalables

**Données requises:**
```bash
cd 6.Projects/data/
ls -lh

# Vous devez avoir:
# - transactions_train.csv (~3.5 GB)
# - articles.csv (~36 MB)
# - customers.csv (~207 MB)
```

**Bibliothèques requises:**
```bash
pip install lightfm pandas numpy matplotlib seaborn scipy scikit-learn
```

### 2. Configuration

Le notebook est paramétré via la **Section 0.2**. Les paramètres par défaut sont:

```python
SAMPLE_SIZE = 50000  # 50K transactions
MIN_USER_TRANSACTIONS = 5
MIN_ITEM_TRANSACTIONS = 10
SPLIT_STRATEGY = 'random'  # Options: 'temporal', 'random', 'userbased'
N_EPOCHS = 30
```

**Modifiez ces paramètres selon vos besoins avant d'exécuter!**

---

## ⚠️ Points d'Attention

### Problèmes Potentiels et Solutions

#### 1. **Section 4: Reconstruction de Matrice**

**Problème:** Dans step4 original, la cellule 8 reconstruit `full_interactions` alors qu'elle a déjà été chargée.

**Solution appliquée:** Le script a conservé cette cellule. Si vous voulez optimiser:
- Cherchez la cellule avec `dataset.build_interactions()` dans Section 4
- Commentez ou supprimez cette reconstruction
- Utilisez directement `full_interactions` de la Section 3

#### 2. **Section 3: Chargement de Données**

**Problème:** step3 chargeait depuis les fichiers sampled de step2.

**Solution appliquée:** Le script a modifié pour utiliser directement les variables en mémoire:
- `transactions_sampled` (créé dans Section 2)
- `articles_sampled` (créé dans Section 2)
- `customers_sampled` (créé dans Section 2)

**Vérifiez** que ces variables sont bien utilisées au lieu de `pd.read_csv()`.

#### 3. **Mémoire**

Avec SAMPLE_SIZE=50K, le notebook devrait utiliser:
- ~2-3 GB de RAM pendant l'EDA
- ~4-5 GB pendant le grid search

**Si vous manquez de mémoire:**
- Réduire `SAMPLE_SIZE` à 10000
- Réduire `GRID_SEARCH_PARAMS` (moins de combinaisons)
- Fermer d'autres applications

---

## 📋 Ordre d'Exécution Recommandé

### Option A: Exécution Complète (Recommandée)

```
Kernel → Restart & Run All
```

**Temps estimé:** ~20-30 minutes (avec SAMPLE_SIZE=50K)

### Option B: Exécution Section par Section

1. **Section 0:** Configuration (2 min)
2. **Section 1:** EDA (5 min)
3. **Section 2:** Sampling (2 min)
4. **Section 3:** Preprocessing (3 min)
5. **Section 4:** Split Strategies (2 min)
6. **Section 5:** Model Training (5 min)
7. **Section 6:** Grid Search (10-15 min) ⏳ Le plus long!
8. **Section 7:** Evaluation (2 min)
9. **Section 8:** Hybrid Model (3 min)
10. **Section 9:** Conclusions (manuel)

---

## 🔍 Checklist de Vérification

Après l'exécution, vérifiez:

### Section 1 (EDA)
- [ ] Dataset complet chargé (31M transactions)
- [ ] Visualisations affichées
- [ ] Statistiques correctes

### Section 2 (Sampling)
- [ ] `base_sample_combine` créé (~259K transactions)
- [ ] `transactions_sampled` créé (50K transactions)
- [ ] Statistiques: avg_trans/user, avg_trans/item

### Section 3 (Preprocessing)
- [ ] `dataset` (LightFM Dataset) créé
- [ ] `user_item_matrix` créé (50,000 nnz)
- [ ] `item_features_matrix` créé
- [ ] `user_features_matrix` créé

### Section 4 (Split)
- [ ] 3 stratégies implémentées
- [ ] Comparaison visualisée
- [ ] Variables créées: `train_interactions`, `test_interactions`

### Section 5 (Training)
- [ ] 4 modèles entraînés (BPR, WARP, Item-based, Hybrid)
- [ ] Métriques calculées pour chaque modèle
- [ ] Comparaison affichée

### Section 6 (Grid Search)
- [ ] Grid search exécuté sur meilleur modèle
- [ ] `best_model` créé
- [ ] `best_params` affichés

### Section 7 (Evaluation)
- [ ] Métriques complètes calculées
- [ ] Visualisations générées
- [ ] Exemples de recommandations

### Section 8 (Hybrid)
- [ ] Modèle hybride final
- [ ] Comparaison avec baselines

---

## 🐛 Dépannage

### Erreur: "File not found: transactions_train.csv"

**Solution:**
```bash
# Vérifier le chemin
cd 6.Projects/
ls data/transactions_train.csv

# Si absent, vérifier dans data/zip/
# Décompresser si nécessaire
```

### Erreur: "NameError: name 'transactions_sampled' is not defined"

**Cause:** Section 3 essaie d'utiliser `transactions_sampled` mais Section 2 n'a pas été exécutée.

**Solution:** Exécuter Section 2 avant Section 3.

### Erreur: "LightFM not found"

**Solution:**
```bash
pip install lightfm
```

### Warning: "UserWarning: LightFM was compiled without OpenMP"

**Impact:** Entraînement plus lent (single-thread au lieu de multi-thread)

**Solution:** Acceptable pour ce projet (différence de quelques secondes). Pour optimiser:
```bash
# Recompiler LightFM avec OpenMP support (avancé)
pip uninstall lightfm
CC="gcc-11" pip install lightfm --no-binary lightfm
```

### Grid Search trop long

**Solution:** Réduire les paramètres dans Section 0.2:
```python
GRID_SEARCH_PARAMS = {
    'no_components': [30, 50],  # Au lieu de [10, 30, 50, 100]
    'learning_rate': [0.05],     # Au lieu de [0.01, 0.05, 0.1]
    'item_alpha': [1e-5],        # Au lieu de [1e-6, 1e-5, 1e-4]
    'user_alpha': [1e-5]         # Au lieu de [1e-6, 1e-5, 1e-4]
}
```

---

## 📝 Compléter la Section 9

Après exécution réussie, compléter manuellement:

```markdown
## 9.1 Synthèse des Résultats

- **Dataset:** H&M Fashion
- **Sample size:** [Indiquer la taille utilisée]
- **Stratégie de split choisie:** [random/temporal/userbased]
- **Meilleur modèle:** [BPR/WARP/Hybrid]
- **Performances:**
  - Precision@10: [valeur]
  - Recall@10: [valeur]
  - AUC: [valeur]

## 9.2 Limitations

[Discuter des limitations: sparsité, cold-start, etc.]

## 9.3 Pistes d'Amélioration

[Proposer des améliorations: features additionnelles, architectures, etc.]

## 9.4 Recommandations pour le Déploiement

[Recommandations pratiques pour la production]
```

---

## 🎯 Résultat Final Attendu

Un notebook complet qui:
- ✅ Se lit de manière fluide et cohérente
- ✅ S'exécute de bout en bout sans erreur
- ✅ Produit tous les résultats et visualisations
- ✅ Démontre la maîtrise de LightFM et des systèmes de recommandation
- ✅ Est prêt pour un rendu académique ou professionnel

**Temps d'exécution total:** ~20-30 minutes

**Bonne chance! 🚀**
