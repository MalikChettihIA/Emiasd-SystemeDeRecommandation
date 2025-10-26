# ✅ Adaptation de step4-TrainTestSplitStrategy.ipynb

## 🎯 Résumé

Le notebook **step4-TrainTestSplitStrategy.ipynb** a été adapté pour :
1. Fonctionner avec la structure multi-tailles (10K/50K/100K)
2. Charger les données depuis **Step 3** (données prétraitées)
3. Sauvegarder les 3 stratégies de split dans des dossiers séparés

**Nouveau fichier** : `step4-TrainTestSplitStrategy-ADAPTED.ipynb`

---

## 📊 Changements Principaux

### ⚙️ NOUVELLE : Cellule de Configuration

Comme step3, une cellule de configuration permet de choisir la taille :

```python
# ⚙️ CONFIGURATION : Choisir la taille du sample
# IMPORTANT: Utiliser la même taille que Step 3

SAMPLE_SIZE = '50K'  # ⭐ Recommandé
```

**IMPORTANT** : Utilisez la **même taille** que dans Step 3 pour garantir la cohérence !

### 📁 Chemins Adaptés

| Aspect | Avant ❌ | Après ✅ |
|--------|---------|---------|
| **Source** | `data/sampled/` (Step 2) | `data/processed/{SAMPLE_SIZE}/` (Step 3) |
| **Destination** | `data/processed/` | `data/processed/{SAMPLE_SIZE}/splits/` |

**Pourquoi charger depuis Step 3 ?**
- ✅ Dataset LightFM déjà fitted avec mappings
- ✅ Matrice d'interactions déjà construite
- ✅ Transactions DataFrame disponible (nécessaire pour split temporel)
- ✅ Cohérence garantie avec les données prétraitées

### 📂 Structure de Sauvegarde

Les **3 stratégies** de split sont maintenant sauvegardées dans des dossiers séparés :

```
data/processed/{SAMPLE_SIZE}/splits/
├── temporal/              ← ⭐ Stratégie recommandée
│   ├── train_interactions.npz
│   ├── test_interactions.npz
│   ├── train_data.csv
│   └── test_data.csv
├── random/
│   ├── train_interactions.npz
│   └── test_interactions.npz
├── userbased/
│   ├── train_interactions.npz
│   ├── test_interactions.npz
│   ├── train_data.csv
│   └── test_data.csv
└── split_metadata.json    ← Métadonnées globales
```

---

## 🎯 Les 3 Stratégies de Split

### 1️⃣ Split Temporel (Temporal) ⭐ **RECOMMANDÉ**

**Principe** :
- Train : 90% des transactions les **plus anciennes**
- Test : 10% des transactions les **plus récentes**

**Avantages** :
- ✅ **Le plus réaliste** : simule la prédiction du futur
- ✅ Pas de data leakage temporel
- ✅ Cohérent avec déploiement en production
- ✅ Standard de l'industrie pour RecSys

**Inconvénients** :
- ⚠️ Cold-start partiel (nouveaux users/items dans test)
- ⚠️ Nécessite filtrage pour garder users/items communs

**Exemple** :
```
Train : 2020-03-26 → 2020-08-30 (5.8M interactions)
Test  : 2020-08-31 → 2020-09-22 (498K interactions)
```

**Quand utiliser** :
- Pour l'évaluation finale du modèle
- Pour simuler un scénario de production réaliste
- **Recommandé pour Step 5-7**

---

### 2️⃣ Split Aléatoire (Random)

**Principe** :
- Train : 80% des interactions de chaque user (aléatoire)
- Test : 20% des interactions de chaque user (aléatoire)
- Implémentation : `random_train_test_split()` de LightFM

**Avantages** :
- ✅ Pas de cold-start (tous users/items dans les 2 sets)
- ✅ Distribution équilibrée
- ✅ Simple à implémenter

**Inconvénients** :
- ⚠️ Moins réaliste (data leakage temporel possible)
- ⚠️ Ne simule pas vraiment la prédiction du futur

**Exemple** :
```
Train : 5.2M interactions (80%)
Test  : 1.3M interactions (20%)
```

**Quand utiliser** :
- Pour debug/développement rapide
- Pour vérifier que le modèle apprend (pas de cold-start)
- **Utile pour tests rapides**

---

### 3️⃣ Split par Utilisateur (User-based)

**Principe** :
- Train : 80% des users avec **toutes** leurs interactions
- Test : 20% des users avec **toutes** leurs interactions

**Avantages** :
- ✅ Simule recommandation pour **nouveaux utilisateurs**
- ✅ Test de capacité de généralisation
- ✅ Évalue robustesse au cold-start

**Inconvénients** :
- ⚠️ Cold-start **complet** (très difficile)
- ⚠️ Performances attendues plus basses
- ⚠️ Nécessite modèle hybride avec features

**Exemple** :
```
Train : 348K users (80%) → 5.2M interactions
Test  : 87K users (20%) → 1.3M interactions
Users test ABSENTS de train : 100%
```

**Quand utiliser** :
- Pour tester la robustesse au cold-start
- Pour évaluer modèles hybrides avec user features
- **Évaluation complémentaire**

---

## 📝 Instructions d'Utilisation

### 1️⃣ Prérequis

**IMPORTANT** : Step 3 doit être terminé avec la même taille !

```bash
# Vérifier que Step 3 est terminé
ls data/processed/50K/
```

**Fichiers attendus** :
- `lightfm_dataset.pkl`
- `user_item_matrix.npz`
- `transactions.csv`
- `item_features_matrix.npz`
- `user_features_matrix.npz`
- etc.

### 2️⃣ Ouvrir le Notebook

```bash
cd /Users/malikchettih/Projects/Emiasd-Projects/Emiasd-SystemeDeRecommandation/6.Projects
jupyter notebook step4-TrainTestSplitStrategy-ADAPTED.ipynb
```

### 3️⃣ Vérifier la Configuration

Dans la cellule de configuration :

```python
SAMPLE_SIZE = '50K'  # ⚙️ Même taille que Step 3 !
```

**Validation automatique** :
- ✅ Vérifie que `data/processed/50K/` existe
- ✅ Vérifie que les fichiers essentiels sont présents
- ❌ Erreur si Step 3 pas terminé

### 4️⃣ Exécuter le Notebook

- `Kernel → Restart & Run All`
- Temps estimé : **5-10 minutes** (selon taille)

### 5️⃣ Vérifier les Résultats

```bash
ls -lh data/processed/50K/splits/
```

**Structure attendue** :
```
splits/
├── temporal/
│   ├── train_interactions.npz    (~20-40 MB)
│   ├── test_interactions.npz     (~2-5 MB)
│   ├── train_data.csv
│   └── test_data.csv
├── random/
│   ├── train_interactions.npz
│   └── test_interactions.npz
├── userbased/
│   ├── train_interactions.npz
│   ├── test_interactions.npz
│   ├── train_data.csv
│   └── test_data.csv
└── split_metadata.json
```

**Vérifier les métadonnées** :
```bash
cat data/processed/50K/splits/split_metadata.json
```

### 6️⃣ Si Tout Est OK

```bash
# Backup
mv step4-TrainTestSplitStrategy.ipynb step4-TrainTestSplitStrategy-OLD.ipynb

# Activer
mv step4-TrainTestSplitStrategy-ADAPTED.ipynb step4-TrainTestSplitStrategy.ipynb
```

---

## 🎯 Workflow Recommandé

### Pour Steps 4-6 : Utiliser 50K avec Temporal Split

```python
# Step 4
SAMPLE_SIZE = '50K'
# Résultat : data/processed/50K/splits/temporal/

# Step 5 : Charger le split temporal
train_inter = load_npz('data/processed/50K/splits/temporal/train_interactions.npz')
test_inter = load_npz('data/processed/50K/splits/temporal/test_interactions.npz')
```

**Pourquoi Temporal ?**
- Le plus réaliste pour évaluation
- Simule scénario de production
- Standard de l'industrie

### Utilisation des Autres Splits

**Random Split** : Pour debug rapide
```python
# Si vous voulez juste tester que le modèle apprend
train_inter = load_npz('data/processed/50K/splits/random/train_interactions.npz')
test_inter = load_npz('data/processed/50K/splits/random/test_interactions.npz')
```

**User-based Split** : Pour évaluation cold-start
```python
# Pour tester robustesse au cold-start (modèle hybride)
train_inter = load_npz('data/processed/50K/splits/userbased/train_interactions.npz')
test_inter = load_npz('data/processed/50K/splits/userbased/test_interactions.npz')
```

---

## 📊 Comparaison des Stratégies

### Résultats Attendus (50K sample)

| Stratégie | Train | Test | Sparsité Train | Sparsité Test | Cold-start |
|-----------|-------|------|----------------|---------------|------------|
| **Temporal** | ~45K | ~5K | 99.95% | 99.99% | Partiel (6% users nouveaux) |
| **Random** | ~40K | ~10K | 99.96% | 99.99% | Minimal |
| **User-based** | ~40K | ~10K | 99.96% | 99.99% | Complet (100% users nouveaux) |

### Performance Attendue des Modèles

| Stratégie | Precision@10 (estimée) | Difficulté | Réalisme |
|-----------|------------------------|------------|----------|
| **Temporal** | ~0.10-0.15 | Moyenne | ⭐⭐⭐⭐⭐ |
| **Random** | ~0.15-0.20 | Faible | ⭐⭐ |
| **User-based** | ~0.05-0.08 | Élevée | ⭐⭐⭐ |

---

## 🔧 Cohérence entre Steps

### Pipeline Complet

```
Step 2 : Créer 10K, 50K, 100K
    ↓
Step 3 : SAMPLE_SIZE = '50K' → data/processed/50K/
    ↓
Step 4 : SAMPLE_SIZE = '50K' → data/processed/50K/splits/
    ↓
Step 5 : Charger splits/temporal/ → Baseline models
    ↓
Step 6 : SAMPLE_SIZE = '50K' → Hyperparameter tuning
    ↓
Step 7 : SAMPLE_SIZE = '100K' → Modèle final
```

### ⚠️ Règle d'Or

**Utilisez la MÊME taille** dans Steps 3, 4, 5, 6 :

```python
# Step 3
SAMPLE_SIZE = '50K'  # Preprocessing

# Step 4
SAMPLE_SIZE = '50K'  # Split (MÊME taille !)

# Step 5
SAMPLE_SIZE = '50K'  # Baseline (MÊME taille !)

# Step 6
SAMPLE_SIZE = '50K'  # Tuning (MÊME taille !)
```

Puis pour le modèle final :
```python
# Step 7
SAMPLE_SIZE = '100K'  # Modèle final
```

---

## 📈 Analyse des Splits

### Ce que step4 fait

Pour chaque stratégie, step4 :

1. **Crée le split** selon la stratégie choisie
2. **Analyse les distributions** :
   - Nombre d'interactions train/test
   - Sparsité
   - Cold-start problem
3. **Construit les matrices LightFM**
4. **Sauvegarde** tout dans le dossier approprié

### Visualisations Incluses

Le notebook génère 4 visualisations :
1. 📊 Nombre d'interactions par stratégie (train vs test)
2. 📉 Sparsité par stratégie
3. 🥧 Proportion train/test (Temporal)
4. ⚠️ Niveau de cold-start par stratégie

---

## 🎓 Métadonnées Sauvegardées

Le fichier `split_metadata.json` contient :

```json
{
  "split_date": "2025-10-22 20:47:00",
  "sample_size": "50K",
  "chosen_strategy": "Temporal Split",
  "train_ratio": 0.9,
  "cutoff_date": "2020-08-31",
  "temporal": {
    "train": {
      "n_interactions": 45000,
      "n_users": 12000,
      "n_items": 8000,
      "sparsity": 0.9995,
      "period": {"start": "2020-03-26", "end": "2020-08-30"}
    },
    "test": {
      "n_interactions": 5000,
      "n_users": 11000,
      "n_items": 7500,
      "period": {"start": "2020-08-31", "end": "2020-09-22"}
    },
    "cold_start": {
      "users_common_pct": 93.8,
      "items_common_pct": 94.2
    }
  },
  ...
}
```

---

## 🚀 Prochaines Étapes

Après step4, vous avez 3 splits prêts pour l'entraînement.

### Step 5 : Baseline Models

```python
# Charger le split temporal (recommandé)
SAMPLE_SIZE = '50K'
SPLIT_STRATEGY = 'temporal'

train_inter = load_npz(f'data/processed/{SAMPLE_SIZE}/splits/{SPLIT_STRATEGY}/train_interactions.npz')
test_inter = load_npz(f'data/processed/{SAMPLE_SIZE}/splits/{SPLIT_STRATEGY}/test_interactions.npz')

# Charger features (depuis Step 3)
item_features = load_npz(f'data/processed/{SAMPLE_SIZE}/item_features_matrix.npz')
user_features = load_npz(f'data/processed/{SAMPLE_SIZE}/user_features_matrix.npz')

# Entraîner modèle
from lightfm import LightFM

model = LightFM(loss='warp', no_components=30)
model.fit(
    interactions=train_inter,
    item_features=item_features,
    user_features=user_features,
    epochs=10
)

# Évaluer
from lightfm.evaluation import precision_at_k
precision = precision_at_k(model, test_inter, item_features=item_features, k=10).mean()
```

---

## ✅ Checklist de Validation

Avant de continuer à Step 5, vérifier :

- [ ] step4 s'exécute sans erreur
- [ ] Dossier `data/processed/50K/splits/` créé
- [ ] 3 sous-dossiers : `temporal/`, `random/`, `userbased/`
- [ ] Chaque dossier contient les matrices `.npz`
- [ ] `split_metadata.json` contient les bonnes statistiques
- [ ] Temporal split a ~90% train / ~10% test
- [ ] Cold-start partiel dans temporal (~94% users/items communs)
- [ ] SAMPLE_SIZE cohérent entre Step 3 et Step 4

---

## 💡 Conseils

### 1. Utilisez Temporal pour l'Évaluation Finale

C'est la stratégie **recommandée** par le projet et la documentation LightFM.

### 2. Utilisez Random pour Debug

Si vous voulez juste vérifier que le modèle apprend sans problème de cold-start.

### 3. Utilisez User-based pour Tester la Robustesse

Évaluez la capacité de votre modèle hybride à recommander pour nouveaux users.

### 4. Gardez la Cohérence de Taille

```
Steps 3-6 : SAMPLE_SIZE = '50K' (expérimentation)
Step 7    : SAMPLE_SIZE = '100K' (modèle final)
```

### 5. Documentez Vos Choix

Dans votre rapport final :

> **Stratégie de Split**
>
> Nous avons implémenté et comparé 3 stratégies de split :
> 1. Temporal Split (90/10)
> 2. Random Split (80/20)
> 3. User-based Split (80/20)
>
> Nous avons choisi le **Temporal Split** pour l'évaluation car il simule le scénario le plus réaliste (prédiction du futur) et évite le data leakage temporel. Cette stratégie est cohérente avec les standards de l'industrie pour les systèmes de recommandation.

---

## 🎉 Avantages de Cette Approche

1. ✅ **Flexibilité** : 3 stratégies disponibles pour différents besoins
2. ✅ **Séparation** : Chaque stratégie dans son propre dossier
3. ✅ **Comparaisons** : Facile de comparer l'impact de la stratégie
4. ✅ **Réutilisabilité** : Splits créés une fois, utilisables dans tous les steps suivants
5. ✅ **Documentation** : Métadonnées complètes pour chaque split
6. ✅ **Cohérence** : Structure alignée avec step2 et step3

---

**🚀 Bon split de données !**
