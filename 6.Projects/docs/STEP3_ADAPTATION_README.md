# ✅ Adaptation de step3-DataPreprocessing.ipynb

## 🎯 Résumé

Le notebook **step3-DataPreprocessing.ipynb** a été adapté pour fonctionner avec la nouvelle structure multi-tailles créée dans **Step 2**.

**Nouveau fichier** : `step3-DataPreprocessing-ADAPTED.ipynb`

---

## 📊 Changements Principaux

### ⚙️ NOUVELLE : Cellule de Configuration

Une nouvelle cellule a été ajoutée au **début du notebook** pour choisir la taille du sample :

```python
# ⚙️ CONFIGURATION : Choisir la taille du sample
# Valeurs possibles : '10K', '50K', '100K'

SAMPLE_SIZE = '50K'  # ⭐ Recommandé pour expérimentation
```

**Avantages** :
- ✅ Choix facile de la taille sans modifier le code partout
- ✅ Un seul changement pour traiter une autre taille
- ✅ Validation automatique que le sample existe

### 📁 Chemins Adaptés

| Aspect | Avant ❌ | Après ✅ |
|--------|---------|---------|
| **Source** | `data/sampled/` | `data/sampled/{SAMPLE_SIZE}/` |
| **Destination** | `data/processed/` | `data/processed/{SAMPLE_SIZE}/` |
| **Metadata** | `sampling_stats.json` | `sampling_metadata.json` |

**Exemple** :
- Pour `SAMPLE_SIZE = '50K'` :
  - Source : `data/sampled/50K/`
  - Destination : `data/processed/50K/`

### 📋 Structure de Fichiers Résultante

Après exécution de step3 avec différentes tailles :

```
data/processed/
├── 10K/
│   ├── lightfm_dataset.pkl
│   ├── user_item_matrix.npz
│   ├── item_features_matrix.npz
│   ├── user_features_matrix.npz
│   ├── transactions.csv
│   └── ... (autres fichiers)
├── 50K/
│   ├── lightfm_dataset.pkl
│   ├── user_item_matrix.npz
│   ├── item_features_matrix.npz
│   ├── user_features_matrix.npz
│   ├── transactions.csv
│   └── ... (autres fichiers)
└── 100K/
    ├── lightfm_dataset.pkl
    ├── user_item_matrix.npz
    ├── item_features_matrix.npz
    ├── user_features_matrix.npz
    ├── transactions.csv
    └── ... (autres fichiers)
```

---

## 📝 Instructions d'Utilisation

### 1️⃣ Ouvrir le Nouveau Notebook

```bash
cd /Users/malikchettih/Projects/Emiasd-Projects/Emiasd-SystemeDeRecommandation/6.Projects
jupyter notebook step3-DataPreprocessing-ADAPTED.ipynb
```

### 2️⃣ Choisir la Taille (Optionnel)

Dans la **cellule de configuration** (après l'introduction) :

```python
SAMPLE_SIZE = '50K'  # ⭐ Recommandé

# Ou choisir une autre taille :
# SAMPLE_SIZE = '10K'  # Pour debug rapide
# SAMPLE_SIZE = '100K' # Pour modèle final
```

### 3️⃣ Exécuter le Notebook

**Option A** : Exécuter tout d'un coup
- Menu : `Kernel → Restart & Run All`

**Option B** : Exécuter cellule par cellule
- Pour comprendre chaque étape

### 4️⃣ Vérifier les Résultats

Après exécution complète, vérifier :

```bash
ls -lh data/processed/50K/
```

**Fichiers attendus** (14 fichiers) :

```
lightfm_dataset.pkl              # Dataset LightFM avec mappings
user_item_matrix.npz             # Matrice d'interactions complète
item_features_matrix.npz         # Features items
user_features_matrix.npz         # Features users

user_id_mapping.pkl              # Mappings
item_id_mapping.pkl
user_features_mapping.pkl
item_features_mapping.pkl

transactions.csv                 # DataFrames pour Step 4
articles_filtered.csv
customers_filtered.csv

item_feature_columns.json        # Listes de features
user_feature_columns.json
all_item_features.json
all_user_features.json

preprocessing_metadata.json      # Métadonnées complètes
```

### 5️⃣ Si Tout Est OK

```bash
# Backup de l'ancien
mv step3-DataPreprocessing.ipynb step3-DataPreprocessing-OLD.ipynb

# Activer le nouveau
mv step3-DataPreprocessing-ADAPTED.ipynb step3-DataPreprocessing.ipynb
```

---

## 🎯 Workflow Recommandé

### Pour Steps 3-6 : Utiliser 50K

```python
SAMPLE_SIZE = '50K'  # Dans step3, step4, step5, step6
```

**Pourquoi ?**
- ⚡ Preprocessing rapide (~2-3 minutes)
- 📊 Données suffisantes pour expérimentation
- 🔄 Itérations rapides pour hyperparameter tuning

### Pour Step 7 : Utiliser 100K

```python
SAMPLE_SIZE = '100K'  # Pour le modèle final
```

**Pourquoi ?**
- Une fois les meilleurs hyperparamètres trouvés sur 50K
- Ré-entraîner sur 100K pour le modèle de production
- Comparer : "50K vs 100K : +X% performance"

### Pour Debug : Utiliser 10K

```python
SAMPLE_SIZE = '10K'  # Pour tests rapides
```

**Quand ?**
- Tester rapidement que le code fonctionne
- Valider le pipeline sans attendre
- Debug d'erreurs

---

## 📊 Comparaison des Tailles

| Taille | Temps Preprocessing | Taille Fichiers | Quand Utiliser |
|--------|---------------------|-----------------|----------------|
| **10K** | ~30 sec | ~5 MB | Debug, validation rapide |
| **50K** | ~2-3 min | ~30 MB | **Expérimentation principale** ⭐ |
| **100K** | ~5-6 min | ~60 MB | Modèle final, production |

---

## 🔄 Changement de Taille

Pour traiter une **autre taille**, il suffit de :

1. Modifier la cellule de configuration :
   ```python
   SAMPLE_SIZE = '100K'  # Au lieu de '50K'
   ```

2. Relancer le notebook (`Kernel → Restart & Run All`)

3. Les fichiers seront créés dans `data/processed/100K/`

**Pas besoin de modifier le reste du code !** 🎉

---

## ⚠️ Prérequis

Avant d'exécuter step3, assurez-vous que :

### 1. Step 2 est Terminé

```bash
ls data/sampled/
```

**Résultat attendu** :
```
10K/  50K/  100K/
```

Chaque dossier doit contenir :
- `transactions_sampled.csv`
- `articles_sampled.csv`
- `customers_sampled.csv`
- `sampling_metadata.json`

### 2. LightFM est Installé

```python
import lightfm
```

Si erreur :
```bash
pip install lightfm
```

---

## 🎓 Traitement par step3

Rappel de ce que fait step3 :

1. **Chargement** : Données échantillonnées depuis `data/sampled/{SAMPLE_SIZE}/`
2. **Dataset LightFM** : Création avec mappings automatiques
3. **Matrice d'interactions** : Construction avec `build_interactions()`
4. **Features matrices** : Construction avec `build_item_features()` et `build_user_features()`
5. **Sauvegarde** : Tous les objets dans `data/processed/{SAMPLE_SIZE}/`

### Objets Créés (Format LightFM Natif)

| Objet | Type | Usage |
|-------|------|-------|
| `dataset` | LightFM Dataset | Mappings internes, création de matrices |
| `user_item_matrix` | scipy.sparse COO | Matrice d'interactions complète |
| `item_features_matrix` | scipy.sparse CSR | 92 features items (catégoriques) |
| `user_features_matrix` | scipy.sparse CSR | 11 features users (âge, club, news) |

---

## 🚀 Prochaines Étapes

Après step3, vous aurez les données prétraitées au format LightFM.

### Step 4 : Train/Test Split Strategy

```python
# Charger les données prétraitées
import pickle
from scipy.sparse import load_npz

SAMPLE_SIZE = '50K'  # Même taille qu'en step3

with open(f'data/processed/{SAMPLE_SIZE}/lightfm_dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

user_item_matrix = load_npz(f'data/processed/{SAMPLE_SIZE}/user_item_matrix.npz')
item_features = load_npz(f'data/processed/{SAMPLE_SIZE}/item_features_matrix.npz')
user_features = load_npz(f'data/processed/{SAMPLE_SIZE}/user_features_matrix.npz')

# Step 4 créera les splits train/test
```

### Step 5 : Baseline Models

```python
from lightfm import LightFM

model = LightFM(loss='warp', no_components=30)
model.fit(
    interactions=train_interactions,  # Créé dans Step 4
    item_features=item_features,
    user_features=user_features,
    epochs=10
)
```

---

## 💡 Conseils

### 1. Commencez avec 50K

Pour Steps 3-6, utilisez toujours **50K** :
- Preprocessing rapide
- Itérations rapides pour hyperparameter tuning
- Données suffisantes pour apprentissage de qualité

### 2. Utilisez 10K pour Validation

Si vous modifiez le code de step3 et voulez tester rapidement :
```python
SAMPLE_SIZE = '10K'  # Test rapide
```

### 3. Finalisez avec 100K

Une fois les meilleurs hyperparamètres trouvés sur 50K (Step 6) :

```python
# Step 7 : Modèle final
SAMPLE_SIZE = '100K'
# Ré-exécuter step3, step4, step5 avec 100K
# Utiliser les meilleurs hyperparamètres de 50K
```

### 4. Gardez la Cohérence

**Utilisez la même taille** dans steps 3, 4, 5, 6 :

```
Step 3 : SAMPLE_SIZE = '50K' → Preprocessing
Step 4 : SAMPLE_SIZE = '50K' → Train/Test Split
Step 5 : SAMPLE_SIZE = '50K' → Baseline Models
Step 6 : SAMPLE_SIZE = '50K' → Hyperparameter Tuning
```

Puis pour le modèle final :
```
Step 7 : SAMPLE_SIZE = '100K' → Modèle final avec meilleurs params
```

---

## ✅ Checklist de Validation

Avant de continuer à Step 4, vérifier :

- [ ] step3 s'exécute sans erreur
- [ ] Dossier `data/processed/50K/` créé
- [ ] 14 fichiers présents dans `data/processed/50K/`
- [ ] `preprocessing_metadata.json` contient les bonnes statistiques
- [ ] Statistiques cohérentes avec le sample :
  - 50K → ~50,000 transactions
  - 50K → ~12,000 users
  - 50K → ~8,000 items
  - Avg trans/user ≈ 4.2
  - Avg trans/item ≈ 6.2

---

## 🎉 Avantages de Cette Approche

1. ✅ **Flexibilité** : Changement de taille en 1 ligne
2. ✅ **Séparation** : Chaque taille a ses propres fichiers processed
3. ✅ **Comparaisons** : Facile de comparer l'impact de la taille
4. ✅ **Pas de confusion** : Dossiers clairement nommés (10K, 50K, 100K)
5. ✅ **Workflow réaliste** : Développement sur 50K, production sur 100K

---

**🚀 Bonne préparation des données !**
