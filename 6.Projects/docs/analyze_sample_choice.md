# Guide de Sélection de la Taille d'Échantillon

## 🎯 Objectif du Projet

Selon le README du projet :
- **Objectif** : Construire un système de recommandation avec LightFM
- **Contrainte** : Temps d'entraînement raisonnable pour itérations rapides
- **Besoin** : Données suffisantes pour apprentissage de qualité

## 📊 Analyse par Taille d'Échantillon

### 🔴 1K - 10K transactions : **TROP PETIT**

**Avantages** :
- ✅ Très rapide (< 30 secondes d'entraînement)
- ✅ Parfait pour tests unitaires/debug

**Inconvénients** :
- ❌ Avg_Trans_User ≈ 2-3 (INSUFFISANT)
- ❌ Avg_Trans_Item ≈ 2-4 (INSUFFISANT)
- ❌ Trop peu d'utilisateurs/items (~500-3000)
- ❌ Modèle très faible, overfitting probable
- ❌ **Ne pas utiliser pour entraînement final**

**Recommandation** : ❌ **NE PAS CHOISIR** (sauf debug rapide)

---

### 🟡 50K transactions : **OPTIMAL POUR EXPÉRIMENTATION**

**Avantages** :
- ✅ Temps d'entraînement : **2-3 minutes** (très raisonnable)
- ✅ Avg_Trans_User ≈ 4-5 (ACCEPTABLE)
- ✅ Avg_Trans_Item ≈ 6-8 (BON)
- ✅ ~10K-15K users, ~6K-10K items (diversité OK)
- ✅ **Itérations rapides** pour hyperparameter tuning
- ✅ Balance performance/qualité

**Inconvénients** :
- ⚠️ Coverage limitée (~10-15% du dataset complet)
- ⚠️ Peut manquer patterns long-tail

**Recommandation** : ✅ **CHOIX RECOMMANDÉ** pour la majorité du projet
- Phase d'expérimentation
- Hyperparameter tuning
- Développement et tests

---

### 🟢 100K transactions : **BON COMPROMIS**

**Avantages** :
- ✅ Temps d'entraînement : **4-6 minutes** (acceptable)
- ✅ Avg_Trans_User ≈ 5-6 (BON)
- ✅ Avg_Trans_Item ≈ 8-10 (TRÈS BON)
- ✅ ~20K-25K users, ~10K-15K items (bonne diversité)
- ✅ Meilleure représentativité
- ✅ Capture plus de patterns

**Inconvénients** :
- ⚠️ 2x plus lent que 50K
- ⚠️ Coverage encore limitée (~20-25%)

**Recommandation** : ✅ **BON CHOIX** si :
- Vous avez le temps (expérimentation moins rapide)
- Vous voulez un modèle plus robuste
- Phase finale avant production

---

### 🔵 500K - 1M transactions : **MAXIMUM DISPONIBLE** (~259K réel)

**Note** : Votre dataset pré-filtré = ~259K transactions max

**Avantages** :
- ✅ Avg_Trans_User ≈ 5.4 (TRÈS BON)
- ✅ Avg_Trans_Item ≈ 13 (EXCELLENT)
- ✅ ~48K users, ~20K items (excellente diversité)
- ✅ Meilleure coverage (~35% users, ~19% items)
- ✅ **Meilleur modèle final**

**Inconvénients** :
- ❌ Temps d'entraînement : **15-20 minutes** (très long)
- ❌ Hyperparameter tuning TRÈS lent
- ❌ Difficile d'itérer rapidement

**Recommandation** : ⚠️ **Utiliser UNIQUEMENT** pour :
- **Modèle final** après avoir optimisé sur 50K
- **Évaluation finale** de performance
- **Comparaison** avec modèle 50K

---

## 🎯 Stratégie Recommandée : APPROCHE EN 2 PHASES

### **PHASE 1 : Développement & Optimisation (50K)**

```python
TARGET_SIZE = 50_000
```

**Utiliser pour** :
1. ✅ Exploration initiale
2. ✅ Step 3 : Preprocessing
3. ✅ Step 4 : Train/Test split comparisons
4. ✅ Step 5 : Modèle baseline
5. ✅ Step 6 : Hyperparameter tuning (CRUCIAL)
6. ✅ Step 7 : Évaluation initiale
7. ✅ Step 8 : Modèle hybride avec features

**Temps estimé** : 2-3 min/entraînement → itérations rapides

---

### **PHASE 2 : Modèle Final (100K ou 259K)**

Une fois les hyperparamètres optimisés sur 50K :

```python
# Option 1 : Validation intermédiaire
TARGET_SIZE = 100_000

# Option 2 : Modèle final maximal
TARGET_SIZE = len(base_sample_combine)  # ~259K
```

**Utiliser pour** :
1. ✅ Ré-entraîner avec les MEILLEURS hyperparamètres trouvés
2. ✅ Évaluation finale de performance
3. ✅ Comparaison 50K vs 100K/259K
4. ✅ Analyse de l'impact de la taille des données

---

## 📋 Checklist de Décision

Quand vous regardez vos `results_df` de Section 7 :

```python
# Pour chaque taille, vérifiez :

✅ Avg_Trans_User >= 4 ?           # Minimum viable
✅ Avg_Trans_Item >= 5 ?           # Minimum viable
✅ Users >= 5000 ?                 # Diversité minimale
✅ Items >= 3000 ?                 # Catalogue minimal
✅ Sparsity < 99.99% ?             # Pas trop sparse
✅ Temps estimé acceptable ?       # Itérations rapides

# Estimation du temps d'entraînement
# (basé sur vos observations précédentes)

1K    →   ~20-30 sec    (trop petit, skip)
10K   →   ~30-60 sec    (trop petit, skip)
50K   →   ~2-3 min      ✅ OPTIMAL pour expérimentation
100K  →   ~4-6 min      ✅ BON pour modèle plus robuste
259K  →   ~15-20 min    ⚠️ Uniquement modèle final
```

---

## 🎓 Exemple de Workflow Complet

### Étapes 3-6 : Utiliser 50K

```python
# Section 8 de step2
TARGET_SIZE = 50_000
```

**Raison** : Itérations rapides pour :
- Tester différentes stratégies de split (step4)
- Comparer loss functions (WARP, BPR, logistic)
- Tuner hyperparamètres (no_components, learning_rate, epochs)
- Expérimenter avec features items

**Temps gagné** : 50K = 2-3 min/run vs 259K = 20 min/run
→ Vous pouvez tester **6-7 configurations** dans le même temps !

---

### Étape 7 : Modèle Final avec 100K ou 259K

Une fois que vous savez que :
- WARP avec no_components=50, learning_rate=0.05, epochs=15 est optimal
- Split temporel 90/10 est le meilleur
- Features items améliorent de X%

**Alors** : Ré-entraîner le modèle final avec plus de données

```python
# Créer un nouveau sample (optionnel)
TARGET_SIZE = 100_000  # ou max(259K)

# Ré-exécuter step3, step4, step5 avec ce nouveau sample
# Utiliser les MEILLEURS hyperparamètres trouvés sur 50K
```

---

## 💡 Règle d'Or

> **"Optimize small, deploy big"**

1. **Développez et optimisez** sur un échantillon rapide (50K)
2. **Validez et finalisez** sur un échantillon plus grand (100K/259K)
3. Ne perdez **PAS** votre temps à attendre 20 minutes par run pendant l'expérimentation

---

## 🚀 Ma Recommandation Finale

Pour votre projet EMIASD :

### **Choix Principal : 50K** ✅

**Section 8 de step2** :
```python
TARGET_SIZE = 50_000
MIN_ITEM = 5
MIN_USER = 4
```

**Pourquoi** :
- ✅ Respecte la recommandation du README du projet
- ✅ Temps d'entraînement raisonnable (2-3 min)
- ✅ Permet beaucoup d'expérimentation
- ✅ Données suffisantes pour apprentissage de qualité
- ✅ Vous pourrez compléter TOUTES les étapes du projet rapidement

**Bonus (optionnel)** :
À la toute fin, créez un deuxième sample de 259K et comparez :
```python
# Dans votre rapport final
"Performance 50K  : Precision@10 = 0.15, AUC = 0.82"
"Performance 259K : Precision@10 = 0.18, AUC = 0.85"
"Gain : +20% avec 5x plus de données"
```

Cela montre que vous comprenez le trade-off temps/performance ! 🎯
