# 🛍️ H&M Fashion Recommender System — LightFM

![LightFM](https://img.shields.io/badge/-LightFM-blue?style=flat) ![Python](https://img.shields.io/badge/-Python-3776AB?style=flat&logo=python&logoColor=white) ![Collaborative Filtering](https://img.shields.io/badge/-Collaborative%20Filtering-blue?style=flat) ![Hybrid Model](https://img.shields.io/badge/-Hybrid%20Model-blue?style=flat)

*Final project — with Henri Balamou.*

A hybrid recommender system built and rigorously evaluated on the **H&M Fashion** dataset (Kaggle), covering the full pipeline from data exploration to a hybrid Collaborative Filtering + Content-Based model, using **[LightFM](https://making.lyst.com/lightfm/docs/home.html)**. Main notebook: [`H&M-Fashion-RecommenderSystem-LightFM.ipynb`](H&M-Fashion-RecommenderSystem-LightFM.ipynb), 9 sections / ~180 cells.

## The dataset — and why it's hard

31.7M transactions over two years (Sept. 2018 – Sept. 2020), 1.36M users, 104,547 articles. **Sparsity: 99.98%** — the median user has exactly **1** transaction. A strict Pareto pattern holds: 30.8% of users generate 80% of interactions, and 20.7% of items generate 80% of sales, leaving a massive long tail (79.3% of the catalog is barely touched). We chose a **temporal 80/20 split** (train on 2018–2019, test on 2020) over a random split, favoring production realism over easier numbers — at the cost of a large temporal cold-start, made worse by COVID-19's disruption of 2020 shopping behavior.

## Pipeline (9 sections)

| Section | What it covers |
|---|---|
| 1. Exploratory analysis | Sparsity, interaction distributions, long-tail (80/20 rule), temporal patterns, metadata |
| 2. Sampling strategy | Pre-filtering active users/popular items, then random sampling; 3 saved sizes (10K/50K/100K) |
| 3. Preprocessing | Building the LightFM `Dataset`, interaction matrix, item/user feature matrices |
| 4. Train/test split strategy | Compared temporal, random, and user-based splits; chose temporal |
| 5. Model training | Systematic baseline experiments (loss function, latent dimensions, learning rate) |
| 6. Hyperparameter optimization | Grid Search (16 configs) + Random Search (50 configs) |
| 7. Evaluation & interpretation | Metrics at multiple K, user segmentation, manual inspection, diversity/coverage |
| 8. Hybrid model | Adds item features (product/colour/garment group...) and user features (age, membership...) |

## What we found

**Collaborative Filtering baseline.** Systematic experimentation showed **WARP loss beats BPR by 600%** (Precision@10 0.06% vs 0.01%) — WARP's ranking-focused, weighted negative sampling suits top-K recommendation far better here. 50 latent components was the sweet spot (30 underfits, 100 brings nothing extra on data this sparse). Grid + Random Search hyperparameter tuning improved Precision@10 by **+16.7%** over the manually-picked baseline, landing on WARP / 50 components / tuned regularization: **Precision@10 ≈ 0.07%, Recall@10 ≈ 0.6%, AUC ≈ 0.38**.

**Where pure CF breaks down.** Full evaluation exposed three structural problems: a **catastrophic popularity bias** (only 13 unique items ever recommended out of 24,216 in the catalog — 0.05% coverage, Gini index 0.23), **zero personalization** (identical bestseller recommendations regardless of purchase history — cold-start users even score *better* than heavy users, since the model just falls back to popularity), and **precision decaying with K** (0.07% at K=5 down to 0.04% at K=50), confirming the few relevant hits cluster at the very top of the list.

**What the hybrid model actually fixes.** Adding item + user features didn't move the global metrics (Precision@10 stayed at 0.07%, popularity bias unresolved — the categorical item features turned out too generic to discriminate between fashion items). But on **cold-start users specifically** (496 users, 5% of the test set, with no purchase history), it delivered a real, measurable win: **Precision@5 +109%** (0.52% → 1.09%) and **AUC +9.4%** (0.4921 → 0.5386). Demographic features (age, membership status, newsletter frequency) generalize to new users by mapping them onto learned segments — exactly the scenario a hybrid model is supposed to help with. Item cold-start didn't see the same benefit: two products sharing the same "Ladieswear / Upper body / Black" tags can be a €10 t-shirt and a €200 jacket, so those categorical features are too coarse to separate them.

## Challenges we hit — and how we solved them

- **Sparsity made the naive approach impossible.** A dense interaction matrix for the full dataset would need ~532GB of RAM. → Two-stage sampling: pre-filter to active users (≥5 transactions) and popular items (≥10 transactions), then sample, preserving the ~99.99% sparsity signature at a workable size.
- **Split strategy choice materially changes the results.** → Compared temporal / random / user-based splits explicitly before committing to temporal, and documented that this choice *penalizes* absolute performance in exchange for a realistic evaluation of production-style temporal generalization.
- **A real bug: index misalignment between interaction and feature matrices.** Early versions built the interaction matrix with a hand-rolled `user_id_map`/`item_id_map`, while LightFM's own `Dataset.fit()` built a *separate* mapping for the feature matrices — the two didn't line up, and `user_features` were silently missing from evaluation calls entirely. Symptom: hybrid-model AUC scores *below random* (~0.32–0.38 on a metric where 0.5 = chance). Fix ([`docs/SOLUTION_V1.4_EVALUATION_FIX.md`](docs/SOLUTION_V1.4_EVALUATION_FIX.md)): drop the manual mappings entirely, build every matrix through LightFM's own `Dataset.build_interactions()` / `build_item_features()` / `build_user_features()`, assert shape alignment right after construction, and pass `user_features` consistently at both training *and* evaluation time. Also removed a leftover sklearn `LabelEncoder`/`MultiLabelBinarizer` path that duplicated LightFM's own encoding (see [`docs/RAPPORT_NETTOYAGE_FINAL.md`](docs/RAPPORT_NETTOYAGE_FINAL.md)). After the fix, AUC scores landed back in the expected 0.5–0.73 range.
- **Precision@10 = 0.07% looks like failure at first glance.** → Contextualized against the literature: comparable sparse, implicit-feedback, temporal-split e-commerce setups typically report 0.1–2% Precision@10; recommending 7 relevant items out of 1,000 in a 24K-item catalog, with a median of 1 transaction/user, is a real signal given the constraints — and the hybrid model's 1.09% on cold-start users sits comfortably inside that range.
- **A hybrid model with no global improvement looked like a dead end.** → Segment-level analysis (rather than trusting the single aggregate number) is what surfaced the +109% cold-start win — turning an apparently "disappointing" result into a specific, actionable finding about which features help and which don't.

## Limitations

- **Extreme sparsity + implicit-only feedback** (purchases only, no ratings, no explicit negatives) caps how much individual preference signal is learnable at all.
- **Item features are too generic** (19 product groups, 49 colours) to discriminate within a visually and stylistically diverse fashion catalog — no text, image, or price features are used.
- **User features are purely demographic** — no purchase-history-derived behavioral features (preferred categories, average basket, purchase frequency).
- **LightFM's architecture is linear** (dot products of embeddings) — it can't capture non-linear feature interactions or purchase sequences the way deep learning or sequence models (RNN/Transformer) could.
- **Static evaluation** — no incremental retraining, so the model is tested 6–12 months out from its training window, which a real production system would avoid by retraining regularly.

## Improvement priorities

- **Richer features**: text embeddings on product descriptions, CNN-based visual features from product images, price/brand/return-rate, seasonality — plus behavioral user features (category mix, average basket, channel, promo-sensitivity).
- **Re-run the hybrid model on a random split** to isolate the pure feature contribution from the temporal cold-start penalty (diagnostics suggest a possible 10–50× metric improvement).
- **Scale up**: 100K/500K-transaction samples are already prepared and ready to use.
- **Post-processing**: MMR-based re-ranking for diversity, long-tail boosting to counter popularity bias, business rules (no repeat recommendations, stock/seasonality filters).
- **Beyond LightFM**: sequence models (RNN/Transformer) for next-purchase prediction, graph neural networks over the user-item-feature graph, or a candidate-generation + re-ranking cascade for production.

## Files in this folder

| Path | Content |
|---|---|
| `H&M-Fashion-RecommenderSystem-LightFM.ipynb` | Final notebook — the 9-section pipeline described above |
| `RS_Rendu_Henri_Dernière_Version.ipynb` | Henri's submission version |
| `data/` | H&M dataset files (transactions, articles, customers) — large, excluded from git |
| `models/{1K,10K,50K}/` | Saved models and result JSONs at each sampling scale |
| `outputs/`, `figures/` | Generated charts (sparsity, temporal, cold-start, demographic analyses) |
| `docs/` | Working notes: diagnostics, fixes, and correction reports written while debugging (see above) |
| `notebooks/` | Iteration history — versioned drafts and per-step notebooks kept during development |
