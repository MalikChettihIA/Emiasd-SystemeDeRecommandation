# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Recommender Systems** educational repository for the EMIASD program. It contains course materials, lab exercises, and a final project focused on building and evaluating recommendation engines. The repository uses two main datasets:
- **MovieLens 100K**: For learning and lab exercises (943 users, 1682 movies, 100K ratings)
- **H&M Fashion Dataset**: For the final project (1.3M customers, 105K articles, 31M transactions)

## Repository Structure

```
1.Cours/           # Course PDFs covering recommender systems topics
2.Notes/           # Personal notes (empty placeholder)
3.Td/              # Lab exercises using MovieLens dataset
4.Documentations/  # Reference books and syllabus
5.Projets/         # Final project implementation with H&M dataset
```

## Working with Jupyter Notebooks

All code implementations use Jupyter notebooks (`.ipynb` files):

**Running notebooks:**
```bash
jupyter notebook
# or
jupyter lab
```

**Common libraries used:**
- pandas, numpy - Data manipulation
- matplotlib, seaborn - Visualization
- sklearn - ML utilities (cosine_similarity, metrics)
- scipy - Sparse matrices and SVD

## Key Files

### Lab Exercises (3.Td/)
- `Lab_using_MovieLens.ipynb` - Main working version
- `Lab_using_MovieLens_solution.ipynb` - Complete solution with implementations
- `Lab_using_MovieLens_original.ipynb` - Original starter template
- `movielens/` - MovieLens 100K dataset (downloaded automatically by notebooks)

### Final Project (5.Projets/)
- `exploratory_data_analysis.ipynb` - EDA on H&M fashion dataset
- `helpers.py` - Utility functions (data type inspection)
- `data/` - Large CSV files (articles, customers, transactions)

## Data Files

### MovieLens Dataset (3.Td/movielens/ml-100k/)
- `u.data` - User ratings (user_id, item_id, rating, timestamp)
- `u.item` - Movie metadata (item_id, title, release_date, genres as binary columns)
- `u.user` - User demographics
- Auto-downloaded by notebooks using urllib and zipfile

### H&M Dataset (5.Projets/data/)
Large CSV files (>3GB total):
- `articles.csv` - Product catalog with 25 columns including genres, colors, departments
- `customers.csv` - Customer profiles with demographics
- `transactions_train.csv` - Purchase history (31.7M records)

**Important:** These files are large and excluded from git via `.gitignore`. Handle with chunking or sampling for experimentation.

## Recommendation Algorithms Implemented

The lab notebooks implement and compare four approaches:

1. **Popularity-Based Recommender**
   - Weighted score: 0.7 × avg_rating + 0.3 × log(num_ratings)
   - Good baseline, handles cold start for new users

2. **Content-Based Filtering**
   - Uses movie genres (18 binary features) + normalized release year
   - Cosine similarity between item feature vectors
   - Good for new items, provides explainable recommendations

3. **User-Based Collaborative Filtering**
   - Builds user-item rating matrix
   - Finds similar users via cosine similarity
   - Recommends items liked by similar users (rating ≥ 4)

4. **Matrix Factorization (SVD)**
   - Uses scipy's `svds()` with 50 latent factors
   - Normalizes ratings by user mean before decomposition
   - Best personalization but requires more data

## Evaluation Metrics

Implementations include:
- Precision@K, Recall@K, F1@K
- Coverage (catalog diversity)
- Train/test split (80/20)
- Sample 100 test users for speed

## Data Processing Patterns

### Sampling Strategy (H&M Project)
The EDA notebook uses a two-step sampling approach:
```python
def enchantillonner(df, column_name, k):
    # Keep only items/users with >= k transactions
    column_counts = df[column_name].value_counts()
    columns_valid = column_counts[column_counts >= k].index
    return df[df[column_name].isin(columns_valid)]
```

Thresholds used:
- k_article = 1000 (reduces to ~17M transactions)
- k_customer = 100 (reduces to ~9.4M transactions)

### Reading Metadata
Helper function pattern from `helpers.py`:
```python
def read_df_metadata(df):
    # Returns DataFrame with columns: ['colonne', 'type']
    # type is either 'numeric' or 'char'
```

## Production Considerations

The lab includes a `RecommendationService` class demonstrating:
- Caching recommendations by (user_id, algorithm, n) key
- Fallback to popularity when primary algorithm fails
- Metrics tracking (cache hit rate, response time, fallback rate)

## Common Tasks

**Explore MovieLens dataset:**
```python
ratings_df = pd.read_csv('3.Td/movielens/ml-100k/u.data', sep='\t',
                         names=['user_id', 'item_id', 'rating', 'timestamp'])
```

**Work with H&M data (use sampling):**
```python
# Load in chunks or sample first N rows
df = pd.read_csv('5.Projets/data/transactions_train.csv', nrows=100000)
```

**Add movie titles to recommendations:**
```python
# Pattern used throughout notebooks
recommendations.merge(movies_df[['item_id', 'title']], on='item_id')
```

## Important Notes

- The notebooks auto-download MovieLens data on first run
- H&M dataset files are large (3.7GB total) - use sampling for development
- All implementations prioritize educational clarity over production performance
- SVD may take 30-60 seconds to train on full MovieLens dataset
- Content-based recommender requires fitting with `ratings_df` to enable user recommendations

## Course Topics Covered

Based on the course materials in `1.Cours/`:
1. Introduction to Recommender Systems
2. Reinforcement Learning for RecSys
3. Deep Learning for RecSys
4. Point-of-Interest (POI) Recommendation
5. System Architecture in RecSys
6. LLM-based Recommender Systems
