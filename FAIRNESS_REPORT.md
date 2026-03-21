# Fairness Analysis Report — DLRM on MovieLens 100K

## 1. Popularity Bias

| Metric | Value |
|--------|-------|
| Popularity threshold | >= 84 ratings in training |
| Popular items (top 20%) | 323 (19.2% of catalog) |
| % of recommendations from popular items | 69.0% |
| % of test interactions with popular items | 60.2% |
| Popularity amplification factor | 1.15x |

**Interpretation:** The model over-recommends popular items relative to their natural frequency in the test set.

## 2. User Activity Bias

| Metric | Value |
|--------|-------|
| Activity threshold | >= 172 ratings in training |
| Active users evaluated | 40 |
| Casual users evaluated | 243 |
| Active user NDCG@10 | 0.8228 |
| Casual user NDCG@10 | 0.7336 |
| Gap | +0.0892 |

**Interpretation:** Active users receive better recommendations, which is expected since the model has more data to learn their preferences.

## 3. Item Coverage

| Metric | Value |
|--------|-------|
| Total catalog size | 1682 |
| Unique items recommended | 574 |
| Catalog coverage | 34.1% |

**Interpretation:** Low coverage indicates the model concentrates recommendations on a small subset of items, potentially creating a filter bubble.

## 4. Genre Diversity

| Metric | Value |
|--------|-------|
| Total genres | 19 |
| Avg unique genres in top-10 | 8.87 |
| Min genres in any user's top-10 | 3 |
| Max genres in any user's top-10 | 13 |

**Interpretation:** On average, each user's top-10 recommendations span 8.9 out of 19 genres.
