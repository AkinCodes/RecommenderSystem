# Baseline Comparison — DLRM vs Simple Baselines on MovieLens 100K

Evaluated on 283 users with top-10 recommendations.

## Results

| Model           | NDCG@10 | Precision@10 | HitRate@10 |
|-----------------|---------|--------------|------------|
| Random          | 0.6046  | 0.5300       | 0.9894     |
| Most Popular    | 0.6940  | 0.5915       | 0.9965     |
| User Mean       | 0.8008  | 0.6855       | 1.0000     |
| DLRM    | 0.7462  | 0.6396       | 0.9965     |

## Baseline Descriptions

- **Random**: Assign random scores to each candidate item.
- **Most Popular**: Rank items by number of ratings in the training set.
- **User Mean**: Rank items by their average rating in the training set (a proxy for item quality).
- **DLRM**: Deep Learning Recommendation Model with user/item embeddings and dense features.
