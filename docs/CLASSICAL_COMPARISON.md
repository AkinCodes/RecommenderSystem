# Classical ML Baselines vs DLRM — MovieLens 100K

Evaluated on 283 users with top-10 recommendations.

## Features for Classical Models

| # | Feature | Description |
|---|---------|-------------|
| 0 | user_mean_rating | Average rating the user gave (normalised) |
| 1 | user_rating_count | Number of ratings by user (normalised) |
| 2 | user_rating_var | Variance of user's ratings (normalised) |
| 3 | user_days_active | Days between first and last rating (normalised) |
| 4 | item_mean_rating | Average rating the item received (normalised) |
| 5 | item_rating_count | Number of ratings for item (normalised) |
| 6 | item_popularity_rank | Popularity rank normalised to [0, 1] |
| 7 | deviation | user_mean - item_mean (preference signal) |

## Results

| Model            |  NDCG@10 |  Prec@10 |   Hit@10 |  Train Time |     Inference |
|------------------|----------|----------|----------|-------------|---------------|
| DLRM             |   0.7370 |   0.6311 |   0.9965 |           - |   0.09ms/user |
| XGBoost          |   0.7836 |   0.6707 |   1.0000 |        0.6s |   0.27ms/user |
| LightGBM         |   0.7820 |   0.6661 |   0.9965 |        1.2s |   0.47ms/user |
| LogReg           |   0.8023 |   0.6880 |   1.0000 |        0.0s |   0.12ms/user |
| Random           |   0.6046 |   0.5300 |   0.9894 |           - |             - |
| Most Popular     |   0.6940 |   0.5915 |   0.9965 |           - |             - |

## Model Details

- **DLRM**: Deep Learning Recommendation Model with user/item embeddings (128-dim) and 3-layer MLP.
- **XGBoost**: Gradient-boosted trees (200 estimators, max_depth=6) on 8 hand-crafted features.
- **LightGBM**: Gradient-boosted trees (200 estimators, max_depth=6) on the same features.
- **LogReg**: Logistic Regression with StandardScaler preprocessing on the same features.
- **Random**: Uniformly random scores.
- **Most Popular**: Rank by training-set popularity count.

## Key Takeaway

Classical models with good feature engineering are competitive baselines. The DLRM's advantage comes from learning user/item embeddings that capture latent interaction patterns beyond hand-crafted features.
