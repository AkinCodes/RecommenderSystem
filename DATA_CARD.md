# Data Card: MovieLens 100K

## Overview

| Field | Detail |
|---|---|
| **Dataset** | MovieLens 100K |
| **Source** | GroupLens Research, University of Minnesota |
| **URL** | https://grouplens.org/datasets/movielens/100k/ |
| **License** | Free for research and education. Citation required. |
| **Collection period** | September 1997 -- April 1998 |

## Dataset Composition

- **100,000 ratings** from **943 users** on **1,682 movies**
- Rating scale: 1--5 stars (integer)
- Users self-reported demographics: age, gender, occupation, zip code
- 71% male, predominantly US-based, skewed toward college-age users

## Preprocessing

| Step | Detail |
|---|---|
| Rating normalisation | Divided by 5 to scale to [0, 1] |
| Train/test split | 80/20 timestamp-based (earlier ratings train, newer test) |
| ID mapping | User and item IDs mapped to contiguous integers starting from 0 |
| Continuous features | Per-user mean rating, normalised rating count |
| Categorical features | user_id, item_id |

## Known Biases

- **Gender imbalance.** 71% of users are male. The model may learn preferences that skew toward male viewing habits.
- **Geographic bias.** Users are primarily US-based. Recommendations will reflect Western, English-language film preferences.
- **Temporal bias.** All ratings are from 1997--1998. The movie catalogue and user tastes are from a single era.
- **Popularity bias.** Popular movies accumulate far more ratings than niche titles. The model will tend to over-recommend well-known films.
- **Self-selection bias.** Users chose which movies to rate, meaning the observed ratings are not a random sample of user-movie preferences.

## Limitations

- **Small by modern standards.** Production recommender systems operate on billions of interactions. 100K ratings is a toy dataset.
- **No implicit feedback.** There are no views, clicks, dwell time, or other behavioural signals -- only explicit star ratings.
- **No content features.** Movie descriptions, posters, cast, and crew information are absent.
- **Stale data.** The catalogue ends in 1998. The model cannot generalise to modern content or shifting user preferences.

## Ethical Considerations

- No personally identifiable information is exposed. User IDs are anonymised integers.
- Demographic data (age, gender, occupation, zip code) is available in the dataset but is **not used** in our model.
- Popularity bias in recommendations should be monitored in any downstream application, as it can reduce exposure for lesser-known content.
