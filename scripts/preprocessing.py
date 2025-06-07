import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

# Paths for saved preprocessing artifacts
SCALER_PATH = "scripts/netflix_scaler.pkl"
TYPE_ENCODER_PATH = "scripts/type_encoder.pkl"
RATING_ENCODER_PATH = "scripts/rating_encoder.pkl"


# --- CONTINUOUS FEATURES SCALING ---


def fit_and_save_scaler(release_years, durations):
    """Fit and save the StandardScaler for continuous features."""
    data = np.stack([release_years, durations], axis=1)
    scaler = StandardScaler()
    scaler.fit(data)
    joblib.dump(scaler, SCALER_PATH)
    print(f"✅ Scaler saved to {SCALER_PATH}")


def load_and_apply_scaler(release_year, duration):
    """Load saved scaler and apply to one input. Returns float32 NumPy array with shape (2,)."""
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Scaler not found. Run fit_and_save_scaler first.")

    scaler = joblib.load(SCALER_PATH)
    scaled = scaler.transform([[release_year, duration]])  # shape: (1, 2)
    return scaled.astype(np.float32).squeeze()  # shape: (2,), dtype: float32


# --- CATEGORICAL FEATURES ENCODING ---


def fit_and_save_encoders(types, ratings):
    """Fit and save LabelEncoders for categorical features."""
    type_encoder = LabelEncoder().fit(types)
    rating_encoder = LabelEncoder().fit(ratings)
    joblib.dump(type_encoder, TYPE_ENCODER_PATH)
    joblib.dump(rating_encoder, RATING_ENCODER_PATH)
    print(f"✅ Encoders saved to {TYPE_ENCODER_PATH} and {RATING_ENCODER_PATH}")


def load_encoders():
    """Load saved encoders for categorical features."""
    if not os.path.exists(TYPE_ENCODER_PATH) or not os.path.exists(RATING_ENCODER_PATH):
        raise FileNotFoundError("Encoders not found. Run fit_and_save_encoders first.")
    type_encoder = joblib.load(TYPE_ENCODER_PATH)
    rating_encoder = joblib.load(RATING_ENCODER_PATH)
    return type_encoder, rating_encoder
