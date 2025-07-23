# app/model_utils.py

import os
import pickle
import pandas as pd
import numpy as np

# Base directory of the current file
BASE_DIR = os.path.dirname(__file__)

def load_model():
    path = os.path.join(BASE_DIR, "../model/final_catboost_models.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

def load_feature_means():
    path = os.path.join(BASE_DIR, "../data/X_train_filled_df.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

def load_scaler():
    path = os.path.join(BASE_DIR, "../model/scaler.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

def load_imputer():
    path = os.path.join(BASE_DIR, "../model/imputer.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

def predict_traits(models_dict, input_vector):
    preds = []

    # Convert to DataFrame if input is numpy array
    if isinstance(input_vector, np.ndarray):
        feature_names = [f"Q{i+1}" for i in range(50)]
        input_df = pd.DataFrame(input_vector.reshape(1, -1), columns=feature_names)
    else:
        input_df = input_vector  # Already a DataFrame

    for trait, model in models_dict.items():
        pred = model.predict(input_df)
        preds.append(pred[0] if isinstance(pred, (list, np.ndarray)) else pred)

    return np.array(preds)

def inverse_standardize(preds):
    """
    Reverses standardization back to 1–5 Likert scale.
    Original labels were standardized (mean=0, std=1) from [1, 5].
    """
    return np.clip((preds * 0.75) + 3, 1, 5)

def score_to_percentage(score):
    """
    Converts a Likert score (1–5) to a percentage [0% - 100%].
    """
    return ((score - 1) / 4) * 100
