import numpy as np
import pickle
import os

# Load top 25 SHAP-ranked question indices (across all 5 traits)
def load_question_ranking():
    path = os.path.join(os.path.dirname(__file__), "cat_question_ranking.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

cat_question_ranking = load_question_ranking()
top_25_questions = cat_question_ranking[:25]  # Only keep top 25 globally ranked questions


# Get the next unanswered question from top 25 SHAP-ranked features
def get_next_question(response_vector):
    for idx in top_25_questions:
        if np.isnan(response_vector[idx]):
            return idx
    return None  # All 25 questions answered


# Run adaptive session asking only top 25 SHAP-based questions
def run_adaptive_session(model_dict, X_train_filled, simulate_answer_func):
    response_vector = np.full((50,), np.nan)

    for idx in top_25_questions:
        response_vector[idx] = simulate_answer_func(idx)

    # Fill missing responses with feature-wise means
    input_vector = np.copy(response_vector)
    input_vector[np.isnan(input_vector)] = np.nanmean(X_train_filled, axis=0)[np.isnan(input_vector)]

    # Predict Big Five traits
    predictions = [model.predict(input_vector.reshape(1, -1))[0] for model in model_dict.values()]
    num_answered = np.count_nonzero(~np.isnan(response_vector))

    return predictions, num_answered
