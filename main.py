import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st 
st.set_page_config(page_title="Big Five Predictor", layout="wide")


from ui_utils import set_background
import pickle
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from app.model_utils import load_model, load_feature_means, predict_traits, inverse_standardize, score_to_percentage
from data.questions import questions, trait_names
from data.trait_descriptions import trait_descriptions
import base64


def load_question_ranking():
    path = os.path.join(os.path.dirname(__file__), "cat_question_ranking.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

# Set the background
set_background("static/background.png")

# Initialize session state variable to track if user accepted
if "started" not in st.session_state:
    st.session_state.started = False

# --------------------- Welcome Page ---------------------
if not st.session_state.started:
    st.title("🧠 Welcome to the Big Five Personality Trait Predictor")

    st.markdown("""
    ### What is the Big Five Factor Model?
    The Big Five Factor Model, also known as the Five Factor Model (FFM), describes personality using five broad dimensions:
    - **Extraversion** (how outgoing, energetic, and drawn to social interaction we tend to be)
    - **Neuroticism** (how emotionally reactive or resilient we are in the face of stress and negative feelings)
    - **Agreeableness** (how compassionate, cooperative, and considerate we are toward others)
    - **Conscientiousness** (how organized, responsible, and reliable we are in meeting tasks and obligations)
    - **Openness to Experience** (how curious, imaginative, and open-minded we are toward new ideas and experiences)
    


    ---
    ### ℹ️ About the Big Five

    The Big Five Personality Model emerged in the 1970s through the work of two independent research teams—**Paul Costa and Robert McCrae** at the National Institutes of Health, and **Warren Norman and Lewis Goldberg** at the University of Michigan and the University of Oregon.

    Although their methods differed slightly, both groups arrived at the same conclusion: that human personality can be understood in terms of five major dimensions, consistent across cultures and languages.

    It’s important to note that researchers **did not aim** to find five traits; the five traits **emerged naturally** from data collected through large-scale surveys in which thousands of participants answered hundreds of personality-related questions.
    Today, the Big Five is considered the most widely accepted and scientifically supported framework for describing personality in psychology.

    ---

    ### 🔒 Privacy Notice

    Your responses and results **will not be saved or shared**. This test is for **your personal insight only**, and no data is stored or transmitted beyond this session.

    This is a **self-assessment tool** designed for educational and personal use, not a clinical or diagnostic instrument.
    """)

    if st.button("✅ I Understand and Would Like to Continue"):
        st.session_state.started = True
        st.rerun()

# --------------------- Main App After Start ---------------------
else:
    st.title("🧠 Big Five Personality Trait Predictor")

    # Load models and training data
    import pickle
    ranking_path = os.path.join(os.path.dirname(__file__), "cat_question_ranking.pkl")
    with open(ranking_path, "rb") as f:
        cat_question_ranking = pickle.load(f)

    cat_models = load_model()
    X_train_filled_df = load_feature_means()

    # Load models and store in session state
    if "models" not in st.session_state:
        st.session_state["models"] = load_model()


    # Feature names stay the same
    cat_features = [f"Q{i+1}" for i in range(50)]

    # Session state init
    if "response_vector" not in st.session_state:
        st.session_state.response_vector = np.full((50,), np.nan)
        st.session_state.question_index = 0
        st.session_state.finished = False
        st.session_state.max_initial = 25
        st.session_state.final_prediction = None

    ranked_questions = cat_question_ranking

    # --------------------- Adaptive Questioning ---------------------
    if not st.session_state.finished and st.session_state.question_index < st.session_state.max_initial:

        # Track question navigation to clear stale answers if user goes backward
        if "previous_q_index" not in st.session_state:
            st.session_state.previous_q_index = 0

        # Clear old answer if going back
        if st.session_state.question_index < st.session_state.previous_q_index:
            prev_q = cat_question_ranking[st.session_state.previous_q_index]
            st.session_state.response_vector[prev_q] = np.nan

        # Update previous question tracker
        st.session_state.previous_q_index = st.session_state.question_index

        next_q_idx = ranked_questions[st.session_state.question_index]
        question_number = next_q_idx + 1
        question_text = questions[next_q_idx]
        radio_key = f"q_{next_q_idx}"

        # Clear all other stale radio answers except the current one
        for key in list(st.session_state.keys()):
            if key.startswith("q_") and key != radio_key:
                del st.session_state[key]

        # Display the current question number and a progress bar
        st.markdown(f"**Question {st.session_state.question_index + 1} of {st.session_state.max_initial}**")
        st.progress((st.session_state.question_index + 1) / st.session_state.max_initial)

        # Show the current question as a radio button with Likert-scale options
        st.radio(
            f"Q{question_number}: {question_text}",
            options=[1, 2, 3, 4, 5],
            index=None,
            key=radio_key,
            format_func=lambda x: {
                1: "1 - Strongly Disagree", 2: "2 - Disagree", 3: "3 - Neutral",
                4: "4 - Agree", 5: "5 - Strongly Agree"
            }[x], # Converts numeric choices into descriptive labels
        )

        # Create two equally sized columns for the navigation buttons
        col1, col2 = st.columns([1, 1])

        # "Back" button logic
        with col1:
            if st.button("⬅️ Back"):
                if st.session_state.question_index > 0:
                    st.session_state.question_index -= 1
                    st.rerun()
                else:
                    # Reset to go back to welcome screen
                    st.session_state.clear()
                    st.rerun()

        # "Submit Answer" button logic
        with col2:
            if st.button("➡️ Submit Answer"):
                user_input = st.session_state.get(radio_key, None)
                if user_input is None:
                    st.warning("⚠️ Please select an answer before continuing.")
                else:
                    # Track question and answer for download
                    if "answers_list" not in st.session_state:
                        st.session_state.answers_list = []

                    st.session_state.answers_list.append({
                        "Question": f"Q{question_number}: {question_text}",
                        "Answer": {
                            1: "1 - Strongly Disagree", 2: "2 - Disagree", 3: "3 - Neutral",
                            4: "4 - Agree", 5: "5 - Strongly Agree"
                        }[user_input]
                    })

                    # Save user response into the response vector and move on to the next question
                    st.session_state.response_vector[next_q_idx] = user_input
                    st.session_state.question_index += 1

                    input_vector = pd.Series(np.copy(st.session_state.response_vector), index=cat_features)
                    input_vector.fillna(X_train_filled_df.mean(), inplace=True)

                    # Generate personality trait predictions based on current input
                    prediction = predict_traits(cat_models, input_vector)

                    # If user finished all initial questions, store final prediction and mark as done
                    if st.session_state.question_index >= st.session_state.max_initial:
                        st.session_state.finished = True
                        st.session_state.final_prediction = prediction

                    # Refresh the app to show next question or final results
                    st.rerun()


    # --------------------- Final Results ---------------------

    # If the adaptive question phase is completed
    if st.session_state.finished:
        prediction = st.session_state.final_prediction
        rescaled_prediction = inverse_standardize(prediction)
        percentages = score_to_percentage(rescaled_prediction)
        pred_df = pd.DataFrame([percentages], columns=trait_names)
        st.session_state["pred_df"] = pred_df

        st.success("✅ Prediction complete!")
        st.dataframe(pred_df.style.format("{:.2f}%"))

        dominant_trait = trait_names[np.argmax(percentages)]
        dominant_value = percentages[np.argmax(percentages)]

        st.markdown(
            f"""
            <div style='text-align: center; color: white; font-size: 28px; font-weight: bold; margin-top: 30px;'>
                Your most dominant trait is: {dominant_trait} ({dominant_value:.1f}%)
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("---")
        st.markdown(trait_descriptions[dominant_trait])

        input_vector = pd.Series(np.copy(st.session_state.response_vector), index=cat_features)
        input_vector.fillna(X_train_filled_df.mean(), inplace=True)
        input_df = input_vector.to_frame().T


        input_vector = pd.Series(np.copy(st.session_state.response_vector), index=cat_features)
        input_vector.fillna(X_train_filled_df.mean(), inplace=True)
        input_df = input_vector.to_frame().T
        # Rename columns to match what the model expects: "0", "1", ..., "49"
        input_df_renamed = input_df.copy()
        input_df_renamed.columns = [str(i) for i in range(input_df.shape[1])]

        # 📘 SHAP Explanation
        st.markdown("### 📊 Understanding Your Personality Insights with SHAP")
        st.markdown(
            """
            The graphs below help explain **why** the model predicted your scores the way it did.

            - Each bar represents a question you answered.
            - The longer the bar, the more influence that question had.
            - If a question **increased** your score for a trait, it appears in **red**.
            - If a question **decreased** your score for a trait, it appears in **blue**.

            🔍 These insights can help you understand which of your responses had the most impact on each trait.

            📥 **Tip:** Download your full results with the download button below to see a complete list of your answers. That way, you can compare your inputs with the SHAP insights.
            """
        )

        # SHAP plot for each trait (use raw input_df with correct column names)
        for trait, model in st.session_state.models.items():
            explainer = shap.Explainer(model)
            shap_values = explainer(input_df_renamed)
            shap_values.feature_names = cat_features  # Use ["Q1", ..., "Q50"] instead of ["0", ..., "49"]


            abs_shap_vals = np.abs(shap_values.values[0])
            top_indices = np.argsort(abs_shap_vals)[::-1][:10]

            shap.initjs()
            st.subheader(f"🔍 SHAP Explanation for {trait}")
            fig = plt.gcf()
            shap.plots.waterfall(shap_values[0, top_indices], max_display=10, show=False)
            st.pyplot(fig)
            plt.clf()

    # ✅ Also keep this at the very end
    if "pred_df" in st.session_state and "answers_list" in st.session_state:
        pred_df = st.session_state["pred_df"]

        # Convert answers list to DataFrame
        answers_df = pd.DataFrame(st.session_state["answers_list"])

        # Create a blank row to separate sections
        spacer = pd.DataFrame([["", ""]], columns=answers_df.columns)

        # Add prediction scores, renaming for clarity
        predictions_df = pred_df.T.reset_index()
        predictions_df.columns = ["Trait", "Score"]

        # Combine everything
        combined_df = pd.concat([answers_df, spacer, predictions_df], ignore_index=True)

        # Download button
        csv = combined_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Your Full Results (Answers + Predictions)",
            data=csv,
            file_name="your_personality_summary.csv",
            mime="text/csv"
        )

    # ✅ Restart button - return empty answers_list
    if st.button("🔄 Restart", key="restart_bottom"):
        st.session_state.response_vector = np.full((50,), np.nan)
        st.session_state.question_index = 0
        st.session_state.finished = False
        st.session_state.final_prediction = None
        st.session_state.answers_list = []
        st.rerun()

