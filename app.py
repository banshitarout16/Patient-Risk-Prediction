import json
import os

import joblib
import pandas as pd
import streamlit as st

from model_utils import FEATURE_COLUMNS, LEVEL_LABELS, risk_label

MODEL_DIR = "models"
ASSET_DIR = "assets"
MODEL_FILE = "random_forest.pkl"

st.set_page_config(page_title="Patient Risk Screening", page_icon="🩺", layout="wide")


st.markdown(
    """
    <style>
    .risk-card {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1.25rem 1.75rem;
        border-radius: 0.9rem;
        margin-bottom: 0.75rem;
        border-left: 6px solid;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    }
    .risk-card .badge {
        font-size: 2.1rem;
        line-height: 1;
    }
    .risk-card .risk-text h2 {
        margin: 0;
        font-size: 1.5rem;
    }
    .risk-card .risk-text p {
        margin: 0.15rem 0 0 0;
        opacity: 0.85;
        font-size: 0.92rem;
    }
    .risk-high {
        background: linear-gradient(90deg, rgba(214, 69, 65, 0.14), rgba(214, 69, 65, 0.03));
        border-left-color: #D64541;
    }
    .risk-low {
        background: linear-gradient(90deg, rgba(15, 155, 142, 0.14), rgba(15, 155, 142, 0.03));
        border-left-color: #0F9B8E;
    }
    .app-footer h4 {
        margin-bottom: 0.3rem;
    }
    .app-footer p {
        font-size: 0.87rem;
        opacity: 0.85;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load(os.path.join(MODEL_DIR, MODEL_FILE))


@st.cache_data
def load_metrics():
    path = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


st.title("🩺 Patient Risk Screening")
st.caption(
    "Predicts whether a patient falls into a **High Risk** or **Low Risk** category "
    "based on lifestyle and clinical measurements, to support early screening decisions."
)

predict_tab, insights_tab = st.tabs(["Risk Prediction", "Model Insights"])


#  Prediction form

with predict_tab:
    st.subheader("Patient Details")

    with st.form("patient_form"):
        vitals_col, lifestyle_col, background_col = st.columns(3)

        with vitals_col:
            st.markdown("**Vitals & Labs**")
            age = st.number_input("Age", min_value=0, max_value=120, value=45)
            bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.0, step=0.1)
            blood_pressure = st.number_input(
                "Blood Pressure (systolic)", min_value=60.0, max_value=250.0, value=120.0
            )
            cholesterol = st.number_input(
                "Cholesterol", min_value=100.0, max_value=350.0, value=200.0
            )
            glucose_level = st.number_input(
                "Glucose Level", min_value=50.0, max_value=250.0, value=100.0
            )
            heart_rate = st.number_input(
                "Heart Rate", min_value=40.0, max_value=200.0, value=72.0
            )

        with lifestyle_col:
            st.markdown("**Lifestyle**")
            sleep_hours = st.slider("Sleep Hours / day", 0.0, 14.0, 7.0, 0.5)
            exercise_hours = st.slider("Exercise Hours / week", 0.0, 14.0, 3.0, 0.5)
            water_intake = st.slider("Water Intake (glasses/day)", 0.0, 12.0, 8.0, 0.5)
            stress_level = st.slider("Stress Level", 0.0, 12.0, 5.0, 0.5)
            smoking = st.selectbox(
                "Smoking", options=list(LEVEL_LABELS.keys()), format_func=lambda x: LEVEL_LABELS[x]
            )
            alcohol = st.selectbox(
                "Alcohol Consumption", options=list(LEVEL_LABELS.keys()), format_func=lambda x: LEVEL_LABELS[x]
            )

        with background_col:
            st.markdown("**Background**")
            diet = st.selectbox(
                "Diet Quality", options=list(LEVEL_LABELS.keys()), format_func=lambda x: LEVEL_LABELS[x]
            )
            mental_health = st.selectbox(
                "Mental Health Concern", options=list(LEVEL_LABELS.keys()), format_func=lambda x: LEVEL_LABELS[x]
            )
            physical_activity = st.selectbox(
                "Physical Activity", options=list(LEVEL_LABELS.keys()), format_func=lambda x: LEVEL_LABELS[x]
            )
            medical_history = st.selectbox(
                "Medical History Severity", options=list(LEVEL_LABELS.keys()), format_func=lambda x: LEVEL_LABELS[x]
            )
            allergies = st.selectbox(
                "Allergies", options=list(LEVEL_LABELS.keys()), format_func=lambda x: LEVEL_LABELS[x]
            )
            diet_type = st.selectbox("Diet Type", ["Standard", "Vegetarian", "Vegan"])
            blood_group = st.selectbox("Blood Group", ["A", "AB", "B", "O"])

        submitted = st.form_submit_button("Predict Risk", type="primary", width="stretch")

    if submitted:
        row = {
            "Age": age,
            "BMI": bmi,
            "Blood_Pressure": blood_pressure,
            "Cholesterol": cholesterol,
            "Glucose_Level": glucose_level,
            "Heart_Rate": heart_rate,
            "Sleep_Hours": sleep_hours,
            "Exercise_Hours": exercise_hours,
            "Water_Intake": water_intake,
            "Stress_Level": stress_level,
            "Smoking": smoking,
            "Alcohol": alcohol,
            "Diet": diet,
            "MentalHealth": mental_health,
            "PhysicalActivity": physical_activity,
            "MedicalHistory": medical_history,
            "Allergies": allergies,
            "Diet_Type__Vegan": diet_type == "Vegan",
            "Diet_Type__Vegetarian": diet_type == "Vegetarian",
            "Blood_Group_AB": blood_group == "AB",
            "Blood_Group_B": blood_group == "B",
            "Blood_Group_O": blood_group == "O",
        }
        input_df = pd.DataFrame([row])[FEATURE_COLUMNS]

        model = load_model()
        prediction = model.predict(input_df)[0]
        label = risk_label(prediction)

        proba = None
        if hasattr(model, "predict_proba"):
            classes = list(model.classes_)
            proba = model.predict_proba(input_df)[0][classes.index(1)]

        st.divider()
        card_class = "risk-high" if prediction == 1 else "risk-low"
        icon = "⚠️" if prediction == 1 else "✅"
        subtext = (
            "Indicators suggest an elevated composite health risk — consider flagging "
            "this patient for closer clinical follow-up."
            if prediction == 1
            else "Indicators fall within a lower composite health risk range for this screening model."
        )
        st.markdown(
            f"""
            <div class="risk-card {card_class}">
                <div class="badge">{icon}</div>
                <div class="risk-text">
                    <h2>{label}</h2>
                    <p>{subtext}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if proba is not None:
            result_col1, result_col2 = st.columns([1, 2])
            with result_col1:
                st.metric("Model Confidence (High Risk probability)", f"{proba * 100:.1f}%")
            with result_col2:
                st.progress(float(proba))

        with st.expander("What does 'risk' mean here, and which model made this call?"):
            st.markdown(
                """
                **What "risk" refers to:** The model predicts a **composite health risk
                indicator** derived from this patient's vitals (BMI, blood pressure,
                cholesterol, glucose) and lifestyle factors (sleep, exercise, stress,
                smoking, alcohol). It is **not** a diagnosis of any single disease —
                think of it as a general early-screening flag, similar to a risk score
                used to prioritize who gets a closer clinical look first.

                **Model used:** `Random Forest Classifier`, hyperparameter-tuned with
                `GridSearchCV` (5-fold cross-validation, optimized for **Recall**).

                **Why Recall was prioritized:** In a screening context, missing a
                genuinely high-risk patient (a false negative) is more costly than
                flagging a low-risk patient for extra review (a false positive). See
                the **Model Insights** tab for the full comparison against Logistic
                Regression, KNN, Gradient Boosting, and a Voting Classifier.
                """
            )

        st.caption(
            "This tool supports early screening decisions and does not replace clinical judgment."
        )


#   Model insights

with insights_tab:
    st.subheader("Why Random Forest?")
    st.write(
        "Five models were compared. **Random Forest** was selected as the final model "
        "because it achieved the highest **Recall** — the priority metric here, since "
        "missing a high-risk patient (a false negative) is more costly than a false alarm."
    )

    metrics_data = load_metrics()
    if metrics_data:
        comparison_df = pd.DataFrame(metrics_data)[
            ["model", "accuracy", "precision", "recall", "f1"]
        ].sort_values("recall", ascending=False)
        comparison_df.columns = ["Model", "Accuracy", "Precision", "Recall", "F1 Score"]
        st.dataframe(
            comparison_df.style.format(
                {"Accuracy": "{:.1%}", "Precision": "{:.1%}", "Recall": "{:.1%}", "F1 Score": "{:.1%}"}
            ),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("Run `train_model.py` to generate the model comparison table.")

    st.divider()

    img_col1, img_col2 = st.columns(2)
    with img_col1:
        cm_path = os.path.join(ASSET_DIR, "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix — Random Forest")
        else:
            st.info("Run `train_model.py` to generate the confusion matrix plot.")

    with img_col2:
        fi_path = os.path.join(ASSET_DIR, "feature_importance.png")
        if os.path.exists(fi_path):
            st.image(fi_path, caption="Top 10 Feature Importances")
        else:
            st.info("Run `train_model.py` to generate the feature importance plot.")

    st.divider()
    st.markdown(
        "**Clinical takeaway:** By minimizing false negatives, this model supports "
        "earlier intervention for at-risk patients, reducing the chance that a "
        "genuinely high-risk case is missed during screening."
    )