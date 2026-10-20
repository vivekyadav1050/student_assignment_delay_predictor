from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
LR_PATH = MODEL_DIR / "logistic_regression_pipeline.joblib"
TREE_PATH = MODEL_DIR / "decision_tree_pipeline.joblib"

st.set_page_config(page_title="Assignment Delay Predictor", page_icon="📚", layout="centered")
st.title("📚 Student Assignment Delay Predictor")
st.write("Estimate whether an assignment may be submitted late using a trained ML model.")
st.warning("Learning demo only: this project uses synthetic sample data. Predictions are estimates, not guarantees.")

@st.cache_resource
def load_models():
    if not LR_PATH.exists() or not TREE_PATH.exists():
        st.error("Trained models not found. Run the Jupyter notebook through the model-saving section first.")
        st.stop()
    return joblib.load(LR_PATH), joblib.load(TREE_PATH)

lr_model, tree_model = load_models()

st.subheader("Predict one assignment")
with st.form("prediction_form"):
    pending_assignments = st.number_input("Pending assignments", min_value=0, max_value=100, value=5)
    study_hours = st.number_input("Average study hours per day", min_value=0.0, max_value=24.0, value=1.0, step=0.5)
    previous_late_count = st.number_input("Previous late submissions", min_value=0, max_value=100, value=4)
    days_remaining = st.number_input("Days remaining until deadline", min_value=0, max_value=365, value=2)
    assignment_difficulty = st.selectbox("Assignment difficulty", ["Easy", "Medium", "Hard"])
    attendance_percentage = st.slider("Attendance percentage", min_value=0.0, max_value=100.0, value=65.0)
    selected_model = st.selectbox("Primary model", ["Logistic Regression", "Decision Tree"])
    submitted = st.form_submit_button("Predict")

if submitted:
    row = pd.DataFrame([{
        "pending_assignments": pending_assignments,
        "study_hours": study_hours,
        "previous_late_count": previous_late_count,
        "days_remaining": days_remaining,
        "assignment_difficulty": assignment_difficulty,
        "attendance_percentage": attendance_percentage,
    }])
    model = lr_model if selected_model == "Logistic Regression" else tree_model
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0, 1])
    if prediction == 1:
        st.error("Prediction: Likely to submit late")
    else:
        st.success("Prediction: Likely to submit on time")
    st.metric("Estimated probability of late submission", f"{probability:.1%}")
    st.caption("This probability is a model estimate and should not be treated as certainty.")
    st.subheader("Compare both models")
    comparison = []
    for name, candidate in [("Logistic Regression", lr_model), ("Decision Tree", tree_model)]:
        pred = int(candidate.predict(row)[0])
        prob = float(candidate.predict_proba(row)[0, 1])
        comparison.append({"Model": name, "Prediction": "Likely late" if pred else "Likely on time",
                           "Estimated late probability": round(prob, 3)})
    st.dataframe(pd.DataFrame(comparison), use_container_width=True)

st.divider()
st.subheader("Batch predictions from CSV")
st.write("Upload a CSV containing the six input columns. Do not include the `submitted_late` target column.")
uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
required_columns = ["pending_assignments", "study_hours", "previous_late_count",
                    "days_remaining", "assignment_difficulty", "attendance_percentage"]
if uploaded is not None:
    try:
        batch = pd.read_csv(uploaded)
        missing = set(required_columns) - set(batch.columns)
        if missing:
            st.error(f"Missing required columns: {', '.join(sorted(missing))}")
        else:
            batch["predicted_late"] = lr_model.predict(batch[required_columns]).astype(int)
            batch["late_probability"] = lr_model.predict_proba(batch[required_columns])[:, 1]
            st.dataframe(batch.head(20), use_container_width=True)
            st.download_button("Download predictions CSV",
                               data=batch.to_csv(index=False).encode("utf-8"),
                               file_name="assignment_predictions.csv",
                               mime="text/csv")
    except Exception as exc:
        st.error(f"Could not read or process this CSV: {exc}")

st.divider()
st.subheader("How it works")
st.markdown("""
1. Enter assignment and workload details.
2. The selected trained model processes the inputs.
3. The app displays a class prediction and estimated late probability.
4. You can upload a CSV to predict multiple assignments.

**Important limitation:** The included dataset is synthetic. Replace it with suitable historical data and evaluate it on genuinely unseen records before drawing real-world conclusions.
""")
