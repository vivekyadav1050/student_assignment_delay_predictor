# Student Assignment Delay Predictor

A beginner-friendly machine learning project that compares Logistic Regression and Decision Tree classification for predicting whether an assignment is likely to be late.

## Included files

- `student_assignment_delay_predictor.ipynb`: detailed notebook walkthrough. Every code cell contains at most four non-empty lines.
- `data/assignments_synthetic.csv`: 1,200 synthetic sample rows.
- `app.py`: Streamlit web app for single and batch predictions.
- `requirements.txt`: Python dependencies.
- `models/`: saved model pipelines are created after running the notebook through its save-model section.

## Important dataset note

The included CSV is synthetic and is for learning only. Its metrics do not represent real-world performance. Do not use predictions to punish students or make high-stakes decisions.

## Run locally on Windows

Open a terminal in this project folder:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Open the notebook:

```bash
jupyter notebook student_assignment_delay_predictor.ipynb
```

Run the notebook cells from top to bottom. It loads the CSV from `data/`, trains both models, evaluates them, and saves model pipelines to `models/`.

Then start the Streamlit app:

```bash
streamlit run app.py
```

The app requires these saved files:
- `models/logistic_regression_pipeline.joblib`
- `models/decision_tree_pipeline.joblib`

## Dataset columns

- `pending_assignments`: count of pending assignments
- `study_hours`: average daily study hours
- `previous_late_count`: count of previous late submissions
- `days_remaining`: days until deadline
- `assignment_difficulty`: Easy, Medium, or Hard
- `attendance_percentage`: attendance percentage
- `submitted_late`: target (`0` = on time, `1` = late)

## Evaluation

The notebook reports accuracy, precision, recall, F1-score, classification reports, and confusion matrices for both models on a held-out test split.

## Limitations and responsible use

Use real data only with appropriate permission, privacy protections, and careful validation. Test on later unseen assignments when possible. Probabilities are estimates, not guarantees.
