from pathlib import Path
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "loan_prediction_dataset.csv"
MODEL_PATH = ROOT / "models.joblib"

app = Flask(__name__)

CATEGORICAL = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
NUMERIC = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
FEATURES = CATEGORICAL + NUMERIC

def train_models():
    data = pd.read_csv(DATA_PATH)
    X = data[FEATURES]
    y = data["Loan_Status"].map({"N": 0, "Y": 1})
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    preprocessor = ColumnTransformer([
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), CATEGORICAL),
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), NUMERIC)
    ])
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=8, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=250, max_depth=10, min_samples_leaf=4, class_weight="balanced", random_state=42)
    }
    models, metrics = {}, {}
    for name, estimator in candidates.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        models[name] = pipe
        metrics[name] = {
            "accuracy": round(accuracy_score(y_test, pred)*100, 2),
            "precision": round(precision_score(y_test, pred, zero_division=0)*100, 2),
            "recall": round(recall_score(y_test, pred, zero_division=0)*100, 2),
            "f1": round(f1_score(y_test, pred, zero_division=0)*100, 2),
            "confusion_matrix": confusion_matrix(y_test, pred, labels=[0,1]).tolist()
        }
    # Select using F1 to balance precision and recall for this demonstration.
    best_name = max(metrics, key=lambda k: metrics[k]["f1"])
    bundle = {"models": models, "metrics": metrics, "best_name": best_name, "feature_names": FEATURES}
    joblib.dump(bundle, MODEL_PATH)
    return bundle

def get_bundle():
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass
    return train_models()

@app.route("/", methods=["GET", "POST"])
def index():
    bundle = get_bundle()
    metrics = bundle["metrics"]
    best_name = bundle["best_name"]
    result = None
    form = {
        "Gender": "Male", "Married": "Yes", "Dependents": "0", "Education": "Graduate",
        "Self_Employed": "No", "ApplicantIncome": "6000", "CoapplicantIncome": "0",
        "LoanAmount": "120", "Loan_Amount_Term": "360", "Credit_History": "1",
        "Property_Area": "Semiurban"
    }
    if request.method == "POST":
        form = {key: request.form.get(key, default) for key, default in form.items()}
        try:
            row = {
                "Gender": form["Gender"], "Married": form["Married"], "Dependents": form["Dependents"],
                "Education": form["Education"], "Self_Employed": form["Self_Employed"],
                "ApplicantIncome": float(form["ApplicantIncome"]),
                "CoapplicantIncome": float(form["CoapplicantIncome"]),
                "LoanAmount": float(form["LoanAmount"]),
                "Loan_Amount_Term": float(form["Loan_Amount_Term"]),
                "Credit_History": int(form["Credit_History"]),
                "Property_Area": form["Property_Area"]
            }
            if any(row[k] < 0 for k in ["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term"]):
                raise ValueError("Income, loan amount and loan term cannot be negative.")
            if row["Loan_Amount_Term"] <= 0:
                raise ValueError("Loan term must be greater than zero.")
            input_df = pd.DataFrame([row], columns=FEATURES)
            model = bundle["models"][best_name]
            prediction = int(model.predict(input_df)[0])
            probability = float(model.predict_proba(input_df)[0][1] * 100)
            result = {
                "approved": prediction == 1,
                "probability": round(probability, 1),
                "model": best_name,
                "loan_category": request.form.get("Loan_Category", "Home Loans"),
                "loan_type": request.form.get("Loan_Type", "Home Purchase Loan")
            }
        except (ValueError, TypeError) as exc:
            result = {"error": str(exc) or "Please check the entered values."}
    selected_category = request.form.get("Loan_Category", "Home Loans") if request.method == "POST" else "Home Loans"
    selected_loan_type = request.form.get("Loan_Type", "Home Purchase Loan") if request.method == "POST" else "Home Purchase Loan"
    return render_template("index.html", metrics=metrics, best_name=best_name, result=result, form=form,
                           selected_category=selected_category, selected_loan_type=selected_loan_type)

@app.route("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    get_bundle()
    app.run(debug=True)
