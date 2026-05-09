"""
src/predict.py
──────────────
Load saved model + scaler and run inference for a single student.
Returns prediction label, probabilities, and top feature importances.
"""
import os, json
import numpy as np
import joblib

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

RISK_LABELS = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}


def load_artifacts():
    model  = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    with open(os.path.join(MODEL_DIR, "model_meta.json")) as f:
        meta = json.load(f)
    return model, scaler, meta


def predict_student(user_input: dict):
    """
    Full prediction pipeline for one student.

    Returns
    -------
    dict with keys:
        risk_label       str   e.g. "High Risk"
        risk_code        int   0/1/2
        confidence       float probability of predicted class
        probabilities    dict  {label: prob}
        top_features     list  [(feature_name, importance), …]
        model_name       str
    """
    from preprocessing import preprocess_single_input, FEATURES

    model, scaler, meta = load_artifacts()
    X = preprocess_single_input(user_input, scaler, FEATURES)

    probs     = model.predict_proba(X)[0]
    risk_code = int(np.argmax(probs))
    risk_label = RISK_LABELS[risk_code]
    confidence = float(probs[risk_code])

    probabilities = {RISK_LABELS[i]: float(p) for i, p in enumerate(probs)}

    # Feature importance from saved metadata
    top_features = meta.get("feature_importance", [])

    return {
        "risk_label":    risk_label,
        "risk_code":     risk_code,
        "confidence":    confidence,
        "probabilities": probabilities,
        "top_features":  top_features,
        "model_name":    meta.get("best_model_name", "Model"),
    }


if __name__ == "__main__":
    sample = {
        "study_time": 2, "absences": 12, "failures": 1,
        "G1": 9.0, "G2": 8.5, "internet": 1,
        "higher_edu": 1, "activities": 0, "romantic": 1,
        "family_support": "low", "gender": "M",
    }
    result = predict_student(sample)
    print(f"Prediction : {result['risk_label']}")
    print(f"Confidence : {result['confidence']:.1%}")
    print(f"Top factors: {result['top_features'][:3]}")
