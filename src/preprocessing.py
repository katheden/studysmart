"""
src/preprocessing.py
────────────────────
Loads REAL Kaggle/UCI data (preferred) or synthetic fallback.
Auto-detects which files are present.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib, os

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
RAW_DIR   = os.path.join(BASE_DIR, "..", "data", "raw")
PROC_DIR  = os.path.join(BASE_DIR, "..", "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

RISK_ORDER     = {"Low Risk": 0, "Medium Risk": 1, "High Risk": 2}
RISK_ORDER_INV = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
FAM_MAP        = {"none": 0, "low": 1, "medium": 2, "high": 3}

FEATURES = [
    "study_time", "absences", "failures", "G1", "G2",
    "internet", "higher_edu", "activities", "romantic",
    "family_support_num", "gender_bin", "high_absence", "study_efficiency",
]


def grade_to_risk(g3):
    if g3 >= 14: return "Low Risk"
    if g3 >= 10: return "Medium Risk"
    return "High Risk"


def _engineer(df):
    df = df.copy()
    df["family_support_num"] = df["family_support"].map(FAM_MAP).fillna(2)
    df["gender_bin"]         = (df["gender"] == "F").astype(int)
    df["high_absence"]       = (df["absences"] > 10).astype(int)
    # Use only information available before the final grade (G3) to avoid data leakage.
    df["study_efficiency"]   = ((df["G1"] + df["G2"]) / 2) / (df["study_time"].clip(lower=0.5) + 0.5)
    return df


def _load_performance_csv():
    """Try real data first, fall back to synthetic."""
    real_path = os.path.join(RAW_DIR, "student_performance.csv")
    if os.path.exists(real_path):
        df = pd.read_csv(real_path)
        # Validate required columns exist
        required = ["G1", "G2", "G3", "study_time", "absences", "failures"]
        if all(c in df.columns for c in required):
            print(f"  ✓  Using REAL dataset: {real_path}  ({len(df)} rows)")
            return df, "real"
    
    # Fallback
    synth_path = os.path.join(RAW_DIR, "student_performance_synthetic.csv")
    if os.path.exists(synth_path):
        print(f"  ⚠  Using synthetic fallback: {synth_path}")
        return pd.read_csv(synth_path), "synthetic"
    
    raise FileNotFoundError(
        "No performance dataset found!\n"
        "Run: python data/raw/download_datasets.py\n"
        "or:  python data/raw/generate_data.py"
    )


def load_and_preprocess(save=True):
    print("Loading student performance data...")
    df, source = _load_performance_csv()

    # Add missing optional columns with defaults if not present
    defaults = {
        "gender":      "M",
        "family_support": "medium",
        "internet":    1,
        "higher_edu":  1,
        "activities":  0,
        "romantic":    0,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
            print(f"  ⚠  Column '{col}' missing → default: {default!r}")

    df["risk"]      = df["G3"].apply(grade_to_risk)
    df["risk_code"] = df["risk"].map(RISK_ORDER)
    df = _engineer(df)

    X = df[FEATURES].astype(float)
    y = df["risk_code"]

    scaler   = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=FEATURES)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    if save:
        X_train.to_csv(os.path.join(PROC_DIR, "X_train.csv"), index=False)
        X_test.to_csv(os.path.join(PROC_DIR,  "X_test.csv"),  index=False)
        y_train.to_csv(os.path.join(PROC_DIR, "y_train.csv"), index=False)
        y_test.to_csv(os.path.join(PROC_DIR,  "y_test.csv"),  index=False)
        joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
        print(f"  ✓  Data source: {source} | Rows: {len(df)} | Saved to data/processed/")

    return X_train, X_test, y_train, y_test, scaler, FEATURES


def preprocess_single_input(user_input, scaler, features):
    row = {
        "study_time":         float(user_input.get("study_time", 2)),
        "absences":           float(user_input.get("absences", 5)),
        "failures":           float(user_input.get("failures", 0)),
        "G1":                 float(user_input.get("G1", 10.0)),
        "G2":                 float(user_input.get("G2", 10.0)),
        "internet":           float(user_input.get("internet", 1)),
        "higher_edu":         float(user_input.get("higher_edu", 1)),
        "activities":         float(user_input.get("activities", 0)),
        "romantic":           float(user_input.get("romantic", 0)),
        "family_support_num": float(FAM_MAP.get(user_input.get("family_support","medium"), 2)),
        "gender_bin":         float(user_input.get("gender","F") == "F"),
        "high_absence":       float(user_input.get("absences", 5) > 10),
        "study_efficiency":   ((float(user_input.get("G1", 10.0)) + float(user_input.get("G2", 10.0))) / 2) /
                              (float(user_input.get("study_time", 2)) + 0.5),
    }
    df_row = pd.DataFrame([[row[f] for f in features]], columns=features)
    return pd.DataFrame(scaler.transform(df_row), columns=features)


if __name__ == "__main__":
    load_and_preprocess()
    print("Done.")
