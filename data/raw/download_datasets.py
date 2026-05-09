"""
data/raw/download_datasets.py
──────────────────────────────
Instructions + automated loader for the real Kaggle/UCI datasets.

═══════════════════════════════════════════════════════════════════
 STEP 1 — Download the datasets manually (free, no API key needed)
═══════════════════════════════════════════════════════════════════

Dataset 1 — UCI Student Performance (Cortez & Silva, 2008)
  URL: https://www.kaggle.com/datasets/larsen0966/student-performance-data-set
  OR:  https://archive.ics.uci.edu/dataset/320/student+performance
  Download: student.zip → extract → rename student-mat.csv → student_performance_raw.csv
  Place at: data/raw/student_performance_raw.csv

  Alternative direct URL (no login):
  https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip

Dataset 2 — Student Habits vs Academic Performance (Kaggle)
  URL: https://www.kaggle.com/datasets/jayaantanaath/student-habits-vs-academic-performance
  Download: student_habits_performance.csv
  Place at: data/raw/student_habits_raw.csv

  Alternative (same structure):
  https://www.kaggle.com/datasets/aryan208/student-habits-and-academic-performance-dataset

═══════════════════════════════════════════════════════════════════
 STEP 2 — Run this script to standardise column names
═══════════════════════════════════════════════════════════════════

  python data/raw/download_datasets.py

This produces:
  data/raw/student_performance.csv   ← ready for preprocessing.py
  data/raw/student_habits.csv        ← ready for EDA
"""

import os, sys
import pandas as pd
import numpy as np

RAW = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────────────────────
# Dataset 1: UCI Student Performance
# ─────────────────────────────────────────────────────────────────────────────
def adapt_uci_performance():
    """
    The UCI dataset already has exactly the columns we need.
    We just standardise the separator (it uses ';') and rename if needed.
    """
    path = os.path.join(RAW, "student_performance_raw.csv")
    if not os.path.exists(path):
        # Try semicolon-separated version (UCI default)
        path_semi = os.path.join(RAW, "student-mat.csv")
        if os.path.exists(path_semi):
            path = path_semi
        else:
            print("✗  student_performance_raw.csv not found.")
            print("   Please download from:")
            print("   https://www.kaggle.com/datasets/larsen0966/student-performance-data-set")
            print("   and place the file at: data/raw/student_performance_raw.csv")
            return False

    # Try comma first, then semicolon
    try:
        df = pd.read_csv(path, sep=",")
        if df.shape[1] < 5:
            raise ValueError
    except Exception:
        df = pd.read_csv(path, sep=";")

    print(f"  Loaded: {path}  →  {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    # UCI columns we need — map to our internal names
    col_map = {
        "sex":       "gender",
        "famsize":   "family_size",
        "Medu":      "mother_edu",
        "Fedu":      "father_edu",
        "studytime": "study_time",
        "failures":  "failures",
        "famsup":    "family_support_yn",
        "internet":  "internet_yn",
        "higher":    "higher_edu_yn",
        "activities":"activities_yn",
        "romantic":  "romantic_yn",
        "absences":  "absences",
        "G1": "G1", "G2": "G2", "G3": "G3",
    }

    df = df.rename(columns=col_map)

    # Standardise binary yes/no columns → 1/0
    for col in ["internet_yn", "higher_edu_yn", "activities_yn", "romantic_yn", "family_support_yn"]:
        if col in df.columns:
            df[col] = df[col].map({"yes": 1, "no": 0}).fillna(0).astype(int)

    # Rename back to our expected schema
    rename_final = {
        "internet_yn":   "internet",
        "higher_edu_yn": "higher_edu",
        "activities_yn": "activities",
        "romantic_yn":   "romantic",
        "family_support_yn": "family_support_raw",
    }
    df = df.rename(columns=rename_final)

    # family_support: UCI uses "famsup" (yes/no) — convert to ordinal using Medu/Fedu as proxy
    # If family_support_raw exists (yes/no), map to medium/low
    if "family_support_raw" in df.columns:
        df["family_support"] = df["family_support_raw"].map({1: "high", 0: "low"})
        # Refine using parent education level if available
        if "mother_edu" in df.columns:
            df.loc[df["mother_edu"] >= 4, "family_support"] = "high"
            df.loc[df["mother_edu"].between(2, 3), "family_support"] = "medium"
            df.loc[df["mother_edu"] <= 1, "family_support"] = "low"
    else:
        df["family_support"] = "medium"

    # Standardise gender
    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"F": "F", "M": "M", 1: "F", 0: "M"}).fillna("M")

    # Keep only expected columns; add defaults for missing ones
    expected = ["gender", "family_support", "internet", "higher_edu",
                "activities", "romantic", "absences", "study_time",
                "failures", "G1", "G2", "G3"]
    for col in expected:
        if col not in df.columns:
            df[col] = 0

    df = df[expected].copy()
    df = df.dropna(subset=["G3"])

    out = os.path.join(RAW, "student_performance.csv")
    df.to_csv(out, index=False)
    print(f"  ✓  Saved to {out}  ({df.shape[0]} rows)")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Dataset 2: Student Habits
# ─────────────────────────────────────────────────────────────────────────────
def adapt_habits():
    """
    The Kaggle 'Student Habits vs Academic Performance' dataset has columns like:
    study_hours_per_day, sleep_hours, social_media_hours, motivation_level,
    stress_level, exam_score, etc. We standardise to our internal schema.
    """
    path = os.path.join(RAW, "student_habits_raw.csv")
    if not os.path.exists(path):
        print("✗  student_habits_raw.csv not found.")
        print("   Please download from:")
        print("   https://www.kaggle.com/datasets/jayaantanaath/student-habits-vs-academic-performance")
        print("   and place it at: data/raw/student_habits_raw.csv")
        return False

    df = pd.read_csv(path)
    print(f"  Loaded: {path}  →  {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    # Flexible column mapping — handles both known variants of this dataset
    col_candidates = {
        "sleep_hours":        ["sleep_hours", "sleep_hours_per_night", "sleep_duration"],
        "social_media_hours": ["social_media_hours", "social_media_usage", "social_media_time"],
        "daily_study_hours":  ["study_hours_per_day", "daily_study_hours", "study_hours"],
        "motivation":         ["motivation_level", "motivation"],
        "stress_level":       ["stress_level", "stress"],
        "exam_anxiety":       ["exam_anxiety_score", "exam_anxiety", "anxiety_level"],
        "mental_load":        ["mental_health_rating", "mental_load", "mental_health"],
        "part_time_job":      ["part_time_job", "works_part_time"],
        "learning_style":     ["preferred_learning_type", "learning_style", "learning_type"],
    }

    out_df = pd.DataFrame()
    for target, candidates in col_candidates.items():
        found = None
        for c in candidates:
            # case-insensitive search
            matches = [col for col in df.columns if col.lower() == c.lower()]
            if matches:
                found = matches[0]
                break
        if found:
            out_df[target] = df[found]
        else:
            # Provide a sensible default
            defaults = {
                "sleep_hours": 7.0, "social_media_hours": 2.5,
                "daily_study_hours": 2.0, "motivation": "medium",
                "stress_level": 5, "exam_anxiety": 5,
                "mental_load": 5.0, "part_time_job": 0,
                "learning_style": "visual",
            }
            out_df[target] = defaults[target]
            print(f"  ⚠  Column '{target}' not found — using default value")

    # Standardise motivation to low/medium/high if numeric
    if out_df["motivation"].dtype != object:
        out_df["motivation"] = pd.cut(
            out_df["motivation"].astype(float),
            bins=[0, 3, 7, 10], labels=["low", "medium", "high"]
        ).astype(str)

    out = os.path.join(RAW, "student_habits.csv")
    out_df.to_csv(out, index=False)
    print(f"  ✓  Saved to {out}  ({out_df.shape[0]} rows)")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== StudySmart Dataset Adapter ===\n")

    print("📊 Dataset 1: UCI Student Performance")
    ok1 = adapt_uci_performance()

    print("\n📊 Dataset 2: Student Habits")
    ok2 = adapt_habits()

    print()
    if ok1 and ok2:
        print("✅  Both datasets ready. You can now run:")
        print("    python src/train_model.py")
    else:
        print("⚠️  One or more datasets missing.")
        print("   Please follow the download instructions above,")
        print("   then re-run this script.")
        print()
        print("   Alternatively, run without real data:")
        print("    python data/raw/generate_data.py   ← synthetic fallback")
        print("    python src/train_model.py")
