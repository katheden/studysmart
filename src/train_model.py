"""
src/train_model.py
──────────────────
Trains Logistic Regression, Random Forest, and Gradient Boosting,
evaluates all three, picks the best, and saves it.
"""
import os, json
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics         import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

from preprocessing import load_and_preprocess

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

LABEL_MAP_INV = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}


def evaluate(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")
    cv  = cross_val_score(model, X_train, y_train,
                          cv=StratifiedKFold(5), scoring="f1_weighted").mean()

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1-score : {f1:.4f}  |  CV F1: {cv:.4f}")
    print(classification_report(y_test, y_pred,
                                target_names=["Low Risk", "Medium Risk", "High Risk"]))
    return {"name": name, "model": model,
            "accuracy": acc, "f1": f1, "cv_f1": cv}


def get_feature_importance(model, feature_names):
    """Return sorted (feature, importance) list."""
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_).mean(axis=0)
    else:
        return []
    pairs = sorted(zip(feature_names, imp), key=lambda x: x[1], reverse=True)
    return pairs


def main():
    print("Loading & preprocessing data …")
    X_train, X_test, y_train, y_test, scaler, features = load_and_preprocess(save=True)

    models = [
        ("Logistic Regression",
         LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),

        ("Random Forest",
         RandomForestClassifier(n_estimators=200, max_depth=8,
                                min_samples_leaf=1, max_features="sqrt",
                                class_weight="balanced", random_state=42)),

        ("Gradient Boosting",
         GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                    max_depth=4, random_state=42)),
    ]

    results = []
    for name, clf in models:
        r = evaluate(name, clf, X_train, y_train, X_test, y_test)
        results.append(r)

    # Pick best by CV F1
    best = max(results, key=lambda r: r["cv_f1"])
    print(f"\n>>> Best model: {best['name']} (CV F1 = {best['cv_f1']:.4f})")

    # Feature importance
    fi = get_feature_importance(best["model"], features)
    print("\nTop feature importances:")
    for feat, imp in fi[:8]:
        print(f"  {feat:25s}  {imp:.4f}")

    # Save
    joblib.dump(best["model"], os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(scaler,        os.path.join(MODEL_DIR, "scaler.pkl"))

    # Save metadata for app
    meta = {
        "best_model_name": best["name"],
        "accuracy":        round(best["accuracy"], 4),
        "f1_weighted":     round(best["f1"], 4),
        "cv_f1":           round(best["cv_f1"], 4),
        "features":        features,
        "feature_importance": [(f, round(float(i), 4)) for f, i in fi],
        "comparison": [
            {"model": r["name"],
             "accuracy": round(r["accuracy"], 4),
             "f1_weighted": round(r["f1"], 4),
             "cv_f1": round(r["cv_f1"], 4)}
            for r in results
        ]
    }
    with open(os.path.join(MODEL_DIR, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\nAll artefacts saved to models/")
    return best, features


if __name__ == "__main__":
    main()
