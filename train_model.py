
import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model_utils import FEATURE_COLUMNS, TARGET_COLUMN

DATA_PATH = "dataset.csv"
MODEL_DIR = "models"
ASSET_DIR = "assets"


def evaluate(name, y_test, y_pred):
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(ASSET_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    all_metrics = []

    # --- Logistic Regression ---
    log_reg = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(solver="liblinear", max_iter=1000)),
        ]
    )
    log_reg.fit(X_train, y_train)
    all_metrics.append(evaluate("logistic_regression", y_test, log_reg.predict(X_test)))

    # --- KNN ---
    knn = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5, metric="euclidean")),
        ]
    )
    knn.fit(X_train, y_train)
    all_metrics.append(evaluate("knn", y_test, knn.predict(X_test)))

    # --- Random Forest (GridSearchCV, tuned for Recall) — FINAL MODEL ---
    param_grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_split": [2, 5],
        "clf__min_samples_leaf": [1, 2],
    }
    rf_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=42)),
        ]
    )
    grid_search = GridSearchCV(
        estimator=rf_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="recall",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    print("Best RF Params:", grid_search.best_params_)
    print("Best CV Recall:", round(grid_search.best_score_, 4))

    best_rf = grid_search.best_estimator_
    y_pred_rf = best_rf.predict(X_test)
    rf_metrics = evaluate("random_forest", y_test, y_pred_rf)
    all_metrics.append(rf_metrics)

    # --- Gradient Boosting ---
    gb = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                GradientBoostingClassifier(
                    n_estimators=150, learning_rate=0.1, max_depth=3, random_state=42
                ),
            ),
        ]
    )
    gb.fit(X_train, y_train)
    all_metrics.append(evaluate("gradient_boosting", y_test, gb.predict(X_test)))

    # --- Voting Classifier ---
    voting_clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                VotingClassifier(
                    estimators=[
                        ("lr", LogisticRegression(max_iter=1000, solver="liblinear")),
                        ("knn", KNeighborsClassifier(n_neighbors=5)),
                        ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
                    ],
                    voting="soft",
                ),
            ),
        ]
    )
    voting_clf.fit(X_train, y_train)
    all_metrics.append(evaluate("voting_classifier", y_test, voting_clf.predict(X_test)))

    # --- Save final model (Random Forest -- highest recall) ---
    joblib.dump(best_rf, os.path.join(MODEL_DIR, "random_forest.pkl"))

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    for m in all_metrics:
        print(f"--- {m['model']} ---")
        print(m)

    # --- Confusion matrix plot (final model) ---
    cm = confusion_matrix(y_test, y_pred_rf)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="coolwarm",
        xticklabels=["Low Risk", "High Risk"],
        yticklabels=["Low Risk", "High Risk"],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title("Confusion Matrix - Random Forest (Final Model)")
    plt.tight_layout()
    plt.savefig(os.path.join(ASSET_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    # --- Feature importance plot ---
    importance = pd.DataFrame(
        {
            "Feature": FEATURE_COLUMNS,
            "Importance": best_rf.named_steps["clf"].feature_importances_,
        }
    ).sort_values(by="Importance", ascending=False).head(10)

    plt.figure(figsize=(9, 6))
    sns.barplot(data=importance, x="Importance", y="Feature", color="teal")
    plt.title("Top 10 Most Important Features - Random Forest")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(ASSET_DIR, "feature_importance.png"), dpi=150)
    plt.close()

    print(f"\nSaved final model + metrics.json to '{MODEL_DIR}/'")
    print(f"Saved confusion_matrix.png + feature_importance.png to '{ASSET_DIR}/'")


if __name__ == "__main__":
    main()
