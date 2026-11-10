# ============================================================
# PROJECT 1: Customer Churn Prediction & Analytics
# Stack: Python, Pandas, NumPy, Scikit-learn, Matplotlib
#
# WHAT IT DOES
#   1. Loads customer data (synthetic Telco-style data out of the box)
#   2. EDA - churn distribution, churn by contract type
#   3. Preprocessing - one-hot encoding + scaling
#   4. Trains Logistic Regression, Decision Tree, Random Forest
#   5. Cross-validation + GridSearchCV hyperparameter tuning
#   6. Evaluation - Precision, Recall, F1-score, ROC-AUC
#
# TO USE A REAL DATASET (e.g. Kaggle "Telco Customer Churn"):
#   df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
#   df = df.drop(columns=["customerID", "TotalCharges"(blank->NaN handled below)])
#   Just replace the "LOAD DATA" section. Columns expected:
#   gender, tenure, MonthlyCharges, Contract, InternetService, TechSupport, Churn
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(IMG_DIR, exist_ok=True)  # save plots to files, no display needed
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix, RocCurveDisplay)

RANDOM_STATE = 42

# ---------------- 1. LOAD DATA ----------------
# --- Option A: synthetic data (runs anywhere, no download needed) ---
def make_churn_data(n=2000, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "gender":          rng.choice(["Male", "Female"], n),
        "tenure":          rng.integers(1, 73, n),                    # months with company
        "MonthlyCharges":  np.round(rng.uniform(20, 120, n), 2),
        "Contract":        rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20]),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.20]),
        "TechSupport":     rng.choice(["Yes", "No"], n, p=[0.35, 0.65]),
    })
    # Churn probability depends on features (month-to-month + high charges + low tenure => more churn)
    logit = (-1.2
             + 1.6 * (df["Contract"] == "Month-to-month")
             + 0.02 * (df["MonthlyCharges"] - 60)
             - 0.045 * df["tenure"]
             + 0.9 * (df["InternetService"] == "Fiber optic")
             - 0.8 * (df["TechSupport"] == "Yes")
             + rng.normal(0, 0.6, n))
    df["Churn"] = (1 / (1 + np.exp(-logit)) > 0.5).astype(int)  # 1 = churned
    return df

df = make_churn_data()
# --- Option B: real data ---
# df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
# df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
# df = df.dropna()
# df["Churn"] = (df["Churn"] == "Yes").astype(int)
# df = df[["gender", "tenure", "MonthlyCharges", "Contract", "InternetService", "TechSupport", "Churn"]]

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nChurn rate:\n", df["Churn"].value_counts(normalize=True).round(2))

# ---------------- 2. EDA ----------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df["Churn"].value_counts().plot(kind="bar", ax=axes[0], color=["steelblue", "tomato"])
axes[0].set_title("Churn Distribution"); axes[0].set_xticklabels(["Stayed", "Churned"], rotation=0)
df.groupby("Contract")["Churn"].mean().sort_values().plot(kind="bar", ax=axes[1], color="salmon")
axes[1].set_title("Churn Rate by Contract Type"); axes[1].set_ylabel("Churn rate")
plt.tight_layout(); plt.savefig(os.path.join(IMG_DIR, "churn_eda.png")); plt.close()

# ---------------- 3. PREPROCESSING ----------------
X = df.drop(columns=["Churn"])
y = df["Churn"]
num_cols = ["tenure", "MonthlyCharges"]
cat_cols = ["gender", "Contract", "InternetService", "TechSupport"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

# ---------------- 4. TRAIN 3 MODELS ----------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree":       DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest":       RandomForestClassifier(random_state=RANDOM_STATE),
}

results = {}
for name, model in models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    cv_auc = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc").mean()  # 5-fold CV
    results[name] = {
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_proba),
        "cv_roc_auc": cv_auc,
    }
    print(f"\n=== {name} ===\n{classification_report(y_test, y_pred)}")

print("\nModel comparison:\n", pd.DataFrame(results).T.round(3))

# ---------------- 5. HYPERPARAMETER TUNING (Random Forest) ----------------
rf_pipe = Pipeline([("prep", preprocessor),
                    ("model", RandomForestClassifier(random_state=RANDOM_STATE))])
param_grid = {"model__n_estimators": [100, 300], "model__max_depth": [None, 5, 10]}
grid = GridSearchCV(rf_pipe, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
grid.fit(X_train, y_train)
print("\nBest RF params:", grid.best_params_, "| CV ROC-AUC:", round(grid.best_score_, 3))

best = grid.best_estimator_
y_pred = best.predict(X_test)
print("\nTuned Random Forest on test set:")
print("Precision:", round(precision_score(y_test, y_pred), 3),
      "| Recall:", round(recall_score(y_test, y_pred), 3),
      "| F1:", round(f1_score(y_test, y_pred), 3),
      "| ROC-AUC:", round(roc_auc_score(y_test, best.predict_proba(X_test)[:, 1]), 3))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# ---------------- 6. PLOTS ----------------
RocCurveDisplay.from_estimator(best, X_test, y_test, name="Tuned Random Forest")
plt.title("ROC Curve - Customer Churn"); plt.savefig(os.path.join(IMG_DIR, "churn_roc_curve.png")); plt.close()

# Feature importance (top 10)
importances = best.named_steps["model"].feature_importances_
feat_names = best.named_steps["prep"].get_feature_names_out()
top = pd.Series(importances, index=feat_names).sort_values().tail(10)
top.plot(kind="barh", figsize=(8, 5), color="teal")
plt.title("Top 10 Feature Importances"); plt.tight_layout(); plt.savefig(os.path.join(IMG_DIR, "churn_feature_importance.png")); plt.close()

print("\nSaved plots: churn_eda.png, churn_roc_curve.png, churn_feature_importance.png")
