import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

# =====================
# Load dataset
# =====================
df = pd.read_csv("data.csv")

print(df.head())

# =====================
# Preprocessing
# =====================

# Drop Loan_ID since it's just an identifier
if "Loan_ID" in df.columns:
    df.drop(columns=["Loan_ID"], inplace=True)

# Encode target variable (Y = 1 approved, N = 0 not approved)
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

# Fill categorical columns with mode
cat_cols = df.select_dtypes(include="object").columns
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Fill numerical columns with median
num_cols = df.select_dtypes(include=np.number).columns
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Label encode categorical features
le = LabelEncoder()
cat_features = df.select_dtypes(include="object").columns
for col in cat_features:
    df[col] = le.fit_transform(df[col])

# Cap outliers using IQR method
def cap_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
    return df

for col in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]:
    df = cap_outliers(df, col)

# =====================
# Feature engineering
# =====================
df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
df["EMI"] = df["LoanAmount"] / df["Loan_Amount_Term"]
df["BalanceIncome"] = df["TotalIncome"] - (df["EMI"] * 1000)

# =====================
# Column split
# =====================
X = df.drop(columns=["Loan_Status"])
y = df["Loan_Status"]

feature_names = list(X.columns)

# =====================
# Train-test split
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =====================
# Random Forest Model
# =====================
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=10, min_samples_split=2, random_state=42, n_jobs=-1
)

# =====================
# Full Pipeline
# =====================
rf_pipeline = Pipeline(steps=[("scaler", StandardScaler()), ("model", rf_model)])

# =====================
# Hyperparameter Tuning
# =====================
param_grid = {
    "model__n_estimators": [50, 100, 200],
    "model__max_depth": [5, 10, 15, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
}

grid_search = GridSearchCV(
    rf_pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1
)

grid_search.fit(X_train, y_train)

print(f"\nBest parameters found: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

best_pipeline = grid_search.best_estimator_

# =====================
# Evaluation
# =====================
y_pred = best_pipeline.predict(X_test)
y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_pred_proba)

print(f"\nAccuracy: {acc:.4f}")
print(f"ROC AUC Score: {roc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Approved", "Approved"]))

# =====================
# Save model (IMPORTANT)
# =====================
with open("loan_model.pkl", "wb") as f:
    pickle.dump(best_pipeline, f)

with open("feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)

print("✅ Random Forest pipeline saved as loan_model.pkl")
print("✅ Feature names saved as feature_names.pkl")
