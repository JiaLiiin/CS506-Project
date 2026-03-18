"""
Train, evaluate, and plot Linear Regression on energy dataset

- Time-based train/test split (1965–2012 train, 2013–2024 test)
- Trains Linear Regression
- Evaluates RMSE and R²
- Plots Actual vs Predicted and Residuals
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

# -------------------------------
# Paths and config
# -------------------------------
DATA_PATH = "Data/owid-energy-data-clean.csv"
OUTPUT_PATH = "results/"
FIG_PATH = "figures/"
MODEL_NAME = "linear_regression"

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

# -------------------------------
# Load data
# -------------------------------
df = pd.read_csv(DATA_PATH)
df = df.sort_values(["country", "year"])

# -------------------------------
# Time-based train/test split (80/20)
# -------------------------------
train = df[df["year"] <= 2012]
test = df[df["year"] > 2012]

X_train = train.drop(columns=["energy_per_capita", "country"])
y_train = train["energy_per_capita"]

X_test = test.drop(columns=["energy_per_capita", "country"])
y_test = test["energy_per_capita"]

# -------------------------------
# Train Linear Regression
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# Predict
# -------------------------------
y_pred = model.predict(X_test)

results_df = pd.DataFrame({
    "year": test["year"],
    "country": test["country"],
    "actual_energy_per_capita": y_test,
    "predicted_energy_per_capita": y_pred
})

results_file = os.path.join(OUTPUT_PATH, f"{MODEL_NAME}_predictions.csv")
results_df.to_csv(results_file, index=False)

# -------------------------------
# Evaluate
# -------------------------------
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

metrics_df = pd.DataFrame({
    "model": ["Linear Regression"],
    "rmse": [rmse],
    "r2": [r2]
})

metrics_file = os.path.join(OUTPUT_PATH, f"{MODEL_NAME}_metrics.csv")
metrics_df.to_csv(metrics_file, index=False)

print(f"Saved predictions to {results_file}")
print(metrics_df)

# -------------------------------
# Actual vs Predicted Plot 
# -------------------------------
plt.figure(figsize=(10, 6))
# Scatter with transparency for overlapping points
plt.scatter(
    results_df["actual_energy_per_capita"], 
    results_df["predicted_energy_per_capita"], 
    alpha=0.4, 
    s=40, 
    color='steelblue'
)
# Regression line
sns.regplot(
    x="actual_energy_per_capita", 
    y="predicted_energy_per_capita", 
    data=results_df,
    scatter=False,
    color="black",
    line_kws={"linewidth": 2, "linestyle": "--"}
)
plt.xlabel("Actual Energy per Capita")
plt.ylabel("Predicted Energy per Capita")
plt.title("Actual vs Predicted Energy per Capita (Linear Regression)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_PATH, f"{MODEL_NAME}_actual_vs_pred.png"))
plt.close()

# -------------------------------
# Residual Plot
# -------------------------------
residuals = results_df["actual_energy_per_capita"] - results_df["predicted_energy_per_capita"]
results_df["residuals"] = residuals

plt.figure(figsize=(10, 6))
plt.scatter(
    results_df["predicted_energy_per_capita"], 
    results_df["residuals"], 
    alpha=0.4, 
    s=40, 
    color='tomato'
)
plt.axhline(0, color='black', linestyle='--', linewidth=2)
plt.xlabel("Predicted Energy per Capita")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted (Linear Regression)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_PATH, f"{MODEL_NAME}_residuals.png"))
plt.close()

print(f"Saved figures to {FIG_PATH}")