"""
Train, evaluate, and plot Random Forest on energy dataset

- Same time-based split
- Same features
- Same evaluation + plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

# Paths and config
DATA_PATH = "Data/owid-energy-data-clean.csv"
OUTPUT_PATH = "results/"
FIG_PATH = "figures/"
MODEL_NAME = "random_forest"

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)
df = df.sort_values(["country", "year"])

# Time-based train/test split
train = df[df["year"] <= 2012]
test = df[df["year"] > 2012]

feature_cols = [
    "year", "log_population", "log_gdp_per_capita",
    "coal_share_energy", "gas_share_energy", "oil_share_energy",
    "biofuel_share_energy",
    "hydro_share_energy", "solar_share_energy", "wind_share_energy",
    "nuclear_share_energy",
]

X_train = train[feature_cols]
X_test = test[feature_cols]

y_train = np.log1p(train["energy_per_capita"])
y_test  = np.log1p(test["energy_per_capita"])

# Train Random Forest
model = RandomForestRegressor(
    n_estimators=200,       # number of trees
    max_depth=None,         # let trees grow fully
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1               # use all cores
)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_pred_actual = np.expm1(y_pred)
y_test_actual = np.expm1(y_test)

results_df = pd.DataFrame({
    "year": test["year"],
    "country": test["country"],
    "actual_energy_per_capita": y_test_actual,
    "predicted_energy_per_capita": y_pred_actual
})

results_file = os.path.join(OUTPUT_PATH, f"{MODEL_NAME}_predictions.csv")
results_df.to_csv(results_file, index=False)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
r2 = r2_score(y_test_actual, y_pred_actual)

metrics_df = pd.DataFrame({
    "model": ["Random Forest"],
    "rmse": [rmse],
    "r2": [r2]
})

metrics_file = os.path.join(OUTPUT_PATH, f"{MODEL_NAME}_metrics.csv")
metrics_df.to_csv(metrics_file, index=False)

print(f"Saved predictions to {results_file}")
print(metrics_df)

# Actual vs Predicted Plot 
plt.figure(figsize=(10, 6))
plt.scatter(
    results_df["actual_energy_per_capita"], 
    results_df["predicted_energy_per_capita"], 
    alpha=0.4, 
    s=40
)

sns.regplot(
    x="actual_energy_per_capita", 
    y="predicted_energy_per_capita", 
    data=results_df,
    scatter=False,
    line_kws={"linewidth": 2, "linestyle": "--"}
)

plt.xlabel("Actual Energy per Capita")
plt.ylabel("Predicted Energy per Capita")
plt.title("Actual vs Predicted Energy per Capita (Random Forest)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_PATH, f"{MODEL_NAME}_actual_vs_pred.png"))
plt.close()

# Residual Plot
residuals = results_df["actual_energy_per_capita"] - results_df["predicted_energy_per_capita"]
results_df["residuals"] = residuals

plt.figure(figsize=(10, 6))
plt.scatter(
    results_df["predicted_energy_per_capita"], 
    results_df["residuals"], 
    alpha=0.4, 
    s=40
)

plt.axhline(0, linestyle='--', linewidth=2)
plt.xlabel("Predicted Energy per Capita")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted (Random Forest)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_PATH, f"{MODEL_NAME}_residuals.png"))
plt.close()

print(f"Saved figures to {FIG_PATH}")