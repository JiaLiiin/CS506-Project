"""
Imputation/cleaning script

Goal:
- Handle remaining missing values after feature selection.
- Apply appropriate imputation strategies based on feature type.
- Output a fully cleaned dataset ready for modeling.

Input:
- Data/owid-energy-data-selected.csv

Output:
- Data/owid-energy-data-clean.csv
"""

import pandas as pd

# -------------------------------
# 1. Load dataset
# -------------------------------
input_file = "Data/owid-energy-data-selected.csv"
df = pd.read_csv(input_file)

# -------------------------------
# 2. Sort data (important for time-based imputation)
# -------------------------------
df = df.sort_values(["country", "year"])

# -------------------------------
# 3. Impute GDP-related features
# -------------------------------
# Use interpolation to preserve trends, then ffill/bfill for edge cases

gdp_cols = ["gdp", "gdp_per_capita"]

df[gdp_cols] = df.groupby("country")[gdp_cols].transform(
    lambda x: x.interpolate().ffill().bfill()
)

# -------------------------------
# 4. Impute energy share features
# -------------------------------
# Use forward/backward fill due to low missingness and stability

share_cols = [
    "renewables_share_energy",
    "hydro_share_energy",
    "wind_share_energy",
    "fossil_share_energy",
    "low_carbon_share_energy",
    "solar_share_energy"
]

df[share_cols] = df.groupby("country")[share_cols].transform(
    lambda x: x.ffill().bfill()
)

# -------------------------------
# 5. Final check
# -------------------------------
missing_summary = df.isna().sum()
print("Missing values after imputation:\n", missing_summary)

# -------------------------------
# 6. Save cleaned dataset
# -------------------------------
output_file = "Data/owid-energy-data-clean.csv"
df.to_csv(output_file, index=False)

print(f"\nCleaned dataset saved to: {output_file}")
print(f"Final shape: {df.shape}")
print(f"Number of countries: {df['country'].nunique()}")