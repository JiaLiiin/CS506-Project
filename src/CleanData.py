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

# Load dataset
input_file = "Data/owid-energy-data-selected.csv"
df = pd.read_csv(input_file)

# Sort data (important for time-based imputation)
df = df.sort_values(["country", "year"])

# Interpolate GDP per capita
df[["log_gdp_per_capita"]] = df.groupby("country")[["log_gdp_per_capita"]].transform(
    lambda x: x.interpolate().ffill().bfill()
)

# Forward/backward fill all share columns
share_cols = [
    "coal_share_energy", "gas_share_energy", "oil_share_energy",
    "biofuel_share_energy",
    "hydro_share_energy", "solar_share_energy", "wind_share_energy",
    "nuclear_share_energy",
]
df[share_cols] = df.groupby("country")[share_cols].transform(
    lambda x: x.ffill().bfill()
)

# Final check
missing_summary = df.isna().sum()
print("Missing values after imputation:\n", missing_summary)

# Save cleaned dataset
output_file = "Data/owid-energy-data-clean.csv"
df.to_csv(output_file, index=False)

print(f"\nCleaned dataset saved to: {output_file}")
print(f"Final shape: {df.shape}")
print(f"Number of countries: {df['country'].nunique()}")