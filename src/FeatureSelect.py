"""
Feature selection and basic preprocessing

Goal:
- Select features relevant for predicting energy_per_capita.
- Drop rows with missing target.
- Filter years with high missingness.
- Drop countries with too many missing values.
- Drop aggregate rows like 'World' and 'North America'.
- Add derived features like gdp_per_capita.
"""

import pandas as pd

# -------------------------------
# 1. Load dataset
# -------------------------------
df = pd.read_csv("Data/owid-energy-data-updated.csv")

# -------------------------------
# 2. Feature selection
# -------------------------------
selected_columns = [
    # identifiers
    "country",
    "year",

    # economic
    "population",
    "gdp",

    # target
    "energy_per_capita",

    # fossil fuels
    "coal_cons_per_capita",
    "gas_energy_per_capita",
    "oil_energy_per_capita",
    "fossil_energy_per_capita",
    "fossil_share_energy",

    # renewables / low carbon
    "renewables_energy_per_capita",
    "low_carbon_energy_per_capita",
    "renewables_share_energy",
    "low_carbon_share_energy",
    "hydro_share_energy",
    "solar_share_energy",
    "wind_share_energy"
]

df_selected = df[selected_columns].copy()

# -------------------------------
# 3. Drop rows with missing target
# -------------------------------
df_selected = df_selected.dropna(subset=["energy_per_capita"])

# -------------------------------
# 4. Add derived features
# -------------------------------
df_selected["gdp_per_capita"] = df_selected["gdp"] / df_selected["population"]

# -------------------------------
# 5. Filter by year (data coverage improves from 1965)
# -------------------------------
df_selected = df_selected[df_selected["year"] >= 1965]

# -------------------------------
# 6. Drop countries with too many missing values
# -------------------------------
threshold = 0.4  # 40% missing allowed
country_missing = df_selected.groupby("country").apply(
    lambda x: x.isna().mean().mean()
)
good_countries = country_missing[country_missing <= threshold].index
df_selected = df_selected[df_selected["country"].isin(good_countries)]

# -------------------------------
# 7. Drop aggregate regions
# -------------------------------
df_selected = df_selected.drop(
    df_selected[df_selected['country'].isin(['World', 'North America'])].index
)

# -------------------------------
# 8. Optional: reset index
# -------------------------------
df_selected = df_selected.reset_index(drop=True)

# -------------------------------
# 9. Save cleaned dataset
# -------------------------------
output_file = "Data/owid-energy-data-selected.csv"
df_selected.to_csv(output_file, index=False)

print(f"Cleaned dataset saved: {output_file}")
print(f"Shape after filtering: {df_selected.shape}")
print(f"Remaining countries: {df_selected['country'].nunique()}")