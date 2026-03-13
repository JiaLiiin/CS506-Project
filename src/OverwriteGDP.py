import pandas as pd
import requests

# Fetch GDP data from World Bank API
url="http://api.worldbank.org/v2/country/all/indicator/NY.GDP.MKTP.CD?format=json&per_page=20000"
response=requests.get(url)
data=response.json()[1]

wb_data=[]
# Pull only the necessary fields from the JSON
#
# JSON looks like this:
# {
#   "indicator": {
#     "id": "NY.GDP.MKTP.CD",
#     "value": "GDP (current US$)"
#   },
#   "country": {
#     "id": "US",
#     "value": "United States"
#   },
#   "countryiso3code": "USA",
#   "date": "2022",
#   "value": 25462700000000
# }
for entry in data:
    if entry["value"] is not None:
        wb_data.append({
            "country": entry["country"]["value"],
            "year": int(entry["date"]),
            "gdp": entry["value"]
        })

wb_df=pd.DataFrame(wb_data)

# --- Update this path to point to Data folder ---
owid=pd.read_csv("Data/owid-energy-data.csv")

# Remove non-country entries to match World Bank dataset
valid_countries = wb_df["country"].unique()
owid = owid[owid["country"].isin(valid_countries)]

# Drop original GDP column and merge with World Bank GDP on country and year 
owid = owid.drop(columns=["gdp"])
merged = pd.merge(owid, wb_df, on=["country", "year"], how="left")

# --- Update this path to save updated CSV in Data folder ---
merged.to_csv("Data/owid-energy-data-updated.csv", index=False)