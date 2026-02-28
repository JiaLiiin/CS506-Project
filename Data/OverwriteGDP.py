import pandas as pd
import requests

url="http://api.worldbank.org/v2/country/all/indicator/NY.GDP.MKTP.CD?format=json&per_page=20000"
response=requests.get(url)
data=response.json()[1]
wb_data=[]
#pull data off from WorldBank, and only save what we need for this project
#json looks like this
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
owid=pd.read_csv("owid-energy-data.csv")
#cuz the kaggle dataset included some continents and unions, which we are not focusing on for this project, so I removed all non-countries
valid_countries=wb_df["country"].unique()
owid=owid[owid["country"].isin(valid_countries)]
#dropped the original gdp cuz it looks different from the WBData and use the WBData instead, and only merge on country and years and keep the rest the same for the kaggle dataset
owid=owid.drop(columns=["gdp"])
merged=pd.merge(owid, wb_df, on=["country", "year"], how="left")
merged.to_csv("owid-energy-data-updated.csv", index=True)