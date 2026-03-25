"""
Preliminary Data Visualization Script

Goal:
- Generate meaningful plots to explore energy consumption patterns across countries.
- Save plots as PNG files in the 'figures/' folder.

Input:
- Data/owid-energy-data-clean.csv (cleaned dataset)

Output:
- Figures saved to 'figures/' folder
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import numpy as np

df=pd.read_csv("Data/owid-energy-data-clean.csv")

# Compute total renewables share
df['renewables_share_energy'] = (
    df['hydro_share_energy'] +
    df['solar_share_energy'] +
    df['wind_share_energy'] +
    df['biofuel_share_energy']
)

# Compute total fossil share
df['fossil_share_energy'] = (
    df['coal_share_energy'] +
    df['gas_share_energy'] +
    df['oil_share_energy']
)

if ('log_gdp_per_capita' in df.columns):
    df['gdp_per_capita'] = np.exp(df['log_gdp_per_capita'])

if ('log_population' in df.columns):
    df['population'] = np.exp(df['log_population'])

if ('gdp_per_capita' in df.columns) and ('population' in df.columns):
    df['gdp'] = df['gdp_per_capita'] * df['population']

figuresDir="figures"
os.makedirs(figuresDir, exist_ok=True)

selectedCountries=['United States', 'China', 'India', 'Germany', 'Brazil']
countryColors={'United States':'#1f77b4', 'China':'#ff7f0e', 'India':'#2ca02c', 'Germany':'#d62728', 'Brazil':'#9467bd'}

plt.figure(figsize=(10, 6))

for country in selectedCountries:
    countryData=df[df['country']==country]
    plt.plot(countryData['year'], countryData['energy_per_capita'], label=country, color=countryColors[country])

plt.xlabel("Year")
plt.ylabel("Energy per Capita (kWh/person)")
plt.title("Energy per Capita Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{figuresDir}/energy_per_capita_timeseries.png")
plt.close()
print("Saved energy_per_capita_timeseries.png")

plt.figure(figsize=(10, 6))

for country in selectedCountries:
    countryData=df[df['country']==country]
    plt.plot(countryData['year'], countryData['renewables_share_energy'], label=f"{country} Renewables", color=countryColors[country], linestyle='--')
    plt.plot(countryData['year'], countryData['fossil_share_energy'], label=f"{country} Fossil", color=countryColors[country], linestyle='-')

plt.xlabel("Year")
plt.ylabel("Share of Energy")
plt.title("Renewables vs Fossil Share of Energy Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{figuresDir}/renewables_vs_fossil_share.png")
plt.close()
print("Saved renewables_vs_fossil_share.png")

plt.figure(figsize=(8, 6))

for country in selectedCountries:
    countryData=df[df['country']==country]
    plt.scatter(countryData['gdp_per_capita'], countryData['energy_per_capita'], alpha=0.6, label=country, color=countryColors[country])

plt.xlabel("GDP per Capita (USD)")
plt.ylabel("Energy per Capita (kWh/person)")
plt.title("GDP vs Energy per Capita")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{figuresDir}/gdp_vs_energy_per_capita.png")
plt.close()
print("Saved gdp_vs_energy_per_capita.png")

plt.figure(figsize=(10, 6))

scatter=None

for country in selectedCountries:
    countryData=df[df['country']==country]
    
    if ('gdp' in df.columns) and ('fossil_share_energy' in df.columns):
        scatter=plt.scatter(countryData['gdp_per_capita'], countryData['energy_per_capita'], s=countryData['gdp']/100000000000, c=countryData['fossil_share_energy'], cmap='Reds', alpha=0.7, edgecolors='black', label=country)

plt.xlabel("GDP per Capita (USD)")
plt.ylabel("Energy per Capita (kWh/person)")
plt.title("Energy vs GDP (Size: Absolute GDP, Color: Fossil Share)")

if (scatter is not None):
    cbar=plt.colorbar(scatter)
    cbar.set_label("Fossil Fuel Share (%)")

plt.grid(True)
plt.tight_layout()
plt.savefig(f"{figuresDir}/multi_feature_bubble_chart.png")
plt.close()
print("Saved multi_feature_bubble_chart.png")

xMin, xMax = 0, 100000
yMin, yMax = 0, 100000

years = sorted(df['year'].unique())
fig, ax = plt.subplots(figsize=(8, 6))

scatters = {}
lines = {}
histories = {}

for country in selectedCountries:  
    firstData = df[(df['country'] == country) & (~df['gdp_per_capita'].isna()) & (~df['energy_per_capita'].isna())]
    
    if not firstData.empty:
        x0 = firstData['gdp_per_capita'].values[0]
        y0 = firstData['energy_per_capita'].values[0]
        scatters[country] = ax.scatter([x0], [y0], label=country, color=countryColors[country], s=100, alpha=0.7)
        histories[country] = [[x0, y0]]
    else:
        scatters[country] = ax.scatter([], [], label=country, color=countryColors[country], s=100, alpha=0.7)
        histories[country] = []

    lines[country], = ax.plot([], [], color=countryColors[country], linestyle='--', alpha=0.5)

ax.set_xlim(xMin, xMax)
ax.set_ylim(yMin, yMax)

ax.set_xlabel("GDP per Capita (USD)")
ax.set_ylabel("Energy per Capita (kWh/person)")
ax.grid(True)
ax.legend()

def update(year):
    ax.set_title(f"GDP vs Energy per Capita ({year})")
    
    for country in selectedCountries:
        countryData = df[(df['country'] == country) & (df['year'] == year)]
        
        if not countryData.empty:
            x = countryData['gdp_per_capita'].values[0]
            y = countryData['energy_per_capita'].values[0]
            
            if not np.isnan(x) and not np.isnan(y):
                scatters[country].set_offsets([[x, y]])
                histories[country].append([x, y])
                
                line_data = np.array(histories[country])
                lines[country].set_data(line_data[:, 0], line_data[:, 1])
            else:
                scatters[country].set_offsets(np.empty((0, 2)))
        else:
            scatters[country].set_offsets(np.empty((0, 2)))

    return list(scatters.values()) + list(lines.values())

anim = FuncAnimation(fig, update, frames=years, interval=500, blit=False)

anim.save(f"{figuresDir}/gdp_vs_energy_per_capita_animation.gif", writer='pillow', dpi=200)

plt.close()
print("Saved gdp_vs_energy_per_capita_animation.gif")