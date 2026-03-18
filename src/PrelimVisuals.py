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

# -------------------------------
# 1. Load cleaned dataset
# -------------------------------
input_file = "Data/owid-energy-data-clean.csv"
df = pd.read_csv(input_file)

# Ensure the figures folder exists
figures_dir = "figures"
os.makedirs(figures_dir, exist_ok=True)

# -------------------------------
# 2. Define colors per country
# -------------------------------
selected_countries = ['United States', 'China', 'India', 'Germany', 'Brazil']
country_colors = {
    'United States': '#1f77b4',  # blue
    'China': '#ff7f0e',         # orange
    'India': '#2ca02c',         # green
    'Germany': '#d62728',       # red
    'Brazil': '#9467bd'         # purple
}

# -------------------------------
# 3. Time-series of energy_per_capita
# -------------------------------
plt.figure(figsize=(10, 6))
for country in selected_countries:
    country_data = df[df['country'] == country]
    plt.plot(
        country_data['year'], 
        country_data['energy_per_capita'], 
        label=country, 
        color=country_colors[country]  # consistent color
    )

plt.xlabel("Year")
plt.ylabel("Energy per Capita (MWh/person)")
plt.title("Energy per Capita Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{figures_dir}/energy_per_capita_timeseries.png")
plt.close()
print("Saved energy_per_capita_timeseries.png")

# -------------------------------
# 4. Renewables vs Fossil Share over time
# -------------------------------
plt.figure(figsize=(10, 6))
for country in selected_countries:
    country_data = df[df['country'] == country]
    plt.plot(
        country_data['year'], 
        country_data['renewables_share_energy'], 
        label=f"{country} Renewables", 
        color=country_colors[country],  # same country color
        linestyle='--'
    )
    plt.plot(
        country_data['year'], 
        country_data['fossil_share_energy'], 
        label=f"{country} Fossil", 
        color=country_colors[country],  # same country color
        linestyle='-'
    )

plt.xlabel("Year")
plt.ylabel("Share of Energy")
plt.title("Renewables vs Fossil Share of Energy Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{figures_dir}/renewables_vs_fossil_share.png")
plt.close()
print("Saved renewables_vs_fossil_share.png")

# -------------------------------
# 5. GDP vs Energy per Capita Scatter
# -------------------------------
plt.figure(figsize=(8, 6))
for country in selected_countries:
    country_data = df[df['country'] == country]
    plt.scatter(
        country_data['gdp_per_capita'], 
        country_data['energy_per_capita'], 
        alpha=0.6, 
        label=country, 
        color=country_colors[country]  # consistent color
    )

plt.xlabel("GDP per Capita (USD)")
plt.ylabel("Energy per Capita (MWh/person)")
plt.title("GDP vs Energy per Capita")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{figures_dir}/gdp_vs_energy_per_capita.png")
plt.close()
print("Saved gdp_vs_energy_per_capita.png")

# Fixed axis limits to match the static chart
x_min, x_max = 0, 100000      # GDP per Capita
y_min, y_max = 0, 100000     # Energy per Capita

# -------------------------------
# 6. GDP vs Energy per Capita Animation
# -------------------------------
years = sorted(df['year'].unique())
fig, ax = plt.subplots(figsize=(8, 6))

# Scatter objects per country
scatters = {}
lines = {}      # new: line for history
histories = {}  # new: store past positions

for country in selected_countries:
    # Initialize first valid data point
    first_data = df[(df['country'] == country) & (~df['gdp_per_capita'].isna()) & (~df['energy_per_capita'].isna())]
    if not first_data.empty:
        x0 = first_data['gdp_per_capita'].values[0]
        y0 = first_data['energy_per_capita'].values[0]
        scatters[country] = ax.scatter([x0], [y0], label=country, color=country_colors[country], s=100, alpha=0.7)
        histories[country] = [[x0, y0]]
    else:
        scatters[country] = ax.scatter([], [], label=country, color=country_colors[country], s=100, alpha=0.7)
        histories[country] = []

    # Initialize line for trailing history
    lines[country], = ax.plot([], [], color=country_colors[country], linestyle='--', alpha=0.5)

# Set fixed axis limits to match static chart
ax.set_xlim(x_min, x_max)       # replace with your static chart axes
ax.set_ylim(y_min, y_max)      # replace with your static chart axes

ax.set_xlabel("GDP per Capita (USD)")
ax.set_ylabel("Energy per Capita (MWh/person)")
ax.grid(True)
ax.legend()

def update(year):
    ax.set_title(f"GDP vs Energy per Capita ({year})")
    for country in selected_countries:
        country_data = df[(df['country'] == country) & (df['year'] == year)]
        if not country_data.empty:
            x = country_data['gdp_per_capita'].values[0]
            y = country_data['energy_per_capita'].values[0]
            if not np.isnan(x) and not np.isnan(y):
                # Update scatter position
                scatters[country].set_offsets(np.array([[x, y]]))
                # Update history
                histories[country].append([x, y])
                line_data = np.array(histories[country])
                # Update line positions
                lines[country].set_data(line_data[:,0], line_data[:,1])
            else:
                scatters[country].set_offsets(np.empty((0, 2)))
        else:
            scatters[country].set_offsets(np.empty((0, 2)))
    # Return both scatters and lines
    return list(scatters.values()) + list(lines.values())

anim = FuncAnimation(fig, update, frames=years, interval=500, blit=False)

# Save GIF using Pillow
anim.save(f"{figures_dir}/gdp_vs_energy_per_capita_animation.gif", writer='pillow', dpi=200)
plt.close()
print("Saved gdp_vs_energy_per_capita_animation.gif using Pillow")

print("\nAll visualizations (static + animation) saved in the 'figures/' folder.")
