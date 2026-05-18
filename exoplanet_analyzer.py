import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import urllib.request

# Download real NASA Exoplanet data
print("Downloading real NASA Exoplanet data...")
url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+pl_name,pl_rade,pl_masse,pl_orbper,pl_eqt,st_dist+from+pscomppars+where+pl_rade+is+not+null+and+pl_masse+is+not+null&format=csv"

try:
    urllib.request.urlretrieve(url, "exoplanets.csv")
    df = pd.read_csv("exoplanets.csv")
    print(f"Successfully loaded {len(df)} exoplanets!")
except:
    # Backup: create sample data if no internet
    print("Using sample data...")
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        'pl_name': [f'Exoplanet_{i}' for i in range(n)],
        'pl_rade': np.random.lognormal(0.5, 0.8, n),
        'pl_masse': np.random.lognormal(1.0, 1.2, n),
        'pl_orbper': np.random.lognormal(3.0, 1.5, n),
        'pl_eqt': np.random.normal(700, 400, n),
        'st_dist': np.random.lognormal(4.0, 1.0, n)
    })

# Clean data
df = df.dropna()
df = df[df['pl_rade'] > 0]
df = df[df['pl_masse'] > 0]
df = df[df['pl_orbper'] > 0]

print(f"\n📊 Dataset Summary:")
print(f"   Total Exoplanets  : {len(df)}")
print(f"   Avg Planet Radius : {df['pl_rade'].mean():.2f} Earth radii")
print(f"   Avg Planet Mass   : {df['pl_masse'].mean():.2f} Earth masses")

# Create figure with 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('NASA Exoplanet Data Analysis\nby ZakirPhysicist',
             fontsize=16, fontweight='bold')

# Plot 1 — Planet Radius Distribution
axes[0,0].hist(df['pl_rade'], bins=40,
               color='royalblue', edgecolor='white', alpha=0.85)
axes[0,0].set_title('Distribution of Planet Sizes')
axes[0,0].set_xlabel('Planet Radius (Earth Radii)')
axes[0,0].set_ylabel('Number of Planets')
axes[0,0].axvline(df['pl_rade'].mean(), color='red',
                   linestyle='--', label=f"Mean: {df['pl_rade'].mean():.2f}")
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Plot 2 — Mass vs Radius
scatter = axes[0,1].scatter(df['pl_rade'], df['pl_masse'],
                             alpha=0.5, c=df['pl_rade'],
                             cmap='plasma', s=20)
axes[0,1].set_title('Planet Mass vs Radius')
axes[0,1].set_xlabel('Planet Radius (Earth Radii)')
axes[0,1].set_ylabel('Planet Mass (Earth Masses)')
axes[0,1].set_xscale('log')
axes[0,1].set_yscale('log')
plt.colorbar(scatter, ax=axes[0,1], label='Radius')
axes[0,1].grid(True, alpha=0.3)

# Plot 3 — Orbital Period Distribution
axes[1,0].hist(np.log10(df['pl_orbper']), bins=40,
               color='mediumseagreen', edgecolor='white', alpha=0.85)
axes[1,0].set_title('Distribution of Orbital Periods')
axes[1,0].set_xlabel('log10(Orbital Period in Days)')
axes[1,0].set_ylabel('Number of Planets')
axes[1,0].grid(True, alpha=0.3)

# Plot 4 — Equilibrium Temperature Distribution
if 'pl_eqt' in df.columns:
    temp_data = df['pl_eqt'].dropna()
    temp_data = temp_data[temp_data > 0]
    axes[1,1].hist(temp_data, bins=40,
                   color='tomato', edgecolor='white', alpha=0.85)
    axes[1,1].set_title('Equilibrium Temperature of Planets')
    axes[1,1].set_xlabel('Temperature (Kelvin)')
    axes[1,1].set_ylabel('Number of Planets')
    axes[1,1].axvline(373, color='blue', linestyle='--', label='100°C (Water boiling)')
    axes[1,1].axvline(273, color='cyan', linestyle='--', label='0°C (Water freezing)')
    axes[1,1].legend(fontsize=8)
    axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('exoplanet_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Analysis complete!")
print("📊 Graph saved as exoplanet_analysis.png")