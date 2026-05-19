import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import h, c, k

# Constants
h = 6.626e-34    # Planck constant
c = 3e8          # Speed of light
k = 1.381e-23    # Boltzmann constant

# Wavelength array (100nm to 3000nm)
wavelength = np.linspace(1e-9, 3e-6, 10000)

def planck_law(wavelength, T):
    """Planck's blackbody radiation law"""
    exponent = (h * c) / (wavelength * k * T)
    B = (2 * h * c**2) / (wavelength**5) * \
        (1 / (np.exp(exponent) - 1))
    return B

def rayleigh_jeans(wavelength, T):
    """Classical Rayleigh-Jeans law"""
    return (2 * c * k * T) / (wavelength**4)

def wien_law(wavelength, T):
    """Wien approximation"""
    exponent = (h * c) / (wavelength * k * T)
    return (2 * h * c**2) / (wavelength**5) * \
            np.exp(-exponent)

def peak_wavelength(T):
    """Wien's displacement law"""
    b = 2.898e-3  # Wien's displacement constant
    return b / T

# Temperatures to analyze
temperatures = [3000, 4000, 5000, 6000, 7000]
colors = ['red', 'orangered', 'orange', 'yellow', 'lightyellow']
star_temps = {
    'Red Dwarf': 3000,
    'Orange Star': 4000,
    'Yellow Star\n(like Sun)': 5778,
    'White Star': 9000,
    'Blue Star': 15000
}

# Create figure
fig, axes = plt.subplots(3, 2, figsize=(12, 13))
fig.suptitle('Black Body Radiation Analyzer\nby ZakirPhysicist',
             fontsize=14, fontweight='bold')

# Plot 1 — Planck curves for different temperatures
ax1 = axes[0, 0]
colors_temp = ['red', 'orangered', 'orange', 'gold', 'royalblue']
for T, color in zip(temperatures, colors_temp):
    B = planck_law(wavelength, T)
    ax1.plot(wavelength * 1e9, B / 1e13,
             color=color, linewidth=2, label=f'T={T}K')

ax1.set_title("Planck's Radiation Curves", fontsize=10)
ax1.set_xlabel('Wavelength (nm)', fontsize=9)
ax1.set_ylabel('Spectral Radiance (×10¹³ W/m³)', fontsize=9)
ax1.legend(fontsize=8)
ax1.set_xlim(0, 3000)
ax1.grid(True, alpha=0.3)

# Add visible light region
ax1.axvspan(380, 700, alpha=0.1, color='cyan', label='Visible Light')

# Plot 2 — Wien's displacement law
ax2 = axes[0, 1]
temp_range = np.linspace(1000, 20000, 1000)
peak_wavelengths = [peak_wavelength(T) * 1e9 for T in temp_range]

ax2.plot(temp_range, peak_wavelengths,
         color='royalblue', linewidth=2)
ax2.fill_between(temp_range, peak_wavelengths,
                 alpha=0.2, color='royalblue')
ax2.axvspan(380, 700, alpha=0.1, color='cyan')
ax2.set_title("Wien's Displacement Law", fontsize=10)
ax2.set_xlabel('Temperature (K)', fontsize=9)
ax2.set_ylabel('Peak Wavelength (nm)', fontsize=9)
ax2.grid(True, alpha=0.3)

# Mark Sun's temperature
ax2.axvline(x=5778, color='orange', linestyle='--', linewidth=2)
ax2.text(5778, 1500, "Sun\n5778K",
         color='orange', fontsize=8, ha='center')

# Plot 3 — UV Catastrophe
ax3 = axes[1, 0]
T = 5000
B_planck = planck_law(wavelength, T)
B_rj = rayleigh_jeans(wavelength, T)
B_wien = wien_law(wavelength, T)

# Clip Rayleigh-Jeans for display
B_rj_clipped = np.clip(B_rj, 0, 2e14)

ax3.plot(wavelength * 1e9, B_planck / 1e13,
         color='royalblue', linewidth=2, label="Planck's Law")
ax3.plot(wavelength * 1e9, B_rj_clipped / 1e13,
         color='tomato', linewidth=2,
         linestyle='--', label='Rayleigh-Jeans (Classical)')
ax3.plot(wavelength * 1e9, B_wien / 1e13,
         color='green', linewidth=2,
         linestyle='-.', label="Wien's Approximation")
ax3.set_title(f'UV Catastrophe at T={T}K', fontsize=10)
ax3.set_xlabel('Wavelength (nm)', fontsize=9)
ax3.set_ylabel('Spectral Radiance (×10¹³ W/m³)', fontsize=9)
ax3.legend(fontsize=8)
ax3.set_xlim(0, 2000)
ax3.set_ylim(0, 1.5)
ax3.grid(True, alpha=0.3)

# Plot 4 — Star temperatures
ax4 = axes[1, 1]
star_colors = ['red', 'orange', 'yellow', 'white', 'lightblue']
star_names = list(star_temps.keys())
star_T = list(star_temps.values())

for i, (name, T, color) in enumerate(zip(star_names, star_T, star_colors)):
    B = planck_law(wavelength, T)
    B_norm = B / B.max()
    ax4.plot(wavelength * 1e9, B_norm,
             linewidth=2, label=f'{name} ({T}K)')

ax4.axvspan(380, 700, alpha=0.1, color='cyan', label='Visible')
ax4.set_title('Normalized Spectra of Different Stars', fontsize=10)
ax4.set_xlabel('Wavelength (nm)', fontsize=9)
ax4.set_ylabel('Normalized Intensity', fontsize=9)
ax4.legend(fontsize=7)
ax4.set_xlim(0, 3000)
ax4.grid(True, alpha=0.3)

# Plot 5 — Total power radiated (Stefan-Boltzmann)
ax5 = axes[2, 0]
sigma = 5.67e-8  # Stefan-Boltzmann constant
temp_range2 = np.linspace(1000, 10000, 1000)
power = sigma * temp_range2**4

ax5.plot(temp_range2, power / 1e6,
         color='tomato', linewidth=2)
ax5.fill_between(temp_range2, power / 1e6,
                 alpha=0.2, color='tomato')
ax5.set_title('Stefan-Boltzmann Law\nTotal Radiated Power', fontsize=10)
ax5.set_xlabel('Temperature (K)', fontsize=9)
ax5.set_ylabel('Power (×10⁶ W/m²)', fontsize=9)
ax5.grid(True, alpha=0.3)

# Mark Sun
sun_power = sigma * 5778**4
ax5.axvline(x=5778, color='orange',
            linestyle='--', linewidth=2)
ax5.text(5778, sun_power / 2e6,
         f'Sun\n{sun_power/1e6:.0f}×10⁶ W/m²',
         color='orange', fontsize=8, ha='left')

# Plot 6 — Peak wavelength for famous objects
ax6 = axes[2, 1]
objects = ['Cosmic\nMicrowave\nBackground', 'Room\nTemperature',
           'Red\nDwarf', 'Sun', 'Blue\nStar']
temps_obj = [2.7, 300, 3000, 5778, 15000]
peaks = [peak_wavelength(T) * 1e9 for T in temps_obj]
bar_colors = ['purple', 'green', 'red', 'orange', 'royalblue']

bars = ax6.bar(objects, peaks, color=bar_colors, alpha=0.8)
ax6.set_title("Peak Wavelengths of Famous Objects", fontsize=10)
ax6.set_ylabel('Peak Wavelength (nm)', fontsize=9)
ax6.set_yscale('log')
ax6.grid(True, alpha=0.3, axis='y')

for bar, peak in zip(bars, peaks):
    ax6.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() * 1.1,
             f'{peak:.1f}nm', ha='center', fontsize=7)

plt.tight_layout()
plt.savefig('blackbody_radiation.png',
            dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Black Body Radiation Analysis Complete!")
print("📊 Visualizations generated:")
print("   • Planck radiation curves")
print("   • Wien's displacement law")
print("   • UV Catastrophe demonstration")
print("   • Stellar spectra comparison")
print("   • Stefan-Boltzmann law")
print("   • Peak wavelengths of famous objects")
print("\n📁 Graph saved as blackbody_radiation.png")