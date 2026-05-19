import numpy as np
import matplotlib.pyplot as plt

# Constants
hbar = 1.0545718e-34
m = 9.10938e-31
L = 1e-9

x = np.linspace(0, L, 1000)

def wave_function(n, x, L):
    return np.sqrt(2/L) * np.sin(n * np.pi * x / L)

def probability_density(n, x, L):
    psi = wave_function(n, x, L)
    return psi**2

def energy_level(n, L, m, hbar):
    E = (n**2 * np.pi**2 * hbar**2) / (2 * m * L**2)
    return E / 1.60218e-19

colors = ['royalblue', 'tomato', 'mediumseagreen', 'darkorchid']
quantum_numbers = [1, 2, 3, 4]

# Create 6 separate clean subplots
fig, axes = plt.subplots(3, 2, figsize=(12, 12))
fig.suptitle('Quantum Wave Function Visualizer\nParticle in a Box | by ZakirPhysicist',
             fontsize=14, fontweight='bold')

# Plot 1 — Wave function n=1
ax1 = axes[0, 0]
psi = wave_function(1, x, L)
E = energy_level(1, L, m, hbar)
ax1.plot(x * 1e9, psi, color='royalblue', linewidth=2)
ax1.fill_between(x * 1e9, psi, alpha=0.2, color='royalblue')
ax1.set_title(f'Wave Function n=1 | E={E:.2f} eV', fontsize=10)
ax1.set_xlabel('Position (nm)', fontsize=9)
ax1.set_ylabel('ψ(x)', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='black', linewidth=0.8)

# Plot 2 — Wave function n=2
ax2 = axes[0, 1]
psi = wave_function(2, x, L)
E = energy_level(2, L, m, hbar)
ax2.plot(x * 1e9, psi, color='tomato', linewidth=2)
ax2.fill_between(x * 1e9, psi, alpha=0.2, color='tomato')
ax2.set_title(f'Wave Function n=2 | E={E:.2f} eV', fontsize=10)
ax2.set_xlabel('Position (nm)', fontsize=9)
ax2.set_ylabel('ψ(x)', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='black', linewidth=0.8)

# Plot 3 — Probability Density
ax3 = axes[1, 0]
for n, color in zip(quantum_numbers, colors):
    prob = probability_density(n, x, L)
    ax3.plot(x * 1e9, prob, color=color, linewidth=2, label=f'n={n}')
ax3.set_title('Probability Density |ψ(x)|²', fontsize=10)
ax3.set_xlabel('Position (nm)', fontsize=9)
ax3.set_ylabel('|ψ(x)|²', fontsize=9)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Plot 4 — Energy Levels
ax4 = axes[1, 1]
energies = [energy_level(n, L, m, hbar) for n in quantum_numbers]
for n, E, color in zip(quantum_numbers, energies, colors):
    ax4.hlines(E, 0.2, 0.8, colors=color, linewidth=3)
    ax4.text(0.82, E, f'n={n} | {E:.1f} eV', va='center', fontsize=8, color=color)
ax4.set_title('Energy Levels', fontsize=10)
ax4.set_ylabel('Energy (eV)', fontsize=9)
ax4.set_xlim(0, 1.3)
ax4.set_xticks([])
ax4.grid(True, alpha=0.3, axis='y')

# Plot 5 — Superposition
ax5 = axes[2, 0]
psi1 = wave_function(1, x, L)
psi2 = wave_function(2, x, L)
psi_super = (psi1 + psi2) / np.sqrt(2)
prob_super = psi_super**2
ax5.plot(x * 1e9, psi_super, color='darkorange', linewidth=2, label='Superposition ψ')
ax5.plot(x * 1e9, prob_super, color='purple', linewidth=2, linestyle='--', label='|ψ|²')
ax5.fill_between(x * 1e9, prob_super, alpha=0.15, color='purple')
ax5.set_title('Quantum Superposition n=1 & n=2', fontsize=10)
ax5.set_xlabel('Position (nm)', fontsize=9)
ax5.set_ylabel('Amplitude', fontsize=9)
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)
ax5.axhline(y=0, color='black', linewidth=0.8)

# Plot 6 — Heisenberg Uncertainty
ax6 = axes[2, 1]
uncertainty = []
for n in range(1, 10):
    psi = wave_function(n, x, L)
    prob = psi**2
    x_mean = np.trapezoid(x * prob, x)
    x2_mean = np.trapezoid(x**2 * prob, x)
    dx = np.sqrt(x2_mean - x_mean**2)
    dp = n * np.pi * hbar / L
    uncertainty.append(dx * dp / hbar)

ax6.scatter(range(1, 10), uncertainty, color='royalblue', s=80, zorder=5)
ax6.axhline(y=0.5, color='red', linestyle='--', label='Heisenberg Limit')
ax6.set_title('Heisenberg Uncertainty Δx·Δp/ℏ', fontsize=10)
ax6.set_xlabel('Quantum Number n', fontsize=9)
ax6.set_ylabel('Δx·Δp / ℏ', fontsize=9)
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('quantum_wavefunction.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Quantum Wave Function Analysis Complete!")
print("📊 Graph saved as quantum_wavefunction.png")