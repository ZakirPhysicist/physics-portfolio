import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite
from scipy.special import factorial

# Constants
hbar = 1.0
m = 1.0
omega = 1.0

# Position array
x = np.linspace(-6, 6, 1000)

def hermite_polynomial(n, x):
    """Calculate Hermite polynomial"""
    H = hermite(n)
    return H(x)

def psi_n(n, x):
    """Calculate quantum harmonic oscillator wave function"""
    norm = (1/np.sqrt(2**n * factorial(n))) * (m * omega / (np.pi * hbar))**0.25
    gaussian = np.exp(-m * omega * x**2 / (2 * hbar))
    H_n = hermite_polynomial(n, np.sqrt(m * omega / hbar) * x)
    return norm * gaussian * H_n

def energy(n):
    """Calculate energy level"""
    return hbar * omega * (n + 0.5)

def classical_probability(n, x):
    """Classical probability distribution"""
    E = energy(n)
    x_max = np.sqrt(2 * E / (m * omega**2))
    prob = np.zeros_like(x)
    mask = np.abs(x) < x_max
    prob[mask] = 1 / (np.pi * np.sqrt(x_max**2 - x[mask]**2))
    return prob

# Colors for different levels
colors = ['royalblue', 'tomato', 'mediumseagreen', 
          'darkorchid', 'darkorange']
levels = [0, 1, 2, 3, 4]

# Create figure with 6 clean subplots
fig, axes = plt.subplots(3, 2, figsize=(12, 13))
fig.suptitle('Quantum Harmonic Oscillator\nby ZakirPhysicist',
             fontsize=14, fontweight='bold')

# Plot 1 — Wave functions for n=0,1,2
ax1 = axes[0, 0]
for n, color in zip(levels[:3], colors[:3]):
    psi = psi_n(n, x)
    E_n = energy(n)
    ax1.plot(x, psi + E_n, color=color, 
             linewidth=2, label=f'n={n}')
    ax1.axhline(y=E_n, color=color, 
                linestyle='--', alpha=0.3)

# Draw potential well
V = 0.5 * m * omega**2 * x**2
ax1.plot(x, V, 'k-', linewidth=2, label='V(x)')
ax1.set_title('Wave Functions in Potential Well', fontsize=10)
ax1.set_xlabel('Position x', fontsize=9)
ax1.set_ylabel('Energy / ψ(x)', fontsize=9)
ax1.legend(fontsize=8)
ax1.set_ylim(-0.5, 5.5)
ax1.set_xlim(-5, 5)
ax1.grid(True, alpha=0.3)

# Plot 2 — Energy Levels
ax2 = axes[0, 1]
for n, color in zip(levels, colors):
    E_n = energy(n)
    ax2.hlines(E_n, 0.2, 0.8, colors=color, linewidth=3)
    ax2.text(0.85, E_n, f'n={n}\nE={E_n:.1f}ℏω',
             va='center', fontsize=8, color=color)

ax2.set_title('Energy Level Ladder', fontsize=10)
ax2.set_ylabel('Energy (ℏω)', fontsize=9)
ax2.set_xlim(0, 1.3)
ax2.set_xticks([])
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3 — Probability density n=0
ax3 = axes[1, 0]
psi0 = psi_n(0, x)
prob0 = psi0**2
classical0 = classical_probability(0, x)
ax3.plot(x, prob0, color='royalblue', 
         linewidth=2, label='Quantum |ψ₀|²')
ax3.plot(x, classical0, color='tomato', linewidth=2,
         linestyle='--', label='Classical')
ax3.fill_between(x, prob0, alpha=0.2, color='royalblue')
ax3.set_title('Ground State n=0: Quantum vs Classical', fontsize=10)
ax3.set_xlabel('Position x', fontsize=9)
ax3.set_ylabel('Probability Density', fontsize=9)
ax3.legend(fontsize=8)
ax3.set_xlim(-4, 4)
ax3.grid(True, alpha=0.3)

# Plot 4 — Probability density n=4
ax4 = axes[1, 1]
psi4 = psi_n(4, x)
prob4 = psi4**2
classical4 = classical_probability(4, x)
ax4.plot(x, prob4, color='darkorchid',
         linewidth=2, label='Quantum |ψ₄|²')
ax4.plot(x, classical4, color='darkorange', linewidth=2,
         linestyle='--', label='Classical')
ax4.fill_between(x, prob4, alpha=0.2, color='darkorchid')
ax4.set_title('Excited State n=4: Quantum vs Classical', fontsize=10)
ax4.set_xlabel('Position x', fontsize=9)
ax4.set_ylabel('Probability Density', fontsize=9)
ax4.legend(fontsize=8)
ax4.set_xlim(-4, 4)
ax4.grid(True, alpha=0.3)

# Plot 5 — All probability densities
ax5 = axes[2, 0]
for n, color in zip(levels, colors):
    prob = psi_n(n, x)**2
    ax5.plot(x, prob, color=color, linewidth=2, label=f'n={n}')
ax5.set_title('Probability Densities All Levels', fontsize=10)
ax5.set_xlabel('Position x', fontsize=9)
ax5.set_ylabel('|ψ(x)|²', fontsize=9)
ax5.legend(fontsize=8)
ax5.set_xlim(-5, 5)
ax5.grid(True, alpha=0.3)

# Plot 6 — Uncertainty principle
ax6 = axes[2, 1]
delta_x_list = []
delta_p_list = []
ns = range(0, 8)

for n in ns:
    psi = psi_n(n, x)
    prob = psi**2
    x_mean = np.trapezoid(x * prob, x)
    x2_mean = np.trapezoid(x**2 * prob, x)
    dx = np.sqrt(abs(x2_mean - x_mean**2))
    dp = np.sqrt(energy(n) * m)
    delta_x_list.append(dx)
    delta_p_list.append(dp)

uncertainty = [dx * dp for dx, dp in 
               zip(delta_x_list, delta_p_list)]
ax6.plot(list(ns), uncertainty, 'o-', 
         color='royalblue', linewidth=2, markersize=8)
ax6.axhline(y=0.5, color='red', linestyle='--',
            label='Minimum ℏ/2')
ax6.set_title('Uncertainty Principle Δx·Δp', fontsize=10)
ax6.set_xlabel('Quantum Number n', fontsize=9)
ax6.set_ylabel('Δx·Δp (ℏ)', fontsize=9)
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('quantum_harmonic_oscillator.png', 
            dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Quantum Harmonic Oscillator Analysis Complete!")
print("📊 Visualizations generated:")
print("   • Wave functions in potential well")
print("   • Energy level ladder")
print("   • Ground state quantum vs classical")
print("   • Excited state quantum vs classical")
print("   • All probability densities")
print("   • Uncertainty principle verification")
print("\n📁 Graph saved as quantum_harmonic_oscillator.png")