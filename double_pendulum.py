import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
g = 9.81   # gravity m/s²
L1 = 1.0   # length of first rod (m)
L2 = 1.0   # length of second rod (m)
m1 = 1.0   # mass of first bob (kg)
m2 = 1.0   # mass of second bob (kg)

def double_pendulum(t, y):
    """Equations of motion for double pendulum"""
    theta1, omega1, theta2, omega2 = y

    delta = theta2 - theta1
    sin_delta = np.sin(delta)
    cos_delta = np.cos(delta)

    denom1 = (m1 + m2) * L1 - m2 * L1 * cos_delta**2
    denom2 = (L2 / L1) * denom1

    # Angular accelerations
    alpha1 = (m2 * L1 * omega1**2 * sin_delta * cos_delta +
              m2 * g * np.sin(theta2) * cos_delta +
              m2 * L2 * omega2**2 * sin_delta -
              (m1 + m2) * g * np.sin(theta1)) / denom1

    alpha2 = (-m2 * L2 * omega2**2 * sin_delta * cos_delta +
               (m1 + m2) * g * np.sin(theta1) * cos_delta -
               (m1 + m2) * L1 * omega1**2 * sin_delta -
               (m1 + m2) * g * np.sin(theta2)) / denom2

    return [omega1, alpha1, omega2, alpha2]

def get_positions(sol, L1, L2):
    """Convert angles to x,y positions"""
    theta1 = sol.y[0]
    theta2 = sol.y[2]

    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)

    return x1, y1, x2, y2

# Time array
t_span = (0, 20)
t_eval = np.linspace(0, 20, 5000)

# Initial conditions — slightly different angles
theta1_0 = np.radians(120)
theta2_0 = np.radians(120)
omega1_0 = 0.0
omega2_0 = 0.0

# Pendulum 1 — original
y0_1 = [theta1_0, omega1_0, theta2_0, omega2_0]
sol1 = solve_ivp(double_pendulum, t_span, y0_1,
                 t_eval=t_eval, method='RK45',
                 rtol=1e-8, atol=1e-8)

# Pendulum 2 — tiny difference (0.001 degrees)
y0_2 = [theta1_0 + np.radians(0.001), omega1_0,
        theta2_0, omega2_0]
sol2 = solve_ivp(double_pendulum, t_span, y0_2,
                 t_eval=t_eval, method='RK45',
                 rtol=1e-8, atol=1e-8)

# Get positions
x1_1, y1_1, x2_1, y2_1 = get_positions(sol1, L1, L2)
x1_2, y1_2, x2_2, y2_2 = get_positions(sol2, L1, L2)

# Compute total energy
def compute_energy(sol):
    theta1 = sol.y[0]
    omega1 = sol.y[1]
    theta2 = sol.y[2]
    omega2 = sol.y[3]

    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)

    # Kinetic energy
    v1_sq = (L1 * omega1)**2
    v2_sq = (L1 * omega1)**2 + (L2 * omega2)**2 + \
             2 * L1 * L2 * omega1 * omega2 * np.cos(theta1 - theta2)
    KE = 0.5 * m1 * v1_sq + 0.5 * m2 * v2_sq

    # Potential energy
    PE = m1 * g * y1 + m2 * g * y2

    return KE + PE

energy1 = compute_energy(sol1)
energy2 = compute_energy(sol2)

# Create figure with 6 subplots
fig, axes = plt.subplots(3, 2, figsize=(12, 14))
fig.suptitle('Double Pendulum Chaos Simulator\nby ZakirPhysicist',
             fontsize=14, fontweight='bold')

# Plot 1 — Trajectory of pendulum 1
ax1 = axes[0, 0]
ax1.plot(x2_1, y2_1, color='royalblue',
         linewidth=0.5, alpha=0.8)
ax1.plot(x2_1[0], y2_1[0], 'go', markersize=8,
         label='Start')
ax1.plot(x2_1[-1], y2_1[-1], 'ro', markersize=8,
         label='End')
ax1.set_title('Trajectory — Pendulum 1', fontsize=10)
ax1.set_xlabel('x (m)', fontsize=9)
ax1.set_ylabel('y (m)', fontsize=9)
ax1.legend(fontsize=8)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)

# Plot 2 — Trajectory of pendulum 2
ax2 = axes[0, 1]
ax2.plot(x2_2, y2_2, color='tomato',
         linewidth=0.5, alpha=0.8)
ax2.plot(x2_2[0], y2_2[0], 'go', markersize=8,
         label='Start')
ax2.plot(x2_2[-1], y2_2[-1], 'ro', markersize=8,
         label='End')
ax2.set_title('Trajectory — Pendulum 2\n(0.001° difference)',
              fontsize=10)
ax2.set_xlabel('x (m)', fontsize=9)
ax2.set_ylabel('y (m)', fontsize=9)
ax2.legend(fontsize=8)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)

# Plot 3 — Both trajectories overlaid
ax3 = axes[1, 0]
ax3.plot(x2_1, y2_1, color='royalblue',
         linewidth=0.5, alpha=0.7, label='Pendulum 1')
ax3.plot(x2_2, y2_2, color='tomato',
         linewidth=0.5, alpha=0.7, label='Pendulum 2')
ax3.set_title('Chaos: Both Trajectories Overlaid', fontsize=10)
ax3.set_xlabel('x (m)', fontsize=9)
ax3.set_ylabel('y (m)', fontsize=9)
ax3.legend(fontsize=8)
ax3.set_aspect('equal')
ax3.grid(True, alpha=0.3)

# Plot 4 — Angle vs time
ax4 = axes[1, 1]
ax4.plot(sol1.t, np.degrees(sol1.y[0]),
         color='royalblue', linewidth=1,
         label='θ1 Pendulum 1')
ax4.plot(sol2.t, np.degrees(sol2.y[0]),
         color='tomato', linewidth=1,
         linestyle='--', label='θ1 Pendulum 2')
ax4.set_title('Angle θ1 vs Time — Divergence', fontsize=10)
ax4.set_xlabel('Time (s)', fontsize=9)
ax4.set_ylabel('Angle (degrees)', fontsize=9)
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# Plot 5 — Angular separation (chaos indicator)
ax5 = axes[2, 0]
separation = np.abs(sol1.y[0] - sol2.y[0])
ax5.semilogy(sol1.t, separation + 1e-10,
             color='purple', linewidth=1.5)
ax5.set_title('Chaos Indicator\nAngular Separation Over Time',
              fontsize=10)
ax5.set_xlabel('Time (s)', fontsize=9)
ax5.set_ylabel('|Δθ1| (radians, log scale)', fontsize=9)
ax5.grid(True, alpha=0.3)
ax5.axhline(y=0.1, color='red', linestyle='--',
            label='Divergence threshold')
ax5.legend(fontsize=8)

# Plot 6 — Energy conservation
ax6 = axes[2, 1]
ax6.plot(sol1.t, energy1, color='royalblue',
         linewidth=1.5, label='Pendulum 1')
ax6.plot(sol2.t, energy2, color='tomato',
         linewidth=1.5, linestyle='--',
         label='Pendulum 2')
ax6.set_title('Energy Conservation Check', fontsize=10)
ax6.set_xlabel('Time (s)', fontsize=9)
ax6.set_ylabel('Total Energy (J)', fontsize=9)
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('double_pendulum.png', dpi=150,
            bbox_inches='tight')
plt.show()

print("\n✅ Double Pendulum Simulation Complete!")
print("📊 Visualizations generated:")
print("   • Trajectory of Pendulum 1")
print("   • Trajectory of Pendulum 2")
print("   • Chaos comparison of both trajectories")
print("   • Angle divergence over time")
print("   • Chaos indicator (log scale)")
print("   • Energy conservation verification")
print("\n📁 Graph saved as double_pendulum.png")