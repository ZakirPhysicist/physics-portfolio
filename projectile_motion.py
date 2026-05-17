import numpy as np
import matplotlib.pyplot as plt

# Constants
g = 9.81  # gravity (m/s²)

def projectile_motion(v0, angle_deg):
    angle_rad = np.radians(angle_deg)
    
    # Calculations
    vx = v0 * np.cos(angle_rad)
    vy = v0 * np.sin(angle_rad)
    
    t_flight = 2 * vy / g
    t = np.linspace(0, t_flight, 300)
    
    x = vx * t
    y = vy * t - 0.5 * g * t**2
    
    max_height = (vy**2) / (2 * g)
    range_ = vx * t_flight

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, color='royalblue', linewidth=2)
    plt.title(f'Projectile Motion | v₀={v0} m/s | Angle={angle_deg}°', fontsize=14)
    plt.xlabel('Horizontal Distance (m)')
    plt.ylabel('Vertical Height (m)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.fill_between(x, y, alpha=0.1, color='royalblue')
    
    # Annotations
    plt.annotate(f'Max Height: {max_height:.2f} m',
                 xy=(x[np.argmax(y)], max_height),
                 xytext=(x[np.argmax(y)] + 5, max_height + 2),
                 arrowprops=dict(arrowstyle='->'))
    
    plt.tight_layout()
    plt.savefig('projectile_motion.png')
    plt.show()
    
    print(f"\n📊 Results:")
    print(f"   Max Height   : {max_height:.2f} m")
    print(f"   Total Range  : {range_:.2f} m")
    print(f"   Flight Time  : {t_flight:.2f} s")

# Run it
v0 = float(input("Enter initial velocity (m/s): "))
angle = float(input("Enter launch angle (degrees): "))
projectile_motion(v0, angle)