import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import lsq_linear

# ----------------------
#  Simulation Settings
# ----------------------
m = 1.0                # Mass (kg)
kp, kd = 10, 0         # PD gains
b = 5.0                # Damping
T_min, T_max = 20.0, 70.0  # Cable tension limits

# Motor anchor positions (triangle for demonstration)
motor_pos = np.array([
    [0.0, 0.0],
    [2.0, 0.0],
    [1.0, 2.0]
])

# Reference path: a circle
r = 0.3
omega = np.pi / 5
center = np.array([1.0, 1.0])
dt = 0.05
t_final = 10.0
time = np.arange(0.0, t_final + dt, dt)

# ----------------------
#  Initial Conditions
# ----------------------
current_pos = center.copy()  # Start at center, while reference is offset
current_vel = np.array([0.0, 0.0])

pos_log = []
ref_log = []
T_log = []

# ----------------------
#  Helper Functions
# ----------------------
def get_Jacobian(pos):
    """
    Returns a 2x3 matrix whose columns are the unit vectors
    from 'pos' to each motor anchor.
    """
    J = np.zeros((2, 3))
    for i in range(3):
        vec = motor_pos[i] - pos
        norm_vec = np.linalg.norm(vec)
        if norm_vec > 1e-12:
            J[:, i] = vec / norm_vec
        else:
            J[:, i] = np.zeros(2)
    return J

def calc_tension_robust(F_des, J):
    """
    Solve min ||J*T - F_des|| subject to T_min <= T[i] <= T_max.
    Returns the 'best feasible' T even if F_des is not perfectly achievable.
    """
    from scipy.optimize import lsq_linear
    res = lsq_linear(J, F_des, bounds=(T_min, T_max), method='trf', lsmr_tol='auto', verbose=0)
    if not res.success:
        # fallback
        T_unbounded, _, _, _ = np.linalg.lstsq(J, F_des, rcond=None)
        return np.clip(T_unbounded, T_min, T_max)
    return res.x

# ----------------------
#  Main Simulation Loop
# ----------------------
for t in time:
    pos_ref = center + r * np.array([np.cos(omega * t), np.sin(omega * t)])
    vel_ref = r * omega * np.array([-np.sin(omega * t), np.cos(omega * t)])
    ref_log.append(pos_ref.copy())
    
    # PD control
    pos_err = pos_ref - current_pos
    vel_err = vel_ref - current_vel
    F_des = m*(kp*pos_err + kd*vel_err) - b*current_vel
    
    J = get_Jacobian(current_pos)
    T = calc_tension_robust(F_des, J)
    F_real = J @ T
    acc = F_real / m
    
    # Integrate
    current_vel += acc * dt
    current_pos += current_vel * dt
    
    pos_log.append(current_pos.copy())
    T_log.append(T.copy())

pos_log = np.array(pos_log)
ref_log = np.array(ref_log)
T_log = np.array(T_log)

# ----------------------
#  Animation
# ----------------------
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(-0.5, 2.5)
ax.set_ylim(-0.5, 2.5)
ax.set_aspect('equal')
ax.grid(True)

ax.plot(motor_pos[:, 0], motor_pos[:, 1], 'ro', markersize=10, label='Motors')
circle = plt.Circle(center, r, fill=False, linestyle='--', label='Desired Path')
ax.add_artist(circle)

cable_line1, = ax.plot([], [], 'bo-', lw=2, markersize=8)
cable_line2, = ax.plot([], [], 'bo-', lw=2, markersize=8)
cable_line3, = ax.plot([], [], 'bo-', lw=2, markersize=8)
trajectory, = ax.plot([], [], 'b--', linewidth=1, alpha=0.6, label='Real Path')
ref_point, = ax.plot([], [], 'bo', markersize=8, label='Reference Position')

def animate(i):
    if i < len(pos_log):
        obj_pos = pos_log[i]
        
        cable_line1.set_data([motor_pos[0,0], obj_pos[0]],
                             [motor_pos[0,1], obj_pos[1]])
        cable_line2.set_data([motor_pos[1,0], obj_pos[0]],
                             [motor_pos[1,1], obj_pos[1]])
        cable_line3.set_data([motor_pos[2,0], obj_pos[0]],
                             [motor_pos[2,1], obj_pos[1]])
        
        trajectory.set_data(pos_log[:i+1, 0], pos_log[:i+1, 1])
        ref_point.set_data([ref_log[i, 0]], [ref_log[i, 1]])
    return cable_line1, cable_line2, cable_line3, trajectory, ref_point

ani = FuncAnimation(fig, animate, frames=len(pos_log), interval=50, blit=False, repeat=False)
ax.legend()
plt.show()

# ----------------------
#  Tension Plot
# ----------------------
plt.figure()
plt.plot(time, T_log[:, 0], label='Motor 1 Tension')
plt.plot(time, T_log[:, 1], label='Motor 2 Tension')
plt.plot(time, T_log[:, 2], label='Motor 3 Tension')
plt.xlabel('Time (s)')
plt.ylabel('Tension (N)')
plt.title('Cable Tensions for Each Motor')
plt.legend()
plt.grid(True)
plt.show()
