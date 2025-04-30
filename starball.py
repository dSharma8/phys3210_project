import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

ani = None

# define constants
solar_mass = 1.98892e30 # one solar mass [kg]
solar_radius = 6.957e8 # one solar radius [m]
c = 2.99792458e8 # speed of light [m/s]
G = 6.67408e-11 # gravitational constant [m^3 kg^-1 s^-2]
dt = 10 # time step [s]
elapsed_time = 0 # current time [s]
num_steps = 10000 # number of simulation time steps
particle_count = 0

# define variables
m_bh = 1e6 * solar_mass # mass of black hole [kg]
m_star = 10 * solar_mass # mass of star [kg] (use .15, .30, .40, .50, .70, 1, 3, 10 M. for experiments)
r_bh = 2 * G * m_bh / c**2 # schwarzschild radius of black hole [m]
r_s = 2 * solar_radius # radius of star [m]
r_roche = r_s * (2 * m_bh / m_star)**(1.0/3.0) # roche limit of black hole [m]
r_t = (m_bh / m_star)**(1/3) * r_s # tidal radius of black hole
v_t = math.sqrt(G * (m_bh + m_star) / r_t) # initial velocity condition
n_particles = 1000 # number of particles in star
m_particles = np.ones(n_particles) * m_star / n_particles # mass of particles [kg]
n_particles_consumed = 0 # number of particles consumed by black hole
particles_consumed_over_time = [] # number of particles consumed by black hole over time
time_series = [] # to store corresponding time
epsilon = 1e7 # constant for force softening [m]

# define initial conditions
# init_x = r_bh*5
# init_y = r_bh*5
# init_v_x = -6e7
# init_v_y = 6e7
init_x = r_t * 0
init_y = r_t * -1
init_v_x = v_t * .65
init_v_y = v_t * .6

# generate central star
x_star, y_star = init_x, init_y
v_x_star, v_y_star = init_v_x, init_v_y

# generate random particle cloud
theta = np.linspace(0, 2*np.pi, n_particles)
radii = np.random.uniform(0, r_s, n_particles)
x_cloud = (radii * np.cos(theta)) + init_x
y_cloud = (radii * np.sin(theta)) + init_y
v_x_cloud = np.ones(n_particles) * init_v_x
v_y_cloud = np.ones(n_particles) * init_v_y

# add internal star rotation
# rel_x = x_cloud - init_x
# rel_y = y_cloud - init_y
# rel_r = np.sqrt(rel_x**2 + rel_y**2)
# mask = rel_r != 0
# v_rot = np.zeros(n_particles)
# v_rot[mask] = np.sqrt(G * m_star / rel_r[mask])
# v_tangent_x = np.zeros(n_particles)
# v_tangent_y = np.zeros(n_particles)
# v_tangent_x[mask] = -v_rot[mask] * (rel_y[mask] / rel_r[mask])
# v_tangent_y[mask] = v_rot[mask] * (rel_x[mask] / rel_r[mask])
# v_x_cloud += v_tangent_x
# v_y_cloud += v_tangent_y

# define function for force of grav on star -- newtonian
def f_grav_point(m1, m2, x, y):
    r = math.sqrt(x**2 + y**2)
    if r == 0: return 0, 0
    f = -1 * G * m1 * m2 / r**2
    fx = f * (x / r) 
    fy = f * (y / r)
    return fx, fy
def f_grav_cloud(m1, m2, x, y):
    """
    r = np.sqrt(x**2 + y**2)
    fx = np.zeros(n_particles)
    fy = np.zeros(n_particles)
    mask = r != 0
    f = np.where(mask, -1 * G * m1 * m2 / r**2, 0)
    fx[mask] = f[mask] * (x[mask] / r[mask]) 
    fy[mask] = f[mask] * (y[mask] / r[mask]) 
    return fx, fy
    """
    # affect of black hole on particles
    r_bh = np.sqrt(x**2 + y**2)
    fx_bh = np.zeros(n_particles)
    fy_bh = np.zeros(n_particles)
    mask_bh = r_bh != 0
    f_bh = np.where(mask_bh, -1 * G * m1 * m2 / r_bh**2, 0)
    fx_bh[mask_bh] = f_bh[mask_bh] * (x[mask_bh] / r_bh[mask_bh]) 
    fy_bh[mask_bh] = f_bh[mask_bh] * (y[mask_bh] / r_bh[mask_bh]) 

    # affect of central star mass on particles
    dx_star, dy_star = x - x_star, y - y_star
    r_star = np.sqrt(dx_star**2 + dy_star**2 + epsilon**2)
    fx_star = np.zeros(n_particles)
    fy_star = np.zeros(n_particles)
    # mask_star = r_star != r_s
    mask_star = True
    f_star = np.where(mask_star, -1 * G * m_star * m2 / r_star**2, 0)
    fx_star[mask_star] = f_star[mask_star] * (dx_star[mask_star] / r_star[mask_star]) 
    fy_star[mask_star] = f_star[mask_star] * (dy_star[mask_star] / r_star[mask_star]) 

    # sum total forces
    fx = fx_bh + fx_star
    fy = fy_bh + fy_star
    
    return fx, fy
    # """

# define function for force of grav on star -- paczyński–wiita potential = phi = - G * M / (r - r_bh)
def f_grav_pw_point(m1, m2, x, y):
    r = math.sqrt(x**2 + y**2)
    if r <= r_bh: return 0, 0
    f = -1 * G * m1 * m2 / (r - r_bh)**2
    fx = f * (x / r) 
    fy = f * (y / r)
    return fx, fy
def f_grav_pw_cloud(m1, m2, x, y):
    #"""
    r = np.sqrt(x**2 + y**2)
    fx = np.zeros(n_particles)
    fy = np.zeros(n_particles)
    mask = r > r_bh
    f = np.where(mask, -1 * G * m1 * m2 / (r - r_bh)**2, 0)
    fx[mask] = f[mask] * (x[mask] / r[mask]) 
    fy[mask] = f[mask] * (y[mask] / r[mask]) 
    return fx, fy
    """
    # affect of black hole on particles
    r = np.sqrt(x**2 + y**2)
    fx_bh = np.zeros(n_particles)
    fy_bh = np.zeros(n_particles)
    mask_bh = r > r_bh
    f_bh = np.where(mask_bh, -1 * G * m1 * m2 / (r - r_bh)**2, 0)
    fx_bh[mask_bh] = f_bh[mask_bh] * (x[mask_bh] / r[mask_bh]) 
    fy_bh[mask_bh] = f_bh[mask_bh] * (y[mask_bh] / r[mask_bh]) 

    # affect of central star mass on particles
    dx_star, dy_star = x - x_star, y - y_star
    r_star = np.sqrt(dx_star**2 + dy_star**2 + epsilon**2)
    fx_star = np.zeros(n_particles)
    fy_star = np.zeros(n_particles)
    mask_star = r_star != r_s
    f_star = np.where(mask_star, -1 * G * m_star * m2 / r_star**2, 0)
    fx_star[mask_star] = f_star[mask_star] * (dx_star[mask_star] / r_star[mask_star]) 
    fy_star[mask_star] = f_star[mask_star] * (dy_star[mask_star] / r_star[mask_star]) 

    # sum total forces
    fx = fx_bh + fx_star
    fy = fy_bh + fy_star
    
    return fx, fy
    """

"""
# define function for tidal forces across star
def t_force(x, y, r_s, m_star, m_bh):
    r_minus = np.sqrt((x-r_s/2) ** 2 + ((y - r_s/2) ** 2))
    r_plus = np.sqrt((x+r_s/2) ** 2 + ((y + r_s/2) ** 2))
    ax.plot(-r_bh, 0, 'o', color='black', markersize=10)

    f_x_m, f_y_m = f_grav(m_bh, m_star, x - r_s/2, y)
    f_x_p, f_y_p = f_grav(m_bh, m_star, x + r_s/2, y)

    tidal_f_x = f_x_p - f_x_m
    tidal_f_y = f_y_p - f_y_m
"""

# rk4 method
def rk4_point(x, y, v_x, v_y):
    # calc acceleration from grav in x and y directions
    def accel(x, y):
        f_x, f_y = f_grav_pw_point(m_bh, m_star, x, y)
        a_x = f_x / m_star
        a_y = f_y / m_star
        return a_x, a_y

    # k1
    a_x1, a_y1 = accel(x, y)
    k1_vx = a_x1 * dt
    k1_vy = a_y1 * dt
    k1_x = v_x * dt
    k1_y = v_y * dt

    # k2
    a_x2, a_y2 = accel(x + 0.5 * k1_x, y + 0.5 * k1_y)
    k2_vx = a_x2 * dt
    k2_vy = a_y2 * dt
    k2_x = (v_x + 0.5 * k1_vx) * dt
    k2_y = (v_y + 0.5 * k1_vy) * dt

    # k3
    a_x3, a_y3 = accel(x + 0.5 * k2_x, y + 0.5 * k2_y)
    k3_vx = a_x3 * dt
    k3_vy = a_y3 * dt
    k3_x = (v_x + 0.5 * k2_vx) * dt
    k3_y = (v_y + 0.5 * k2_vy) * dt

    # k4
    a_x4, a_y4 = accel(x + k3_x, y + k3_y)
    k4_vx = a_x4 * dt
    k4_vy = a_y4 * dt
    k4_x = (v_x + k3_vx) * dt
    k4_y = (v_y + k3_vy) * dt

    # combine k1, k2, k3, k4
    x += (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6
    y += (k1_y + 2*k2_y + 2*k3_y + k4_y) / 6
    v_x += (k1_vx + 2*k2_vx + 2*k3_vx + k4_vx) / 6
    v_y += (k1_vy + 2*k2_vy + 2*k3_vy + k4_vy) / 6

    return x, y, v_x, v_y
def rk4_cloud(x, y, v_x, v_y):
    # calc acceleration from grav in x and y directions
    def accel(x, y):
        f_x, f_y = f_grav_pw_cloud(m_bh, m_particles, x, y)
        a_x = f_x / m_particles
        a_y = f_y / m_particles
        return a_x, a_y

    # k1
    a_x1, a_y1 = accel(x, y)
    k1_vx = a_x1 * dt
    k1_vy = a_y1 * dt
    k1_x = v_x * dt
    k1_y = v_y * dt

    # k2
    a_x2, a_y2 = accel(x + 0.5 * k1_x, y + 0.5 * k1_y)
    k2_vx = a_x2 * dt
    k2_vy = a_y2 * dt
    k2_x = (v_x + 0.5 * k1_vx) * dt
    k2_y = (v_y + 0.5 * k1_vy) * dt

    # k3
    a_x3, a_y3 = accel(x + 0.5 * k2_x, y + 0.5 * k2_y)
    k3_vx = a_x3 * dt
    k3_vy = a_y3 * dt
    k3_x = (v_x + 0.5 * k2_vx) * dt
    k3_y = (v_y + 0.5 * k2_vy) * dt

    # k4
    a_x4, a_y4 = accel(x + k3_x, y + k3_y)
    k4_vx = a_x4 * dt
    k4_vy = a_y4 * dt
    k4_x = (v_x + k3_vx) * dt
    k4_y = (v_y + k3_vy) * dt

    # combine k1, k2, k3, k4
    x += (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6
    y += (k1_y + 2*k2_y + 2*k3_y + k4_y) / 6
    v_x += (k1_vx + 2*k2_vx + 2*k3_vx + k4_vx) / 6
    v_y += (k1_vy + 2*k2_vy + 2*k3_vy + k4_vy) / 6

    return x, y, v_x, v_y

# define plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-r_t*2, r_t*2)
ax.set_ylim(-r_t*2, r_t*2)

# plot black hole
ax.add_patch(plt.Circle((0, 0), r_bh, color='black', fill=True)) # black hole (shwarzschild radius)
ax.add_patch(plt.Circle((0, 0), r_t, color='blue', fill=False, linestyle='dashed')) # tidal radius
# ax.add_patch(plt.Circle((0, 0), r_roche, color='green', fill=False, linestyle='dashed')) # roche limit

# plot star and create scatter plot for star particles
star, = ax.plot([], [], 'o', color='orange', markersize=3)
star_cloud = ax.scatter(x_cloud, y_cloud, s=0.2, c='orange')
particles_consumed_count = ax.text(0.02, 0.95, '', transform=ax.transAxes)
timer = ax.text(0.02, 0.05, '', transform=ax.transAxes)

# add star trail lines
trail_star, = ax.plot([], [], '-', color='orange', linewidth=1)
trail_x, trail_y = [], []

# initialize plot
def init():
    star.set_data([], [])
    star_cloud.set_offsets(np.empty((0, 2)))
    return star, star_cloud

# update plot across time steps
def update(frame):
    global x_star, y_star, v_x_star, v_y_star
    global x_cloud, y_cloud, v_x_cloud, v_y_cloud
    global n_particles_consumed, elapsed_time
    global ani
    
    x_star, y_star, v_x_star, v_y_star = rk4_point(x_star, y_star, v_x_star, v_y_star)
    x_cloud, y_cloud, v_x_cloud, v_y_cloud = rk4_cloud(x_cloud, y_cloud, v_x_cloud, v_y_cloud)

    # in the case star falls into event horizon
    if math.sqrt(x_star**2 + y_star**2) <= r_bh: 
        x_star, y_star = 0, 0
        v_x_star, v_y_star = 0, 0
    r = np.sqrt(x_cloud**2 + y_cloud**2)
    mask = r <= r_bh
    # update count of consumed particles
    consumed = np.sum(mask & (x_cloud != 0))
    n_particles_consumed += consumed
    particles_consumed_count.set_text(f"Particles consumed: {n_particles_consumed} / {n_particles}")
    particles_consumed_over_time.append(n_particles_consumed)
    x_cloud[mask], y_cloud[mask] = 0, 0
    v_x_cloud[mask], v_y_cloud[mask] = 0, 0

    # update timer
    elapsed_time += dt
    time_series.append(elapsed_time)
    timer.set_text(f"Elapsed time: {int(elapsed_time / 60)} min")

    # update star data
    star.set_data([x_star], [y_star])

    # update cloud data
    speed = np.sqrt(v_x_cloud**2 + v_y_cloud**2)
    speed_norm = np.clip(speed / 1e8, 0, 1) 
    star_color = np.column_stack((speed_norm, np.full_like(speed_norm, 0.2), 1.0 - speed_norm))
    star_cloud.set_offsets(np.column_stack((x_cloud, y_cloud)))
    star_cloud.set_color(star_color)
    
    # update star_trail
    trail_x.append(x_star)
    trail_y.append(y_star)
    trail_star.set_data(trail_x, trail_y)

    # when finished, end animation
    if n_particles_consumed == n_particles: ani.event_source.stop()

    return star, star_cloud, particles_consumed_count, timer, # trail_star

# funciton to run animation
def run_animation():
    global ani

    ani_running = True

    # pause functionality
    def onClick(event):
        nonlocal ani_running
        if ani_running:
            ani.event_source.stop()
            ani_running = False
        else:
            ani.event_source.start()
            ani_running = True

    fig.canvas.mpl_connect('button_press_event', onClick)
    ani = animation.FuncAnimation(fig, update, frames=500, init_func=init, blit=True, interval=10)

# plot animation
run_animation()
plt.show()

# get graph for particles consumed over time
# plt.figure()
# plt.plot(np.array(time_series) / 60, particles_consumed_over_time)
# plt.xlabel("Time (minutes)")
# plt.ylabel("Particles Consumed")
# plt.title("Particles Eaten by Black Hole Over Time")
# plt.grid(True)
# plt.show()