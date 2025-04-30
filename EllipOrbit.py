import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize
from matplotlib.animation import FuncAnimation

# === 1) Turn on interactive mode so figures show up right away ===
plt.ion()

# Create new figure window for KE and PE
fig_energy, ax_energy = plt.subplots()
line_ke, = ax_energy.plot([], [], label='Kinetic Energy (KE)')
line_pe, = ax_energy.plot([], [], label='Potential Energy (PE)')
#line_te, = ax_energy.plot([], [], label='Total Energy (TE)')

ax_energy.set_xlim(0, 10)  
ax_energy.set_ylim(-1e42, 1e42)
ax_energy.set_xlabel('Time (s)')
ax_energy.set_ylabel('Energy (Joules)')
ax_energy.legend()

# (rest of your setup: constants, initial conditions, star/cloud generation...) 

ani = None

KE_list = []     # Total Kinetic Energy of the system
PE_list = []     # Total Potential Energy of the system
TE_list = []     # Total Energy (KE + PE)

# define constants
solar_mass = 1.98892e30 # one solar mass [kg]
solar_radius = 6.957e8 # one solar radius [m]
c = 2.99792458e8 # speed of light [m/s]
G = 6.67408e-11 # gravitational constant [m^3 kg^-1 s^-2]
dt = 500 # time step [s]
elapsed_time = 0 # current time [s]
num_steps = 10000 # number of simulation time steps
particle_count = 0
#
# define variables
m_bh = 1e6 * solar_mass # mass of black hole [kg]
m_bh_test = 0.7e6 * solar_mass
m_star = 5 * solar_mass # mass of star [kg] (use .15, .30, .40, .50, .70, 1, 3, 10 M. for experiments)
r_bh = 2 * G * m_bh_test / c**2 # schwarzschild radius of black hole [m]
r_s = 2 * solar_radius # radius of star [m]
r_roche = r_s * (2 * m_bh / m_star)**(1.0/3.0) # roche limit of black hole [m]
r_t = (m_bh / m_star)**(1/3) * r_s # tidal radius of black hole
#r_t = ((2.8*10080/(2*math.pi))**2*G*(solar_mass*1e6))**(1.0/3.0) # From Hayasaki paper
v_t = math.sqrt(G * m_bh / r_t) # initial velocity condition
n_particles = 1000 # number of particles in star
m_particles = np.ones(n_particles)*m_star/n_particles # mass of particles [kg]
n_particles_consumed = 0 # number of particles consumed by black hole
particles_consumed_over_time = [] # number of particles consumed by black hole over time
time_series = [] # to store corresponding time

# define initial conditions
# init_x = r_bh*5
# init_y = r_bh*5
# init_v_x = -6e7
# init_v_y = 6e7

# Case 10
"""
init_x, init_y= 0.556*r_t, -1.71*r_t
init_v_x, init_v_y = 0.377*v_t, 0.1225*v_t
"""
# Case 5

init_x, init_y= 0*r_t, -2.5*r_t
init_v_x, init_v_y = 0.359*v_t, 0.251*v_t
#init_v_x, init_v_y = 0.559*v_t, 0.451*v_t
min_max_flip = True

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

# define function for force of grav on star -- paczyński–wiita potential = phi = - G * M / (r - r_bh)
def f_grav_pw_point(m1, m2, x, y):
    r = math.sqrt(x**2 + y**2)
    if r <= r_bh: return 0, 0
    f = -1 * G * m1 * m2 / (r - r_bh)**2
    fx = f * (x / r) 
    fy = f * (y / r)
    return fx, fy
def f_grav_pw_cloud(m1, m2, x, y):
    r = np.sqrt(x**2 + y**2)
    fx = np.zeros(n_particles)
    fy = np.zeros(n_particles)
    mask = r > r_bh
    f = np.where(mask, -1 * G * m1 * m2 / (r - r_bh)**2, 0)
    fx[mask] = f[mask] * (x[mask] / r[mask]) 
    fy[mask] = f[mask] * (y[mask] / r[mask]) 
    return fx, fy

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
        f_x, f_y = f_grav_pw_point(m_bh_test, m_star, x, y)
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
        f_x, f_y = f_grav_pw_cloud(m_bh_test, m_particles, x, y)
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
ax.set_xlim(-r_t*4, r_t*4)
ax.set_ylim(-r_t*4, r_t*4)

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
# KE = []
# PE = []
orbit_start = 0
should_close = False
energies = [[0,0,0]]  # Will store [time, KE, PE] rows
def update(frame):
    global x_star, y_star, v_x_star, v_y_star
    global x_cloud, y_cloud, v_x_cloud, v_y_cloud
    global n_particles_consumed, elapsed_time
    global ani, min_max_flip, orbit_start, should_close

    def KE_PE_graphs(x, y, vx, vy, m):
        v_squared = vx**2 + vy**2
        r = np.sqrt(x**2 + y**2)
        ke = 0.5*m*v_squared
        pe = -G*m*m_bh_test/r
        
        avg_ke = np.sum(ke)/len(x)
        avg_pe = np.sum(pe)/len(x)

        # KE.append(avg_ke)
        # PE.append(avg_pe)
        return avg_ke, avg_pe
    
    avg_ke, avg_pe = KE_PE_graphs(x_cloud, y_cloud, v_x_cloud, v_y_cloud, m_particles)
    save_energy = [elapsed_time, avg_ke, abs(avg_pe)]
    last_avg_ke = energies[-1][1]
    energies.append(save_energy)

    energy_array = np.array(energies)
    print(save_energy)
    

    if min_max_flip == True:
        print("Hello")
        if abs(last_avg_ke) < abs(avg_ke):
            min_max_flip = False
            print("new orbit begins")
            
            if orbit_start > 1:
                print(f"time is {elapsed_time/60}" )
                #ani.event_source.stop()
                should_close = True
            orbit_start += 1
    else:
        if abs(last_avg_ke) > abs(avg_ke):
            min_max_flip = True
            print("min ke reached")
            orbit_start += 1



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

    if elapsed_time >= 75000*60:
        ani.event_source.stop()
        ani_energy.event_source.stop()

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

    update_energy(frame)

    if should_close == True:
        ani.event_source.stop()
        ani_energy.event_source.stop()
    return star, star_cloud, particles_consumed_count, timer, trail_star

# === 2) Give update_energy a frame argument ===
def update_energy(frame):
# def update_energy():
    if len(energies) < 2:
        return line_ke, line_pe, #line_te

    energy_array = np.array(energies)
    times = energy_array[:, 0]
    kes = energy_array[:, 1]
    pes = energy_array[:, 2]
    tes = kes + pes

    line_ke.set_data(times, kes)
    line_pe.set_data(times, pes)
    #line_te.set_data(times, tes)

    # Dynamic rescaling
    ax_energy.set_xlim(0, max(times) * 1.1)
    y_min = min(np.min(kes), np.min(pes)) * 1.1
    y_max = max(np.max(kes), np.max(pes)) * 1.1
    ax_energy.set_ylim(y_min, y_max)

    return line_ke, line_pe, #line_te

# === 3) Provide an init_func for the energy animation ===
def init_energy():
    line_ke.set_data([], [])
    line_pe.set_data([], [])
    #line_te.set_data([], [])
    return line_ke, line_pe, #line_te

# Your existing run_animation, but pass init_energy into FuncAnimation
def run_animation():
    global ani, ani_energy

    # pause/play on click
    ani_running = True
    def onClick(event):
        nonlocal ani_running
        if ani_running:
            ani.event_source.stop()
            ani_running = False
        else:
            ani.event_source.start()
            ani_running = True

    fig.canvas.mpl_connect('button_press_event', onClick)

    ani = FuncAnimation(fig, update,
                        frames=500,
                        init_func=init,
                        blit=False,
                        interval=50)

    ani_energy = FuncAnimation(fig_energy,
                                update_energy,
                                frames=500,
                                init_func=init_energy,
                                blit=False,
                                interval=50)

run_animation()
plt.show(block=True)
