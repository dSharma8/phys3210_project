import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# define constants
solar_mass = 1.98892e30 # one solar mass [kg]
solar_radius = 6.957e8 # one solar radius [m]
c = 2.99792458e8 # speed of light [m/s]
G = 6.67408e-11 # gravitational constant [m^3 kg^-1 s^-2]
dt = 10 # time step (s)
num_steps = 10000 # number of sim time steps

# define variables
m_bh = 1e6 * solar_mass # mass of black hole [kg]
m_star = 10 * solar_mass # mass of red giant star [kg]
r_bh = 2 * G * m_bh / c**2 # schwarzschild radius of black hole [m]
r_s = solar_radius # radius of star [m]
n_particles = 1000 # number of particles in star

# mass of particles
m_particles = np.ones(n_particles) * m_star / n_particles

# generate particles
theta = np.linspace(0, 2*np.pi, n_particles)
radii = np.random.uniform(0, r_s, n_particles)  # random radii
x1 = radii * np.cos(theta)  # x positions
x1 += 5*r_bh
y1 = radii * np.sin(theta)  # y positions
y1 += 5*r_bh
v_x1 = np.ones(n_particles) * -3e7
v_y1 = np.ones(n_particles) * 3e7

# define function for force of grav on star in x and y directions
def f_grav(m1, m2, x, y):
    r = np.sqrt(x**2 + y**2)
    fx = np.zeros(n_particles)
    fy = np.zeros(n_particles)
    mask = r != 0
    f = np.where(mask, -1 * G * m1 * m2 / r**2, 0)
    fx[mask] = f[mask] * (x[mask] / r[mask]) 
    fy[mask] = f[mask] * (y[mask] / r[mask]) 
    return fx, fy

# define function for tidal forces across star
def t_force(x, y, r_s, m_star, m_bh):
    r_minus = np.sqrt((x-r_s/2) ** 2 + ((y - r_s/2) ** 2))
    r_plus = np.sqrt((x+r_s/2) ** 2 + ((y + r_s/2) ** 2))
    ax.plot(-r_bh, 0, 'o', color='black', markersize=10)

    f_x_m, f_y_m = f_grav(m_bh, m_star, x - r_s/2, y)
    f_x_p, f_y_p = f_grav(m_bh, m_star, x + r_s/2, y)

    tidal_f_x = f_x_p - f_x_m
    tidal_f_y = f_y_p - f_y_m

# rk4 method
def rk4(x, y, v_x, v_y):
    # calc acceleration from grav in x and y directions
    def accel(x, y):
        f_x, f_y = f_grav(m_bh, m_star, x, y)
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

# define plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-r_bh*15, r_bh*15)
ax.set_ylim(-r_bh*15, r_bh*15)

# plot black hole
ax.plot(0, 0, 'o', color='black', markersize=10)
ax.add_patch(plt.Circle((0, 0), r_bh, color='black', fill=False, linestyle='dashed'))

# create scatter plot for star particles
star1 = ax.scatter(x1, y1, s=0.2, c='orange')

# initialize plot
def init():
    star1.set_offsets(np.empty((0, 2)))
    return star1, 

# update plot across time steps
def update(frame):
    global x1, y1, v_x1, v_y1
    
    x1, y1, v_x1, v_y1 = rk4(x1, y1, v_x1, v_y1)

    # in the case star falls into event horizon
    r = np.sqrt(x1**2 + y1**2)
    mask = r <= r_bh
    x1[mask], y1[mask] = 0, 0
    v_x1[mask], v_y1[mask] = 0, 0

    star1.set_offsets(np.column_stack((x1, y1)))

    return star1,

# plot animation
ani = animation.FuncAnimation(fig, update, frames=500, init_func=init, blit=True, interval=10)
plt.show()
