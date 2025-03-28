import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# define constants
solar_mass = 1.98892e30 # one solar mass [kg]
c = 2.99792458e8 # speed of light [m/s]
G = 6.67408e-11 # gravitational constant [m^3 kg^-1 s^-2]
dt = 10 # time step (s)
num_steps = 10000 # number of sim time steps

# define variables
m_bh = 10e6 * solar_mass # mass of black hole [kg]
m_star = 10 * solar_mass # mass of red giant star [kg]
r_s = 2 * G * m_bh / c**2 # schwarzschild radius of black hole [m]

# define initial conditions of star
x, y = r_s*5, r_s*5
# v_x, v_y = -50000000, 0

r_initial = math.sqrt(x**2 + y**2)
v_orbit = math.sqrt(G * m_bh/r_initial)
v_x = -v_orbit * (y/r_initial)
v_y = v_orbit * (x/r_initial)

# define function for force of grav on star in x and y directions
def f_grav(m1, m2, x, y):
    r = math.sqrt(x**2 + y**2)
    if r == 0: return 0, 0
    f = -1 * G * m1 * m2 / r**2
    fx = f * (x / r)
    fy = f * (y / r)
    return fx, fy

# define plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-r_s*10, r_s*10)
ax.set_ylim(-r_s*10, r_s*10)

# plot star, black hole, and event horizon
star, = ax.plot([], [], 'o', color='black', markersize=3)
ax.plot(0, 0, 'o', color='black', markersize=10)
ax.add_patch(plt.Circle((0, 0), r_s, color='black', fill=False, linestyle='dashed'))

# initialize plot
def init():
    star.set_data([], [])
    return star,

# update plot across time steps
def update(frame):
    global x, y, v_x, v_y

    # calc acceleration from grav in x and y directions
    f_x, f_y = f_grav(m_bh, m_star, x, y)
    a_x = f_x / m_star
    a_y = f_y / m_star

    # update star velocity and position
    v_x += a_x * dt
    x += v_x * dt
    v_y += a_y * dt
    y += v_y * dt
    
    # in the case star falls into event horizon
    if math.sqrt(x**2 + y**2) <= r_s: x, y = 0, 0

    star.set_data([x], [y])
    return star,

# plot animation
ani = animation.FuncAnimation(fig, update, frames=500, init_func=init, blit=True, interval=10)
plt.show()