import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# define constants
solar_mass = 1.98892e30 # one solar mass [kg]
c = 2.99792458e8 # speed of light [m/s]
G = 6.67408e-11 # gravitational constant [m^3 kg^-1 s^-2]
dt = 100 # time step (s)
num_steps = 10000 # number of sim time steps

# define variables
m_bh = 10e6 * solar_mass # mass of black hole [kg]
m_star = 10 * solar_mass # mass of red giant star [kg]
r_bh = 2 * G * m_bh / c**2 # schwarzschild radius of black hole [m]

# define initial conditions of star
x1, y1 = r_bh*5, r_bh*5
v_x1, v_y1 = -3e7, 3e7

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
ax.set_xlim(-r_bh*15, r_bh*15)
ax.set_ylim(-r_bh*15, r_bh*15)

# plot star, black hole, and event horizon
star1, = ax.plot([], [], 'o', markersize=3)
ax.plot(0, 0, 'o', color='black', markersize=10)
ax.add_patch(plt.Circle((0, 0), r_bh, color='black', fill=False, linestyle='dashed'))

# add star trail lines
# trail1, = ax.plot([], [], '-', color='orange', linewidth=1)
# trail_x1, trail_y1 = [], []

# initialize plot
def init():
    star1.set_data([], [])
    return star1, 

# update plot across time steps
def update(frame):
    global x1, y1, v_x1, v_y1

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
    
    x1, y1, v_x1, v_y1 = rk4(x1, y1, v_x1, v_y1)

    # in the case star falls into event horizon
    if math.sqrt(x1**2 + y1**2) <= r_bh: 
        x1, y1 = 0, 0
        v_x1, v_y1 = 0, 0

    # trail_x1.append(x1)
    # trail_y1.append(y1)
    # trail1.set_data(trail_x1, trail_y1)

    speed = math.sqrt(v_x1**2 + v_y1**2)
    speed_norm = min(speed / 1e8, 1.0) 
    star_color = (speed_norm, 0.2, 1.0 - speed_norm)
    star1.set_color(star_color)
    star1.set_data([x1], [y1])

    return star1, # trail1

# plot animation
ani = animation.FuncAnimation(fig, update, frames=500, init_func=init, blit=True, interval=10)
plt.show()