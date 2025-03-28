import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# constants
G = 6.67430e-11  # gravitational constant
m_star = 2e30     # mass of the star 
r_star = 7e8      # radius of the star 
n_par = 50  # number of particles to simulate

# create particles within a certain radius
theta = np.linspace(0, 2*np.pi, n_par)
radii = np.random.uniform(0, r_star, n_par)  # random radii
x = radii * np.cos(theta)  # x positions
y = radii * np.sin(theta)  # y positions

# mass of particles
masses = np.ones(n_par) * 1e24

def f_grav(m1, m2, x, y):
    r = math.sqrt(x**2 + y**2)
    if r == 0: return 0, 0
    f = -1 * G * m1 * m2 / r**2
    fx = f * (x / r)
    fy = f * (y / r)
    return fx, fy

def t_force(x,y,r_star, m_star, m_bh):

    r_minus = np.sqrt((x-r_star/2) ** 2 + ((y - r_star/2) ** 2))
    r_plus = np.sqrt((x+r_star/2) ** 2 + ((y + r_star/2) ** 2))

    f_x_m, f_y_m = f_grav(m_bh, m_star, x - r_star/2, y)
    f_x_p, f_y_p = f_grav(m_bh, m_star, x + r_star/2, y)

    tidal_f_x = f_x_p - f_x_m
    tidal_f_y = f_y_p - f_y_m

# setup the plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-r_star*10, r_star*10)
ax.set_ylim(-r_star*10, r_star*10)
ax.set_aspect('equal', 'box')
ax.set_xlabel('X')
ax.set_ylabel('Y')

# create scatter plot for particles
scatter = ax.scatter(x, y)
plt.show()
    
def update(frame):
    #bleh

    


