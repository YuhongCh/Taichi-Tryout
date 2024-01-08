import taichi as ti
import taichi.math as tm

"""
Below are Scene setting
"""
# window setting
window_name = "Basic Cloth Simulation Demo"
window_dimension = (1024, 1024)
background_color = (1, 1, 1)


# physical setting
dt = 0.03
t_inverse = 1 / (dt * dt)
num_substep = 50
damping = 0.99
gravity = tm.vec3(0, -9.81, 0)
mu_T = 0.8
mu_N = 0.0

"""
Below are Cloth setting
"""
# number of vertices per side
n = 64

# total cloth grid length
grid_length = 2
grid_interval = grid_length / n

# mass matrix
mass = 1

# spring coefficient
spring_k = 8000


"""
Below are Sphere setting
"""
sphere_radius = 0.4
