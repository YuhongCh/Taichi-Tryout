
import taichi.math as tm

"""
Below are Scene setting
"""
# window setting
window_name = "Efficient Simulation of Inextensible Cloth"
window_dimension = (1024, 1024)
background_color = (1, 1, 1)


# physical setting
dt = 1e-3
t_inverse = 1 / (dt * dt)
num_substep = 32
damping = 0.99
gravity = tm.vec3(0, -9.81, 0)

"""
Below are Cloth setting
"""
# number of vertices per side
n = 21

# total cloth grid length
grid_length = 2
grid_interval = grid_length / n

# mass matrix
mass = 1

# strain threshold
threshold = 10

"""
Below are Sphere setting
"""
sphere_radius = 0.3
