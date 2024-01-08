import taichi as ti
import taichi.math as tm

from metadata import *

ti.init(arch=ti.vulkan)

# container prep
x_hat = ti.Vector.field(3, dtype=float, shape=(n, n))
x = ti.Vector.field(3, dtype=float, shape=(n, n))
v = ti.Vector.field(3, dtype=float, shape=(n, n))
gradient = ti.Vector.field(3, dtype=float, shape=(n, n))

vertices = ti.Vector.field(3, dtype=float, shape=n * n)
colors = ti.Vector.field(3, dtype=float, shape=n * n)
triangles = ti.field(dtype=int, shape=(n - 1) * (n - 1) * 6)

neighbor_offset = []
neighbor_offset.append(ti.Vector([0, -1]))
neighbor_offset.append(ti.Vector([0, 1]))
neighbor_offset.append(ti.Vector([-1, 0]))
neighbor_offset.append(ti.Vector([1, 0]))
neighbor_offset.append(ti.Vector([-1, -1]))
neighbor_offset.append(ti.Vector([1, 1]))
neighbor_offset.append(ti.Vector([-1, 1]))
neighbor_offset.append(ti.Vector([1, -1]))

sphere_pos = ti.Vector.field(3, dtype=float, shape=1)
sphere_pos[0] = [0, 0, 0]

# simulation setting
time = 0.0


@ti.kernel
def init_cloth():
    for i, j in x:
        x[i, j] = [i * grid_interval - 0.5 * grid_length,
                   0.8,
                   j * grid_interval - 0.5 * grid_length]
        v[i, j] = [0, 0, 0]

    for i, j in ti.ndrange(n-1, n-1):
        index = 6 * (i * (n - 1) + j)
        triangles[index] = i * n + j
        triangles[index + 1] = (i + 1) * n + j
        triangles[index + 2] = i * n + (j + 1)

        triangles[index + 3] = (i + 1) * n + (j + 1)
        triangles[index + 4] = i * n + (j + 1)
        triangles[index + 5] = (i + 1) * n + j

    for i in ti.ndrange(n * n):
        colors[i] = (0, 0.5, 0.5)


@ti.kernel
def handle_collision():
    for i, j in x:
        vertex2sphere = x[i, j] - sphere_pos[0]
        dist = vertex2sphere.norm()
        if dist <= sphere_radius:
            normal = vertex2sphere.normalized()
            target_x = sphere_pos[0] + sphere_radius * normal
            v[i, j] = 1 / dt * (target_x - x[i, j])
            x[i, j] = target_x


@ti.kernel
def implicit_update():
    for i, j in v:
        v[i, j] *= damping
        x_hat[i, j] = x[i, j] + dt * v[i, j]
        x[i, j] = x_hat[i, j]


@ti.kernel
def implicit_substep():
    for i, j in x:
        gradient[i, j] = t_inverse * mass * (x[i, j] - x_hat[i, j]) - gravity

    for row, col in gradient:
        for offset in ti.static(neighbor_offset):
            nrow = row + offset[0]    # neighbor row
            ncol = col + offset[1]    # neighbor col
            coef = spring_k * (1 - offset.norm() * grid_interval / (x[row, col] - x[nrow, ncol]).norm())

            spring = coef * (x[row, col] - x[nrow, ncol])
            gradient[row, col] += spring
            gradient[nrow, ncol] -= spring

    coef = t_inverse * mass + 4 * spring_k
    for i, j in x:
        x[i, j] -= (1 / coef) * gradient[i, j]


@ti.kernel
def assign_vertices():
    for i, j in x:
        vertices[i * n + j] = x[i, j]


# simulation run
window = ti.ui.Window(window_name, window_dimension, vsync=True)
canvas = window.get_canvas()
canvas.set_background_color(background_color)
scene = window.get_scene()
camera = ti.ui.Camera()

init_cloth()

while window.running:

    implicit_update()
    for i in range(num_substep):
        implicit_substep()
    handle_collision()
    assign_vertices()
    time += dt

    camera.position(0.0, 0.0, 3)
    camera.lookat(0.0, 0.0, 0)
    scene.set_camera(camera)

    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.mesh(vertices,
               indices=triangles,
               per_vertex_color=colors,
               two_sided=True)

    # Draw a smaller ball to avoid visual penetration
    scene.particles(sphere_pos, radius=sphere_radius)
    canvas.scene(scene)
    window.show()
