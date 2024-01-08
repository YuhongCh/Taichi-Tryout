import taichi as ti
import taichi.math as tm

from metadata import *

ti.init(arch=ti.vulkan)

# container prep
x = ti.Vector.field(3, dtype=float, shape=(n, n))
v = ti.Vector.field(3, dtype=float, shape=(n, n))

static_vertices = ti.Vector.field(3, dtype=float, shape=2)
vertices = ti.Vector.field(3, dtype=float, shape=n * n)
colors = ti.Vector.field(3, dtype=float, shape=n * n)
triangles = ti.field(dtype=int, shape=(n - 1) * (n - 1) * 6)

constraint = ti.field(dtype=float, shape=2*n*n)
grad_constraint = ti.field(dtype=float, shape=2*n*n)


@ti.func
def compute_constraint():
    for i, j in x:
        index = 2 * (i * n + j)
        constraint[index] = grad_constraint[index] = 0
        constraint[index + 1] = grad_constraint[index + 1] = 0
        if i + 1 < n:
            constraint[index] = (x[i + 1, j] - x[i, j]).norm_sqr() / grid_interval - grid_interval
            grad_constraint[index] = 2 * (x[i + 1, j] - x[i, j]).norm() / grid_interval
        if j + 1 < n:
            constraint[index + 1] = (x[i, j + 1] - x[i, j + 1]).norm_sqr() / grid_interval - grid_interval
            grad_constraint[index + 1] = 2 * (x[i, j + 1] - x[i, j + 1]).norm() / grid_interval


@ti.kernel
def init_cloth():
    for i, j in x:
        x[i, j] = [i * grid_interval - 0.5 * grid_length,
                   0.8,
                   j * grid_interval - 0.5 * grid_length]
        v[i, j] = [0, 0, 0]

    static_vertices[0] = x[0, 0]
    static_vertices[1] = x[n-1, 0]

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
def update():
    for i, j in v:
        v[i, j] *= damping
        v[i, j] += gravity * dt / mass
        x[i, j] += v[i, j] * dt


@ti.kernel
def fast_projection():
    compute_constraint()
    a = constraint.sum()


@ti.kernel
def assign_vertices():
    # use to keep one side of vertices static
    x[0, 0] = static_vertices[0]
    x[n-1, 0] = static_vertices[1]

    for i, j in x:
        vertices[i * n + j] = x[i, j]


# simulation run
window = ti.ui.Window(window_name, window_dimension, vsync=True)
canvas = window.get_canvas()
canvas.set_background_color(background_color)
scene = window.get_scene()
camera = ti.ui.Camera()

time = 0.0
init_cloth()

while window.running:
    update()
    fast_projection()
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
    canvas.scene(scene)
    window.show()