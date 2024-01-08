import taichi as ti
import taichi.math as tm

from metadata import *

ti.init(arch=ti.vulkan)

# container prep
x = ti.Vector.field(3, dtype=float, shape=(n, n))
v = ti.Vector.field(3, dtype=float, shape=(n, n))

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


@ti.func
def get_force(row: int, col: int) -> tm.vec3:
    force = tm.vec3(0.0, 0.0, 0.0)
    force += gravity
    for offset in ti.static(neighbor_offset):
        nrow = row + offset[0]    # neighbor row
        ncol = col + offset[1]    # neighbor col
        if 0 <= nrow < n and 0 <= ncol < n:
            x_diff = x[row, col] - x[nrow, ncol]

            ref_dist = (offset * grid_interval).norm()
            curr_dist = x_diff.norm()
            xd_norm = x_diff.normalized()
            # spring force
            force += -spring_k * (curr_dist - ref_dist) * xd_norm
    return force


@ti.kernel
def handle_collision():
    for i, j in x:
        vertex2sphere = x[i, j] - sphere_pos[0]
        dist = vertex2sphere.norm()
        if dist <= sphere_radius:
            normal = vertex2sphere.normalized()
            # impulse approach
            x[i, j] = sphere_pos[0] + sphere_radius * normal
            if v[i, j].dot(normal) < 0:
                vn = v[i, j].dot(normal) * normal
                vt = v[i, j] - vn
                alpha = max(0, 1 - mu_T * (1 + mu_N) * vn.norm() / vt.norm())
                v[i, j] = -mu_N * vn + alpha * vt


@ti.kernel
def explicit_update():
    for i, j in v:
        v[i, j] *= damping
        force = get_force(i, j)
        v[i, j] += force * dt / mass
        x[i, j] += v[i, j] * dt


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

    for i in range(num_substep):
        explicit_update()
        handle_collision()
        time += dt
    assign_vertices()

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
