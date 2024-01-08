import taichi as ti
import taichi.math as tm
import time

# from Scene import init_base_scene
from Object import Object, Transform, Material
from Camera import Camera

NUM_OBJECT = 2


ti.init(arch=ti.vulkan)

# initialize the window and camera
image_resolution = (640, 480)
aspect_ratio = image_resolution[0] / image_resolution[1]
image_pixels = ti.Vector.field(3, ti.float32, image_resolution)

camera = Camera(position=tm.vec3(0, 0, 4),
                lookat=tm.vec3(0, 0, 2),
                up=tm.vec3(0, 1, 0),
                vertical_fov=30,
                aspect_ratio=aspect_ratio,
                aperture=0.01,
                focus=4)

# initialize the scene object
scene_objects = Object.field(shape=NUM_OBJECT)
scene_objects[0] = Object.Object(params=0.5,
                                 type=Object.SHAPE_SPHERE,
                                 transform=Object.Transform(position=tm.vec3(0, 0, -1)),
                                 material=Object.Material(albedo=tm.vec3(1, 0, 0)))
scene_objects[1] = Object.Object(params=0.5,
                                 type=Object.SHAPE_CUBE,
                                 transform=Object.Transform(position=tm.vec3(1, 0, -1)),
                                 material=Object.Material(albedo=tm.vec3(0, 1, 0)))


@ti.kernel
def render(delta_time: float, render_camera: Camera):
    for i, j in image_pixels:
        u = i / image_resolution[0]
        v = j / image_resolution[1]

        ray = render_camera.get_ray(u, v, tm.vec4(1.0))
        record = ray.raycast(scene=scene)

        if record.hit:
            ray.color.rgb = 0.5 + 0.5 * record.hit_object.get_normal(record.position)
            ray.color.rgb *= record.hit_object.material.albedo
        else:
            ray.color.rgb = tm.vec3(0.0, 0.0, 1.0)

        image_pixels[i, j] = ray.color.rgb


window = ti.ui.Window("Taichi Renderer", image_resolution)
canvas = window.get_canvas()

start_time = time.time()
while window.running:
    window.get_event()

    if window.is_pressed('a'):
        camera.position += tm.vec3(0.05, 0, 0)
    elif window.is_pressed('d'):
        camera.position += tm.vec3(-0.05, 0, 0)
    elif window.is_pressed('s'):
        camera.position += tm.vec3(0, 0, 0.05)
    elif window.is_pressed('w'):
        camera.position += tm.vec3(0, 0, -0.05)
    elif window.is_pressed('q'):
        camera.position += tm.vec3(0, -0.05, 0)
    elif window.is_pressed('e'):
        camera.position += tm.vec3(0, 0.05, 0)

    render(time.time() - start_time, camera)
    canvas.set_image(image_pixels)
    window.show()