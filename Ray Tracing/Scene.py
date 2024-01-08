import taichi as ti
import taichi.math as tm

import Object

@ti.dataclass
class Scene:
    num_object: int
    objects: ti.field(dtype=Object.Object)

    def __init__(self):
        self.objects = ti.field(Object.Object, shape=self.num_object)

    @ti.func
    def get_nearest_object(self, p: tm.vec3) -> (Object.Object, float):
        dist = 5000
        nearest_object = Object.Object()
        for i in self.objects:
            curr_dist = self.objects[i].get_signed_distance(p)
            if curr_dist < dist:
                dist = curr_dist
                nearest_object = self.objects[i]
        return nearest_object, dist


def init_base_scene() -> Scene:
    scene = Scene(num_object = 2)
    scene.objects[0] = Object.Object(params=0.5,
                                     type=Object.SHAPE_SPHERE,
                                     transform=Object.Transform(position=tm.vec3(0, 0, -1)),
                                     material=Object.Material(albedo=tm.vec3(1, 0, 0)))
    scene.objects[1] = Object.Object(params=0.5,
                                     type=Object.SHAPE_CUBE,
                                     transform=Object.Transform(position=tm.vec3(1, 0, -1)),
                                     material=Object.Material(albedo=tm.vec3(0, 1, 0)))
    return scene
