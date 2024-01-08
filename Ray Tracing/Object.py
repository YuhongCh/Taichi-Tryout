import taichi as ti
import taichi.math as tm

SHAPE_SPHERE = 1
SHAPE_CUBE = 2


@ti.dataclass
class Material:
    albedo: tm.vec3


@ti.dataclass
class Transform:
    position: tm.vec3
    rotation: tm.vec4   # use quaterion
    scale:    tm.vec3


@ti.dataclass
class Object:
    material:   Material
    transform:  Transform
    params:     float       # for sphere use for radius, for cube use for length
    type:       ti.u8

    @ti.func
    def get_signed_distance(self, p: tm.vec3) -> float:
        signed_distance = 0.0
        if self.type == SHAPE_SPHERE:
            signed_distance = tm.length(self.transform.position - p) - self.params
        elif self.type == SHAPE_CUBE:
            q = abs(p - self.transform.position) - self.params
            signed_distance = tm.length(max(q, 0.0)) + min(max(q.x, q.y, q.z), 0.0)
        return signed_distance

    @ti.func
    def get_normal(self, p: tm.vec3) -> tm.vec3:
        epsilon = tm.vec2(1, -1)
        normal = epsilon.xyy * self.get_signed_distance(p + epsilon.xyy * 0.0001) + \
                 epsilon.yyx * self.get_signed_distance(p + epsilon.yyx * 0.0001) + \
                 epsilon.yxy * self.get_signed_distance(p + epsilon.yxy * 0.0001) + \
                 epsilon.xxx * self.get_signed_distance(p + epsilon.xxx * 0.0001)
        return tm.normalize(normal)
