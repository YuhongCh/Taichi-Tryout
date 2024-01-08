import taichi as ti
import taichi.math as tm

from Scene import Scene
from Object import Object

PRECISION = 0.0001
MAX_RAYMARCHING = 512
MIN_TIME = 0.1
MAX_TIME = 2000.0


@ti.dataclass
class RayHitRecord:
    position:   tm.vec3
    time:       float
    hit:        bool
    hit_object: Object


@ti.dataclass
class Ray:
    position:   tm.vec3
    direction:  tm.vec3
    color:      tm.vec4

    @ti.func
    def at(self, time: float) -> tm.vec3:
        return self.position + self.direction * time

    @ti.func
    def raycast(self, scene: Scene) -> RayHitRecord:
        record = RayHitRecord(position=self.position, time=MIN_TIME, hit=False)

        for _ in range(MAX_RAYMARCHING):
            record.position = self.at(record.time)
            nearest_object, dist = scene.get_nearest_object(record.position)

            if dist < PRECISION:
                record.hit = True
                record.hit_object = nearest_object
                break
            record.time += dist
            if dist > MAX_TIME:
                break
        return record





