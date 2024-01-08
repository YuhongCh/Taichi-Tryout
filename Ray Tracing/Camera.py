import taichi as ti
import taichi.math as tm

from Ray import Ray

@ti.dataclass
class Camera:
    position:     tm.vec3
    lookat:       tm.vec3
    up:           tm.vec3
    vertical_fov: float
    aspect_ratio: float
    aperture:     float
    focus:        float

    @ti.func
    def get_random_disk(self) -> tm.vec2:
        x = ti.random()
        a = 2 * 3.1415926 * ti.random()
        return ti.sqrt(x) * tm.vec2(tm.cos(a), tm.sin(a))


    @ti.func
    def get_ray(self, u: float, v: float, color: tm.vec4) -> Ray:
        """
        Get Ray object start from camera position and point toward (u,v) coordinate of screen
        :param u: value [0, 1], 0 means left most screen, 1 means right most screen
        :param v: value [0, 1], 0 means lower most screen, 1 means upper most screen
        :param color: RGBA color value
        :return: Ray shoot from camera position to (u, v) coordinate
        """
        half_height = tm.tan(0.5 * tm.radians(self.vertical_fov))
        half_width = half_height * self.aspect_ratio
        zdir = tm.normalize(self.position - self.lookat)
        xdir = tm.normalize(tm.cross(self.up, zdir))
        ydir = tm.cross(zdir, xdir)

        lower_left_corner = self.position - self.focus * zdir \
                            - half_width * self.focus * xdir \
                            - half_height * self.focus * ydir
        horizontal = 2 * half_width * self.focus * xdir
        vertical = 2 * half_height * self.focus * ydir

        lens_radius = self.aperture * 0.5
        random_disk = lens_radius * self.get_random_disk()
        offset = xdir * random_disk.x + ydir * random_disk.y

        ray_position = self.position + offset
        ray_lookat = lower_left_corner + u * horizontal + v * vertical
        ray_direction = tm.normalize(ray_lookat - ray_position)
        return Ray(position=ray_position, direction=ray_direction, color=color)

