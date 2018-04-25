import math


def generate_point(base, index):
    f, result, i = 1.0, 0.0, index
    while i > 0.0:
        f /= base
        result += f * (i % base)
        i = math.floor(i / base)

    return result


def generate_sphere(base, total):
    points = [(generate_point(base, i), i) for i in xrange(total)]
    sphere_points = []
    for point in points:
        t = 2.0 * point[0] - 1.0
        phi = (point[1] + 0.5) / total
        phirad = phi * math.pi * 2.0
        st = math.sqrt(1.0 - t*t)
        sphere_points.append((st * math.cos(phirad), st * math.sin(phirad), t))

    return sphere_points
