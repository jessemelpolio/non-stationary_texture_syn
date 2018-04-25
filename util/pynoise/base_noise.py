import constants


def _integer_surround(number):
    """Return the 2 closest integers to number, smaller integer first."""
    if number > 0:
        return int(number), int(number) + 1
    else:
        return int(number) - 1, int(number)


def _interpolate(first, second, between):
    return first + (second - first) * between


def _cubic_scurve(value):
    return value * value * (3.0 - 2.0 * value)


def _quintic_scurve(value):
    return (6.0 * value ** 5) - (15.0 * value ** 4) + (10.0 * value ** 3)


def _get_vector(x, y, z, seed):
    index = (
        constants.x_noise * x +
        constants.y_noise * y +
        constants.z_noise * z +
        constants.seed * seed
    )
    index ^= index >> 8  # xorshift random

    return constants.vectors[index % len(constants.vectors)]


def noise_vector(x, y, z, int_x, int_y, int_z, seed_offset):
    vector = _get_vector(int_x, int_y, int_z, seed_offset)
    diff_vector = (x - int_x, y - int_y, z - int_z)

    return (
        vector[0] * diff_vector[0] +
        vector[1] * diff_vector[1] +
        vector[2] * diff_vector[2]
    )


def noise_gradients(x, y, z, seed_offset):
    unit_x = _integer_surround(x)
    unit_y = _integer_surround(y)
    unit_z = _integer_surround(z)

    x_decimal = _cubic_scurve(x - unit_x[0])
    y_decimal = _cubic_scurve(y - unit_y[0])
    z_decimal = _cubic_scurve(z - unit_z[0])

    n000 = noise_vector(x, y, z, unit_x[0], unit_y[0], unit_z[0], seed_offset)
    n100 = noise_vector(x, y, z, unit_x[1], unit_y[0], unit_z[0], seed_offset)
    n010 = noise_vector(x, y, z, unit_x[0], unit_y[1], unit_z[0], seed_offset)
    n110 = noise_vector(x, y, z, unit_x[1], unit_y[1], unit_z[0], seed_offset)
    n001 = noise_vector(x, y, z, unit_x[0], unit_y[0], unit_z[1], seed_offset)
    n101 = noise_vector(x, y, z, unit_x[1], unit_y[0], unit_z[1], seed_offset)
    n011 = noise_vector(x, y, z, unit_x[0], unit_y[1], unit_z[1], seed_offset)
    n111 = noise_vector(x, y, z, unit_x[1], unit_y[1], unit_z[1], seed_offset)

    interp1 = _interpolate(n000, n100, x_decimal)
    interp2 = _interpolate(n010, n110, x_decimal)
    interp3 = _interpolate(n001, n101, x_decimal)
    interp4 = _interpolate(n011, n111, x_decimal)
    interp5 = _interpolate(interp1, interp2, y_decimal)
    interp6 = _interpolate(interp3, interp4, y_decimal)

    return _interpolate(interp5, interp6, z_decimal)
