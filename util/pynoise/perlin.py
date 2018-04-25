from base_noise import noise_gradients


class Perlin(object):
    def __init__(
        self,
        frequency=1.0,
        lacunarity=1.0,
        octaves=6,
        persistance=0.5,
        seed=0,
    ):
        self.frequency = frequency
        self.lacunarity = lacunarity
        self.octaves = octaves
        self.persistance = persistance
        self.seed = seed

    def value(self, x, y, z):
        value, signal, persist = 0.0, 0.0, 1.0

        x *= self.frequency
        y *= self.frequency
        z *= self.frequency

        for i in xrange(self.octaves):
            value += noise_gradients(x, y, z, self.seed + i) * persist

            x *= self.lacunarity
            y *= self.lacunarity
            z *= self.lacunarity
            persist *= self.persistance

        return value
