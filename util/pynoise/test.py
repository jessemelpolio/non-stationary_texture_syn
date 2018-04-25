from PIL import Image
import perlin

perlin_generator = perlin.Perlin(frequency=50)
noise_image = Image.new("RGB", (500, 500))

for x in xrange(500):
    for y in xrange(500):
        colour = (int((perlin_generator.value(x/500.0, y/500.0, 0.0) + 1) * 128), ) * 3
        noise_image.putpixel((x, y), colour)

out = open("perlin.png", "w")
noise_image.save(out)
out.close()