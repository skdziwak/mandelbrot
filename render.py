import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from matplotlib import cm
from matplotlib import pyplot as plt
from PIL import Image
from pycuda.compiler import SourceModule
import shutil, os

def read(path):
    with open(path, encoding='utf-8') as f:
        return f.read()

BLOCK = (32, 32, 1)
GRID = (40, 40)
FRAMES = 4000
ZPF = 1.02
ITERATIONS = 500
X = -1.0909471548936537
Y = 0.236112271234165140
WIDTH = GRID[0] * BLOCK[0]
HEIGHT = GRID[1] * BLOCK[1]
FPS = 30

if os.path.exists('tmp'):
    shutil.rmtree('tmp')
os.mkdir('tmp')

print('Initializing')
mod = SourceModule(read('mandelbrot.cpp'))
mandelbrot = mod.get_function("mandelbrot")

print('Allocating buffer')
dest = np.zeros((HEIGHT, WIDTH), dtype=np.float32)

zoom = 1.5
params = np.array([zoom, X, Y, ITERATIONS], dtype=np.float32)

print('Executing')
try:
    for f in range(FRAMES):
        print('Generating frame {}/{}'.format(f + 1, FRAMES))
        zoom /= ZPF
        params[0] = zoom

        mandelbrot(drv.Out(dest), drv.In(params), block=BLOCK, grid=GRID)

        result = plt.get_cmap('PuBuGn')(dest)
        result *= 255

        Image.fromarray(result.astype('uint8'), 'RGBA').save('tmp/frame{}.png'.format(f + 1))
except KeyboardInterrupt:
    print('Render interrupted')
    p = 'tmp/frame{}.png'.format(f + 1)
    if os.path.exists(p):
        os.remove(p)

os.system('ffmpeg -y -r {fps} -f image2 -s {width}x{height} -i tmp/frame%d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p result.mp4'.format(width=WIDTH, height=HEIGHT, fps=FPS))
shutil.rmtree('tmp')