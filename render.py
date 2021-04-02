import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from matplotlib import cm
from matplotlib import pyplot as plt
from PIL import Image
from pycuda.compiler import SourceModule
import shutil, os
import argparse

def read(path):
    with open(path, encoding='utf-8') as f:
        return f.read()

parser = argparse.ArgumentParser(description='Mandelbrot Set zoom animation generator.')
parser.add_argument('-bw', nargs='?', action='store', default=32, type=int, help='Block width')
parser.add_argument('-bh', nargs='?', action='store', default=32, type=int, help='Block height')
parser.add_argument('-gw', nargs='?', action='store', default=48, type=int, help='Grid width')
parser.add_argument('-gh', nargs='?', action='store', default=48, type=int, help='Grid height')
parser.add_argument('-l', nargs='?', action='store', default=500, type=int, help='Iterations limit')
parser.add_argument('-x', nargs='?', action='store', default=0, type=float, help='X offset')
parser.add_argument('-y', nargs='?', action='store', default=0, type=float, help='Y offset')
parser.add_argument('-fps', nargs='?', action='store', default=30, type=int, help='Y offset')
parser.add_argument('-z', nargs='?', action='store', default=1.02, type=float, help='Zoom speed')
parser.add_argument('-cm', nargs='?', action='store', default='PuBuGn', type=str, help='Matplotlib colormap')
parser.add_argument('frames', type=int, help='Number of rendered frames')
parser.add_argument('output', type=str, help='Output file (mp4)')
parser.add_argument('-i', action='store_true', help='Invert colormap')

args = parser.parse_args()

INVERT = args.i
BLOCK = (args.bw, args.bh, 1)
GRID = (args.gw, args.gh)
FRAMES = args.frames
ZPF = args.z
X = args.x
Y = args.y
ITERATIONS = args.l
FPS = args.fps
OUTPUT = args.output
CM = args.cm

WIDTH = GRID[0] * BLOCK[0]
HEIGHT = GRID[1] * BLOCK[1]

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

        result = plt.get_cmap(CM)(dest)
        result *= 255
        if INVERT:
            result = 1 - result

        Image.fromarray(result.astype('uint8'), 'RGBA').save('tmp/frame{}.png'.format(f + 1))
except KeyboardInterrupt:
    print('Render interrupted')
    p = 'tmp/frame{}.png'.format(f + 1)
    if os.path.exists(p):
        os.remove(p)

os.system('ffmpeg -y -r {fps} -f image2 -s {width}x{height} -i tmp/frame%d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {output}'.format(width=WIDTH, height=HEIGHT, fps=FPS, output=OUTPUT))
shutil.rmtree('tmp')