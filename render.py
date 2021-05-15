import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from matplotlib import cm
from matplotlib import pyplot as plt
from PIL import Image
from pycuda.compiler import SourceModule
import shutil, os
import argparse
import random
import traceback

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
parser.add_argument('-nf', action='store_true', help='Don\'t save frames')
parser.add_argument('-ci', action='store_true', help='Contrast improvement')
parser.add_argument('-lp', action='store_true', help='Go to last position')
parser.add_argument('-p', nargs='?', action='store', default=4, type=int, help='Precision')

args = parser.parse_args()

INVERT = args.i
BLOCK = (args.bw, args.bh, 1)
GRID = (args.gw, args.gh)
FRAMES = args.frames
ZPF = args.z
INIT_POS = (args.x, args.y)
ITERATIONS = args.l
FPS = args.fps
OUTPUT = args.output
CM = args.cm
NO_FRAMES = args.nf
LAST_POS = args.lp
CONTRAST_IMPROVEMENT = args.ci
PRECISION = args.p

WIDTH = GRID[0] * BLOCK[0]
HEIGHT = GRID[1] * BLOCK[1]

if os.path.exists('tmp'):
    shutil.rmtree('tmp')
os.mkdir('tmp')

if LAST_POS and os.path.exists('last.txt'):
    with open('last.txt') as file:
        data = [float(x) for x in file.read().split(';')]
        INIT_POS = (data[0], data[1])

print('Initializing')
module = SourceModule('#define PRECISION {}\n'.format(PRECISION) + read('mandelbrot.cpp'), include_dirs=[os.path.join(os.getcwd(), 'include')], no_extern_c=True)
mandelbrot = module.get_function("mandelbrot")

print('Allocating buffer')
dest = np.zeros((HEIGHT, WIDTH), dtype=np.float32)

params = np.array([0, 0, 0, ITERATIONS], dtype=np.float32)

colormap = plt.get_cmap(CM)
zoom = 1
X = INIT_POS[0]
Y = INIT_POS[1]
print('Executing')
try:
    success = True
    for f in range(FRAMES):
        print('Generating frame {}/{}'.format(f + 1, FRAMES))
        zoom /= ZPF
        params[0] = zoom
        params[1] = X
        params[2] = Y

        mandelbrot(drv.Out(dest), drv.In(params), block=BLOCK, grid=GRID)

        if CONTRAST_IMPROVEMENT:
            dest -= np.amin(dest)
            dest /= np.amax(dest)
        
        if not NO_FRAMES:
            if INVERT:
                dest = 1 - dest
            result = colormap(dest)
            result *= 255

            Image.fromarray(result.astype('uint8'), 'RGBA').save('tmp/frame{}.png'.format(f + 1))
except KeyboardInterrupt:
    print('Render interrupted')
    p = 'tmp/frame{}.png'.format(f + 1)
    if os.path.exists(p):
        os.remove(p)



with open('last.txt', 'w') as file:
    file.write(';'.join(str(a) for a in [X, Y, zoom]))

if not NO_FRAMES and OUTPUT.lower() != 'null':
    os.system('ffmpeg -y -r {fps} -f image2 -s {width}x{height} -i tmp/frame%d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {output}'.format(width=WIDTH, height=HEIGHT, fps=FPS, output=OUTPUT))