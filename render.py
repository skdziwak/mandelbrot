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
parser.add_argument('-mc', nargs='?', action='store', default=0, type=float, help='Minimal contrast')
parser.add_argument('-cm', nargs='?', action='store', default='PuBuGn', type=str, help='Matplotlib colormap')
parser.add_argument('frames', type=int, help='Number of rendered frames')
parser.add_argument('output', type=str, help='Output file (mp4)')
parser.add_argument('-i', action='store_true', help='Invert colormap')
parser.add_argument('-nf', action='store_true', help='Don\'t save frames')
parser.add_argument('-r', action='store_true', help='Repeat if failed')
parser.add_argument('-ci', action='store_true', help='Contrast improvement')
parser.add_argument('-a', action='store_true', help='Automatic positioning')
parser.add_argument('-lp', action='store_true', help='Go to last position')
parser.add_argument('-ai', nargs='?', action='store', default=80, type=int, help='Automatic positioning interval')
parser.add_argument('-ar', nargs='?', action='store', default=0.1, type=float, help='Automatic positioning rate')

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
AUTO = args.a
AUTO_INTERVAL = args.ai
AUTO_RATE = args.ar
LAST_POS = args.lp
MINIMAL_CONTRAST = args.mc
REPEAT = args.r
CONTRAST_IMPROVEMENT = args.ci

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
mod = SourceModule(read('mandelbrot.cpp'), include_dirs=[os.path.join(os.getcwd(), 'include')], no_extern_c=True)
mandelbrot = mod.get_function("mandelbrot")

print('Allocating buffer')
dest = np.zeros((HEIGHT, WIDTH), dtype=np.float32)

params = np.array([0, 0, 0, ITERATIONS], dtype=np.float32)

while True:
    target = (0, 0)
    auto_counter = 0
    zoom = 1.5
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
            
            if MINIMAL_CONTRAST != 0 and np.amax(dest) - np.amin(dest) < MINIMAL_CONTRAST:
                print('Contrast condition failed.')
                success = False
                break

            if AUTO:
                if auto_counter == 0:
                    edges = np.where(np.abs(dest - 0.5) < 0.35)
                    points = list(zip(edges[0], edges[1]))
                    point = random.choice(points)
                    x = point[1]
                    y = point[0]
                    x = (x / WIDTH - 0.5) * 2 * zoom
                    y = (y / HEIGHT - 0.5) * 2 * zoom
                    target = (X + x, Y + y)
                    auto_counter = AUTO_INTERVAL
                else:
                    auto_counter -= 1
                X = (X * (1 - AUTO_RATE) + target[0] * AUTO_RATE)
                Y = (Y * (1 - AUTO_RATE) + target[1] * AUTO_RATE)

            if not NO_FRAMES:
                if CONTRAST_IMPROVEMENT:
                    dest = np.tanh(dest * 6 - 3) / 2 + 0.5
                if INVERT:
                    result = 1 - result
                result = plt.get_cmap(CM)(dest)
                result *= 255

                Image.fromarray(result.astype('uint8'), 'RGBA').save('tmp/frame{}.png'.format(f + 1))
        if success:
            break
    except KeyboardInterrupt:
        print('Render interrupted')
        p = 'tmp/frame{}.png'.format(f + 1)
        if os.path.exists(p):
            os.remove(p)
        break
    except IndexError:
        print('Empty sequence.')
    if REPEAT:
        print('Repeating.')
    else:
        print('Halting.')
        break


with open('last.txt', 'w') as file:
    file.write(';'.join(str(a) for a in [X, Y, zoom]))

if not NO_FRAMES and OUTPUT.lower() != 'null':
    os.system('ffmpeg -y -r {fps} -f image2 -s {width}x{height} -i tmp/frame%d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {output}'.format(width=WIDTH, height=HEIGHT, fps=FPS, output=OUTPUT))