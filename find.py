import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from matplotlib import pyplot as plt
from PIL import Image
from pycuda.compiler import SourceModule
import random, os, time
import argparse

parser = argparse.ArgumentParser(description='Mandelbrot Set conntrasting points finder.')
parser.add_argument('steps', type=int, help='Number of steps')
parser.add_argument('-bw', nargs='?', action='store', default=16, type=int, help='Block width')
parser.add_argument('-bh', nargs='?', action='store', default=16, type=int, help='Block height')
parser.add_argument('-gw', nargs='?', action='store', default=4, type=int, help='Grid width')
parser.add_argument('-gh', nargs='?', action='store', default=4, type=int, help='Grid height')
parser.add_argument('-r', action='store_true', help='Render results')
parser.add_argument('-z', nargs='?', action='store', default=1.02, type=float, help='Zoom speed')
parser.add_argument('-p', nargs='?', action='store', default=4, type=int, help='Precision')
parser.add_argument('-l', nargs='?', action='store', default=128, type=int, help='Iterations limit')

args = parser.parse_args()

BLOCK = (args.bw, args.bh, 1)
GRID = (args.gw, args.gh)
RENDER = args.r
WIDTH = GRID[0] * BLOCK[0]
HEIGHT = GRID[1] * BLOCK[1]
STEPS = args.steps
ZPF = args.z
PRECISION = args.p
ITERATIONS = args.l

def read(path):
    with open(path, encoding='utf-8') as f:
        return f.read()

module = SourceModule('#define PRECISION {}\n'.format(PRECISION) + read('mandelbrot.cpp'), include_dirs=[os.path.join(os.getcwd(), 'include')], no_extern_c=True)
mandelbrot = module.get_function("mandelbrot")
mfilter = module.get_function("filter")

matrix = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
matrix2 = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
matrix3 = np.zeros((HEIGHT, WIDTH), dtype=np.float32)

def render(x, y, zoom, path):
    global matrix3
    params = np.array([zoom, x, y, 800], dtype=np.float32)
    mandelbrot(drv.Out(matrix3), drv.In(params), grid=GRID, block=BLOCK)
    matrix3 = np.nan_to_num(matrix3)
    matrix3 -= np.amin(matrix3)
    matrix3 /= np.amax(matrix3)
    img = plt.get_cmap('turbo')(matrix3)
    img *= 255
    Image.fromarray(img.astype('uint8'), 'RGBA').save(path)

def calculate(x, y, zoom):
    global WIDTH, HEIGHT, BLOCK, GRID, ITERATIONS, matrix, matrix2
    params = np.array([zoom, x, y, ITERATIONS], dtype=np.float32)

    mandelbrot(drv.Out(matrix), drv.In(params), grid=GRID, block=BLOCK)
    mfilter(drv.Out(matrix2), drv.In(matrix), grid=GRID, block=BLOCK)

    matrix2 = np.nan_to_num(matrix2)
    matrix2 -= np.amin(matrix2)
    matrix2 /= np.amax(matrix2)
    points = np.where(matrix2 > 0.95)
    points = list(zip(points[0], points[1]))
    result = []
    for b, a in points:
        a = (a / WIDTH * 2 - 1) * zoom + x
        b = (b / HEIGHT * 2 - 1) * zoom + y
        result.append((a, b))
    return result

minstep = 2000

def find(points=[(0, 0)], zoom=2.0, steps=80):
    global minstep
    if steps < minstep:
        print('New min step: {}'.format(steps))
        minstep = steps
    if steps == 0:
        return points, zoom
    points.sort(key=lambda x: random.random())
    for p in points:
        pts = calculate(p[0], p[1], zoom)
        out = find(pts, zoom / ZPF, steps-1)
        if not out is None:
            return out
    return None


points, zoom = find(steps=STEPS)
if RENDER:
    z = 2
    for i in range(STEPS):
        print('Frame {}; Zoom {}'.format((i+1), z))
        render(points[0][0], points[0][1], z, 'tmp/find{}.png'.format(i + 1))
        z /= ZPF
print('Saving point')

with open('last.txt', 'w') as file:
    file.write(';'.join(str(a) for a in [points[0][0], points[0][1], zoom]))