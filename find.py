import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from matplotlib import pyplot as plt
from PIL import Image
from pycuda.compiler import SourceModule
import random, os, time

BLOCK = (16, 16, 1)
GRID = (4, 4)
ITERATIONS = 128
WIDTH = GRID[0] * BLOCK[0]
HEIGHT = GRID[1] * BLOCK[1]

def read(path):
    with open(path, encoding='utf-8') as f:
        return f.read()

module = SourceModule(read('mandelbrot.cpp'), include_dirs=[os.path.join(os.getcwd(), 'include')], no_extern_c=True)
mandelbrot = module.get_function("mandelbrot")
mfilter = module.get_function("filter")

matrix = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
matrix2 = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
matrix3 = np.zeros((64 * 16, 64 * 16), dtype=np.float32)

def render(x, y, zoom, path):
    global matrix3
    params = np.array([zoom, x, y, 500], dtype=np.float32)
    mandelbrot(drv.Out(matrix3), drv.In(params), grid=(64, 64), block=(16, 16, 1))
    img = plt.get_cmap('turbo')(matrix3)
    img *= 255
    Image.fromarray(img.astype('uint8'), 'RGBA').save(path)

def calculate(x, y, zoom):
    global WIDTH, HEIGHT, BLOCK, GRID, ITERATIONS, matrix, matrix2
    params = np.array([zoom, x, y, ITERATIONS], dtype=np.float32)

    mandelbrot(drv.Out(matrix), drv.In(params), grid=GRID, block=BLOCK)
    mfilter(drv.Out(matrix2), drv.In(matrix), grid=GRID, block=BLOCK)
    
    # plt.imshow(matrix2)
    # plt.show()

    m = np.amax(matrix2)
    matrix2 /= m
    points = np.where(matrix2 >= .98)
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
        print(steps)
        minstep = steps
    if steps == 0:
        return points, zoom
    points.sort(key=lambda x: random.random())
    for p in points:
        pts = calculate(p[0], p[1], zoom)
        out = find(pts, zoom / 1.2, steps-1)
        if not out is None:
            return out
    return None


points, zoom = find(steps=50)
print('Saving random point')

with open('last.txt', 'w') as file:
    file.write(';'.join(str(a) for a in [points[0][0], points[0][1], zoom]))