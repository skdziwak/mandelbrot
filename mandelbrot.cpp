#include "multi_prec.h"
#include "matrix_utils.h"

typedef multi_prec<4> number;

extern "C" {

    __device__ int stability(number xc, number yc, int iterations) {
        number x(0.0);
        number y(0.0);
        for (int i = 0; i < iterations; i++) {
            number a = x;
            number b = y;
            x = a * a - b * b + xc;
            y = 2.0 * a * b + yc;
            if (x * x + y * y > 10000000.0) return i;
        }
        return -1;
    }

    __global__ void mandelbrot(float *dest, float *params) {
        const int x = getX();
        const int y = getY();
        const int i = getIndex2D(x, y);
        const number fx = (x / number(gridDim.x * blockDim.x) - 0.5) * 2;
        const number fy = (y / number(gridDim.y * blockDim.y) - 0.5) * 2;
        const number zoom(params[0]);
        const number sx(params[1]);
        const number sy (params[2]);
        const float iterations(params[3]);

        int s = stability(fx * zoom + sx, fy * zoom + sy, int(iterations));
        if (s == -1) {
            dest[i] = 1;
        } else {
            dest[i] = float(s) / iterations;
        }
    }

}