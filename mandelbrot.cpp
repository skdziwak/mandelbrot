#include "multi_prec.h"
#include "matrix_utils.h"

typedef multi_prec<PRECISION> number;

class Complex {
public:
number r, i;
    __device__ Complex(number r, number i) : r(r), i(i) {
        
    }
    __device__ Complex operator*(const Complex& other) {
        return Complex(this->r * other.r - this->i * other.i, this->r * other.i + this->i * other.r);
    }
    __device__ Complex operator+(const Complex& other) {
        return Complex(this->r + other.r, this->i + other.i);
    }
    __device__ void operator=(const Complex& other) {
        this->r = other.r;
        this->i = other.i;
    }
    __device__ number dist_squared() {
        return this->i * this->i + this->r * this->r;
    }
};

extern "C" {

    __device__ int stability(number xc, number yc, int iterations) {
        Complex c(xc, yc);
        Complex p(0.0, 0.0);
        for (int i = 0; i < iterations; i++) {
            p = p * p + c;
            if (p.dist_squared() > 20000000000.0) return i;
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

    __global__ void filter(float *dest, float *in) {
        const int x = getX();
        const int y = getY();
        const int i = getIndex2D(x, y);
        if (x == 0 || x == getWidth()-1 || y == 0 || y == getHeight()-1) {
            dest[i] = 0.0; 
            return;
        }

        int j = 0;
        float score = 0.0;
        for (int a = -1 ; a <= 1 ; a+=2) {
            for (int b = -1 ; b <= 1 ; b+=2) {
                j = getIndex2D(x + a, y + b);
                score += in[j] * (a + b);
            }
        }
        dest[i] = abs(score / 6.0);
    }

}