__device__ int stability(double xc, float yc, int iterations) {
    double x = 0;
    double y = 0;
    for (int i = 0; i < iterations; i++) {
        double a = x;
		double b = y;
        x = a * a - b * b + xc;
        y = 2.0 * a * b + yc;
        if (x * x + y * y > 10000) return i;
    }
    return -1;
}

__global__ void mandelbrot(float *dest, float *params) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int i = x + y * gridDim.x * blockDim.x;
    const double fx = (x / double(gridDim.x * blockDim.x) - 0.5) * 2;
    const double fy = (y / double(gridDim.y * blockDim.y) - 0.5) * 2;
    const float zoom = params[0];
    const float sx = params[1];
    const float sy = params[2];
    const float iterations = params[3];

    int s = stability(fx * zoom + sx, fy * zoom + sy, int(iterations));
    if (s == -1) {
        dest[i] = 1;
    } else {
        dest[i] = float(s) / iterations;
    }
}