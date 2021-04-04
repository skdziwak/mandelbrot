#ifndef _MATRIX_UTILS_H
#define _MATRIX_UTILS_H

extern "C" {
    __device__ int getX() {
        return blockIdx.x * blockDim.x + threadIdx.x;
    }

    __device__ int getY() {
        return blockIdx.y * blockDim.y + threadIdx.y;
    }

    __device__ int getIndex2D(int x, int y) {
        return x + y * gridDim.x * blockDim.x;
    }

    __device__ int getWidth() {
        return gridDim.x * blockDim.x;
    }

    __device__ int getHeight() {
        return gridDim.y * blockDim.y;
    }
}

#endif