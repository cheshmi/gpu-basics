//
// Created by kazem on 11/29/24.
//



#include <iostream>
#include <cuda_runtime.h>

// CUDA kernel function for element-wise vector multiplication
__global__ void vectorMultiply(float* A, float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        C[idx] = A[idx] * B[idx];  // Element-wise multiplication
    }
}

float vectorMultiplyWrapper(float* h_A, float* h_B, float* h_C, int N) {
    size_t size = N * sizeof(float);

    // Allocate memory on the device (GPU)
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy input vectors from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch the kernel with enough blocks and threads
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    // cuda event to measure time
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    vectorMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Wait for kernel to finish
    cudaDeviceSynchronize();
    cudaEventCreate(&stop);
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);

    // Copy the result from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Output the result (for demonstration, print the first 10 elements)
//    for (int i = 0; i < 10; i++) {
//        std::cout << "C[" << i << "] = " << h_C[i] << std::endl;
//    }

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return elapsed;
}
