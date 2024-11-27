//
// Created by albakrih on 26/11/24.
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

int main() {
    int N = 1000; // Size of the vectors
    size_t size = N * sizeof(float);

    // Allocate memory on the host
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    // Initialize vectors A and B with some values
    for (int i = 0; i < N; i++) {
        h_A[i] = i + 1.0f;  // Example: A = [1, 2, 3, ..., N]
        h_B[i] = (i + 1) * 2.0f;  // Example: B = [2, 4, 6, ..., 2*N]
    }

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
    vectorMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Check for kernel launch errors
    cudaDeviceSynchronize();

    // Copy the result from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Output the result (for demonstration, print the first 10 elements)
    for (int i = 0; i < 10; i++) {
        std::cout << "C[" << i << "] = " << h_C[i] << std::endl;
    }

    // Free memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
