//
// Created by Kazem on 2024-09-25.
//

#ifdef __OPENCL__
#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/opencl.h>
#endif
#include "err_code.h"
#endif

// if cuda is available
#ifdef __CUDA__
#include <cuda_runtime.h>
#endif

#include <iostream>
#include <benchmark/benchmark.h>
#include <chrono>
#include <omp.h>
#include "vec_mul.h"


int num_teach_threads = 8;

static void BM_VECMUL(benchmark::State &state,
                      void (*vecImpl1)(std::vector<float> a, std::vector<float> b, std::vector<float> &c)) {
    int m = state.range(0);
    std::vector<float> A(m);
    std::vector<float> B(m);
    std::vector<float> C(m);
    for (int i = 0; i < m; ++i) {
        A[i] = 1.0;
    }
    for (int i = 0; i < m; ++i) {
        B[i] = 1.0;
    }

    for (auto _: state) {
        vecImpl1(A, B, C);
    }
}


static void BM_VECMUL_PARALLEL(benchmark::State &state,
                               void (*vecImpl1)(std::vector<float> a, std::vector<float> b, std::vector<float> &c, int num_threads)) {
    int m = state.range(0);
    int num_threads = state.range(1);
    std::vector<float> A(m);
    std::vector<float> B(m);
    std::vector<float> C(m);
    for (int i = 0; i < m; ++i) {
        A[i] = 1.0;
    }
    for (int i = 0; i < m; ++i) {
        B[i] = 1.0;
    }

    for (auto _: state) {
        auto begin = omp_get_wtime();
        vecImpl1(A, B, C, num_threads);
        auto elapsed = omp_get_wtime() - begin;
        state.SetIterationTime(elapsed);
    }
}

#ifdef __OPENCL__
const char *KernelSource = "\n" \
"__kernel void vadd(                                                 \n" \
"   __global float* a,                                                  \n" \
"   __global float* b,                                                  \n" \
"   __global float* c,                                                  \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count)                                                       \n" \
"       c[i] = a[i] + b[i];                                             \n" \
"}                                                                      \n" \
"\n";


static void BM_VECMUL_OPENCL(benchmark::State &state,
                               void (*vecImpl1)(std::vector<float> a, std::vector<float> b, std::vector<float> &c, int num_threads)) {
    int m = state.range(0);
    std::vector<float> A(m);
    std::vector<float> B(m);
    std::vector<float> C(m);
    for (int i = 0; i < m; ++i) {
        A[i] = 1.0;
    }
    for (int i = 0; i < m; ++i) {
        B[i] = 1.0;
    }
    // declare device containers
    // Initialize OpenCL
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;

    // Get the number of platforms
    cl_uint num_platforms;
    clGetPlatformIDs(0, NULL, &num_platforms);

    // Get the platforms
    cl_platform_id* platforms = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
    clGetPlatformIDs(num_platforms, platforms,
                     NULL);

    // Find a GPU platform
    cl_bool found_gpu_platform = CL_FALSE;
    for (cl_uint i = 0; i < num_platforms; ++i) {
        cl_uint num_devices;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);

        if (num_devices > 0) {
            clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
            found_gpu_platform = CL_TRUE;
            break;
        }
    }

    if (!found_gpu_platform) {
        std::cerr << "No GPU device found." << std::endl;
    }

    // Get device information
    char device_name[1024];
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);

    int err;
// Create a compute context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    checkError(err, "Creating context");
    // create cl_queue_properties
    cl_queue_properties *properties = new cl_queue_properties[3];
    properties[0] = CL_QUEUE_PROPERTIES;
    properties[1] = CL_QUEUE_PROFILING_ENABLE;
    properties[2] = 0;

    // Create a command queue
    command_queue = clCreateCommandQueueWithProperties(context, device_id,
                                                       properties, &err);
    checkError(err, "Creating command queue");


    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
    checkError(err, "Creating program");

    // Build the program
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n%s\n", err_code(err));
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
    }

    // Create the compute kernel from the program
    kernel = clCreateKernel(program, "vadd", &err);
    checkError(err, "Creating kernel");

    // Create device buffers
    cl_mem buffer_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, m * sizeof(float), A.data(), NULL);
    cl_mem buffer_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, m * sizeof(float), B.data(), NULL);
    cl_mem buffer_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, m * sizeof(float), NULL, NULL);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&buffer_a);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&buffer_b);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&buffer_c);
    clSetKernelArg(kernel, 3, sizeof(int), (void *)&m);

    size_t global_work_size = m;
    // add gpu name to the log
    state.SetLabel(device_name);
    for (auto _: state) {
        // Execute kernel
        cl_event event;
        cl_ulong time_start = 0;
        cl_ulong time_end = 0;

        clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);

        // Wait for kernel to finish
        clFinish(command_queue);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
        cl_ulong elapsed = time_end - time_start;
        // Read results back to host
        clEnqueueReadBuffer(command_queue, buffer_c, CL_TRUE, 0, m * sizeof(float), C.data(), 0, NULL, NULL);

        // Print first 10 elements of the result
//        for (int i = 0; i < 10; ++i) {
//            std::cout << C[i] << " ";
//        }
        state.SetIterationTime(elapsed);
    }

    // Clean up
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
    clReleaseMemObject(buffer_a);
    clReleaseMemObject(buffer_b);
    clReleaseMemObject(buffer_c);
    free(platforms);

}
#endif

#ifdef __CUDA__
// placeholder for the kernel
extern float vectorMultiplyWrapper(float *a, float *b, float *c, int n);


static void BM_VECMUL_CUDA(benchmark::State &state,
                             void (*vecImpl1)(std::vector<float> a, std::vector<float> b, std::vector<float> &c, int num_threads)) {
    int m = state.range(0);
    std::vector<float> A(m);
    std::vector<float> B(m);
    std::vector<float> C(m);
    size_t size = m * sizeof(float);
    for (int i = 0; i < m; ++i) {
        A[i] = 1.0;
    }
    for (int i = 0; i < m; ++i) {
        B[i] = 1.0;
    }

    const char *device_name = "GPU";

    // add gpu name to the log
    state.SetLabel(device_name);
    for (auto _: state) {
        float elapsed = vectorMultiplyWrapper(A.data(), B.data(), C.data(), m);
        state.SetIterationTime(elapsed);
    }

}
#endif


BENCHMARK_CAPTURE(BM_VECMUL, baseline_vec_mul, swiftware::hpp::vec_mul)->Ranges({{2<<18, 2<<20}});

BENCHMARK_CAPTURE(BM_VECMUL_PARALLEL, parallel_vec_mul, swiftware::hpp::vec_mul_parallel)->Ranges({{2<<18, 2<<20}, {4, 8}})->UseManualTime();

#ifdef __OPENCL__
BENCHMARK_CAPTURE(BM_VECMUL_OPENCL, opencl_vec_mul, swiftware::hpp::vec_mul_parallel)->Ranges({{2<<18, 2<<20}})->UseManualTime()->Iterations(100);
#endif

#ifdef __CUDA__
BENCHMARK_CAPTURE(BM_VECMUL_CUDA, cuda_vec_mul, swiftware::hpp::vec_mul_parallel)->Ranges({{2<<18, 2<<20}})->UseManualTime()->Iterations(100);
#endif

//
BENCHMARK_MAIN();

