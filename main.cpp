#include <CL/cl.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

int main() {

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
        return 1;
    }

    // Get device information
    char device_name[1024];
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    std::cout << "GPU Device: " << device_name << std::endl;

    // Generate random vectors on the host
    int N = 10000;
    float *a = (float*)malloc(N * sizeof(float));
    float *b = (float*)malloc(N * sizeof(float));
    srand(time(NULL));
    for (int i = 0; i < N; ++i) {
        a[i] = rand() / (float)RAND_MAX;
        b[i] = rand() / (float)RAND_MAX;
    }


    for (int i = 0; i < 10; ++i) {
        std::cout << a[i] << " ";
    }
    std::cout << std::endl;

    for (int i = 0; i < 10; ++i) {
        std::cout << b[i] << " ";
    }
    std::cout << std::endl;

    // Create device buffers
    cl_mem buffer_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * sizeof(float), a, NULL);
    cl_mem buffer_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * sizeof(float), b, NULL);
    cl_mem buffer_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * sizeof(float), NULL, NULL);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&buffer_a);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&buffer_b);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&buffer_c);
    clSetKernelArg(kernel, 3, sizeof(int), (void *)&N);

    // Execute kernel
    size_t global_work_size = N;
    clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);

    // Read results back to host
    clEnqueueReadBuffer(command_queue, buffer_c, CL_TRUE, 0, N * sizeof(float), a, 0, NULL, NULL);

    // Print first 10 elements of the result
    for (int i = 0; i < 10; ++i) {
        std::cout << a[i] << " ";
    }
    std::cout << std::endl;

    // Clean up
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
    free(a);
    free(b);

    return 0;
}