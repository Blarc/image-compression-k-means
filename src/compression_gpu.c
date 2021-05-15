#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include <omp.h>
#include <CL/cl.h>

#include "image_io.h"
#include "compression.h"

#define MAX_SOURCE_SIZE	(16384)
#define BINS 256
#define WORKGROUP_SIZE  (256)

void initialise_centers(byte_t *data, double *centers, int n_pixels, int n_channels, int n_clusters);

void kmeans_compression_gpu(byte_t *data, int width, int height, int n_channels, int n_clusters, int max_iterations) {

    int n_pixels = width * height;
    int *labels = malloc(n_pixels * sizeof(int));
    double *centers = malloc(n_clusters * n_channels * sizeof(double));
    double *distances = malloc(n_pixels * sizeof(double));
    int changed = 0;

    initialise_centers(data, centers, n_pixels, n_channels, n_clusters);

    printf("[+] Reading the kernel...\n");
    fflush(stdout);
    FILE *kernel_fp = fopen("kernel_gpu.cl", "r");
    if (!kernel_fp)
    {
        fprintf(stderr, "\t [-] Error loading the kernel\n");
        exit(1);
    }

    char *source_str = (char*)malloc(MAX_SOURCE_SIZE);
    size_t source_size = fread(source_str, 1, MAX_SOURCE_SIZE, kernel_fp);
    source_str[source_size] = '\0';

    fclose(kernel_fp);

    // Get platforms
    cl_uint num_platforms;
    cl_int clStatus = clGetPlatformIDs(0, NULL, &num_platforms);
    cl_platform_id *platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id)*num_platforms);
    clStatus = clGetPlatformIDs(num_platforms, platforms, NULL);

    //Get platform devices
    cl_uint num_devices;
    clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
    num_devices = 1; // limit to one device
    cl_device_id *devices = (cl_device_id *)malloc(sizeof(cl_device_id)*num_devices);
    clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);

    // Context
    cl_context context = clCreateContext(NULL, num_devices, devices, NULL, NULL, &clStatus);

    // Command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, devices[0], 0, &clStatus);

    // Create and build a program
    printf("[+] Creating and building the program: ");
    fflush(stdout);
    cl_program program = clCreateProgramWithSource(context,	1, (const char **)&source_str, NULL, &clStatus);
    clStatus = clBuildProgram(program, 1, devices, NULL, NULL, NULL);
    printf("%s\n", clStatus);
    fflush(stdout);

    // Log
    size_t build_log_len;
    clStatus = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_len);
    if (build_log_len > 2)
    {
        char *build_log =(char *)malloc(sizeof(char)*(build_log_len+1));
        clStatus = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG,
                                         build_log_len, build_log, NULL);
        printf("%s", build_log);
        free(build_log);
        return;
    }

    // Divide work
    size_t local_item_size = WORKGROUP_SIZE;
    size_t num_groups = ((n_pixels - 1) / local_item_size + 1);
    size_t global_item_size = num_groups * local_item_size;

    // Transfer data from host
    // TODO @jakobm Why CL_MEM_COPY_HOST_PTR
    cl_mem data_ = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, n_pixels * sizeof(byte_t), data, &clStatus);
    cl_mem centers_ = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, n_clusters * n_channels * sizeof(double), centers, &clStatus);
    cl_mem labels_ = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, n_pixels * sizeof(int), labels, &clStatus);
    cl_mem distances_ = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, n_pixels * sizeof(double), distances, &clStatus);
    cl_mem changed_ = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int), &changed, &clStatus);

    printf("[+] Creating kernels:\n");
    printf("\t[+] Creating assign_pixels kernel: ");
    fflush(stdout);
    cl_kernel assign_pixels_kernel = clCreateKernel(program, "assign_pixels", &clStatus);
    clStatus = clSetKernelArg(assign_pixels_kernel, 0, sizeof(cl_mem), (void *) &data_);
    clStatus |= clSetKernelArg(assign_pixels_kernel, 1, sizeof(cl_mem), (void *) &centers_);
    clStatus |= clSetKernelArg(assign_pixels_kernel, 2, sizeof(cl_mem), (void *) &labels_);
    clStatus |= clSetKernelArg(assign_pixels_kernel, 3, sizeof(cl_mem), (void *) &distances_);
    clStatus |= clSetKernelArg(assign_pixels_kernel, 4, sizeof(cl_mem), (void *) &changed_);
    clStatus |= clSetKernelArg(assign_pixels_kernel, 5, sizeof(cl_int), (void *) &n_pixels);
    clStatus |= clSetKernelArg(assign_pixels_kernel, 6, sizeof(cl_int), (void *) &n_channels);
    clStatus |= clSetKernelArg(assign_pixels_kernel, 7, sizeof(cl_int), (void *) &n_clusters);
    printf("%s\n", clStatus);
    fflush(stdout);

    printf("[+] Starting assign_pixels kernel: ");
    fflush(stdout);
    clStatus = clEnqueueNDRangeKernel(command_queue, assign_pixels_kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
    clFinish(command_queue);
    printf("%s\n", clStatus);
    fflush(stdout);
    
    printf("[+] Copying results to the host: ");
    fflush(stdout);
    clStatus = clEnqueueReadBuffer(command_queue, changed_, CL_TRUE, 0, sizeof(int), &changed, 0, NULL, NULL);
    printf("%s\n", clStatus);
    fflush(stdout);

    printf("\t[+] Changed: %d\n", changed);

    free(centers);
    free(labels);
    free(distances);

}

void initialise_centers(byte_t *data, double *centers, int n_pixels, int n_channels, int n_clusters)
{
    for (int cluster = 0; cluster < n_clusters; cluster++) {
        // Pick a random pixel
        int random_int = rand() % n_pixels;

        // Set the random pixel as one of the centers
        for (int channel = 0; channel < n_channels; channel++) {
            // Save picked pixel's channels
            centers[cluster * n_channels + channel] = data[random_int * n_channels + channel];
        }
    }
}
