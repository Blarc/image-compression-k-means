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
#define WORKGROUP_SIZE  (1024)

void initialise_centers(byte_t *data, long *centers, int n_pixels, int n_channels, int n_clusters);

void kmeans_compression_gpu(byte_t *data, int width, int height, int n_channels, int n_clusters, int max_iterations) {

    double start_time = 0;
    int n_pixels = width * height;
    int *labels = (int*) malloc(n_pixels * sizeof(int));
    long *centers = (long*) malloc(n_clusters * n_channels * sizeof(long));
    double *distances = (double*) malloc(n_pixels * sizeof(double));
    int *counts = (int*) malloc(n_clusters * sizeof(int));
    int changed = 0;

    initialise_centers(data, centers, n_pixels, n_channels, n_clusters);

    // printf("[+] Reading the kernel...\n");
    // fflush(stdout);
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
    // printf("[+] Creating and building the program: ");
    // fflush(stdout);
    cl_program program = clCreateProgramWithSource(context,	1, (const char **)&source_str, NULL, &clStatus);
    clStatus = clBuildProgram(program, 1, devices, NULL, NULL, NULL);
    // printf("%s\n", clStatus);
    // fflush(stdout);

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
    start_time = omp_get_wtime();
    cl_mem data_ = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, n_pixels * n_channels * sizeof(byte_t), data, &clStatus);
    cl_mem centers_ = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR , n_clusters * n_channels * sizeof(long), centers, &clStatus);
    cl_mem labels_ = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR , n_pixels * sizeof(int), labels, &clStatus);
    cl_mem distances_ = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR , n_pixels * sizeof(double), distances, &clStatus);
    cl_mem changed_ = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR , sizeof(int), &changed, &clStatus);
    cl_mem counts_ = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR , n_clusters * sizeof(int), counts, &clStatus);

    // printf("[+] Creating kernels:\n");
    // printf("\t[+] Creating assign_pixels kernel: ");
    // fflush(stdout);
    cl_kernel assign_pixels_kernel = clCreateKernel(program, "assign_pixels", &clStatus);
    clStatus = clSetKernelArg(assign_pixels_kernel, 0, sizeof(cl_mem), (void *) &data_);
    clStatus |= clSetKernelArg(assign_pixels_kernel, 1, sizeof(cl_mem), (void *) &centers_);
    clStatus |= clSetKernelArg(assign_pixels_kernel, 2, sizeof(cl_mem), (void *) &labels_);
    clStatus |= clSetKernelArg(assign_pixels_kernel, 3, sizeof(cl_mem), (void *) &distances_);
    clStatus |= clSetKernelArg(assign_pixels_kernel, 4, sizeof(cl_mem), (void *) &changed_);
    clStatus |= clSetKernelArg(assign_pixels_kernel, 5, sizeof(cl_int), (void *) &n_pixels);
    clStatus |= clSetKernelArg(assign_pixels_kernel, 6, sizeof(cl_int), (void *) &n_channels);
    clStatus |= clSetKernelArg(assign_pixels_kernel, 7, sizeof(cl_int), (void *) &n_clusters);
    // printf("%s\n", clStatus);
    // fflush(stdout);

    // printf("\t[+] Creating partial_sum_centers kernel: ");
    // fflush(stdout);
    cl_kernel partial_sum_centers_kernel = clCreateKernel(program, "partial_sum_centers", &clStatus);
    clStatus = clSetKernelArg(partial_sum_centers_kernel, 0, sizeof(cl_mem), (void *) &data_);
    clStatus |= clSetKernelArg(partial_sum_centers_kernel, 1, sizeof(cl_mem), (void *) &centers_);
    clStatus |= clSetKernelArg(partial_sum_centers_kernel, 2, sizeof(cl_mem), (void *) &labels_);
    clStatus |= clSetKernelArg(partial_sum_centers_kernel, 3, sizeof(cl_mem), (void *) &distances_);
    clStatus |= clSetKernelArg(partial_sum_centers_kernel, 4, sizeof(cl_int), (void *) &n_pixels);
    clStatus |= clSetKernelArg(partial_sum_centers_kernel, 5, sizeof(cl_int), (void *) &n_channels);
    clStatus |= clSetKernelArg(partial_sum_centers_kernel, 6, sizeof(cl_int), (void *) &n_clusters);
	clStatus |= clSetKernelArg(partial_sum_centers_kernel, 7, sizeof(cl_mem), (void *) &counts_);
    // printf("%s\n", clStatus);
    // fflush(stdout);

    // Divide work - partial_sum_centers_new
    size_t local_item_size_partial_sum_centers = WORKGROUP_SIZE - (WORKGROUP_SIZE % ((n_channels + 1) * n_clusters));
    size_t num_groups_partial_sum_centers = ((n_pixels * (n_channels + 1) * n_clusters - 1) / local_item_size_partial_sum_centers + 1);
    size_t global_item_size_partial_sum_centers = num_groups_partial_sum_centers * local_item_size_partial_sum_centers;

    // printf("\t[+] Creating partial_sum_centers_new kernel: ");
    // fflush(stdout);
    cl_kernel partial_sum_centers_kernel_new = clCreateKernel(program, "partial_sum_centers_new", &clStatus);
    clStatus = clSetKernelArg(partial_sum_centers_kernel_new, 0, sizeof(cl_mem), (void *) &data_);
    clStatus |= clSetKernelArg(partial_sum_centers_kernel_new, 1, sizeof(cl_mem), (void *) &centers_);
    clStatus |= clSetKernelArg(partial_sum_centers_kernel_new, 2, sizeof(cl_mem), (void *) &labels_);
    clStatus |= clSetKernelArg(partial_sum_centers_kernel_new, 3, sizeof(cl_mem), (void *) &distances_);
    clStatus |= clSetKernelArg(partial_sum_centers_kernel_new, 4, sizeof(cl_int), (void *) &n_pixels);
    clStatus |= clSetKernelArg(partial_sum_centers_kernel_new, 5, sizeof(cl_int), (void *) &n_channels);
    clStatus |= clSetKernelArg(partial_sum_centers_kernel_new, 6, sizeof(cl_int), (void *) &n_clusters);
	clStatus |= clSetKernelArg(partial_sum_centers_kernel_new, 7, sizeof(cl_mem), (void *) &counts_);
    // LOCAL
	clStatus |= clSetKernelArg(partial_sum_centers_kernel_new, 8, local_item_size_partial_sum_centers * sizeof(int), NULL);
    // printf("%s\n", clStatus);
    // fflush(stdout);

    // printf("\t[+] Creating centers_mean kernel: ");
    // fflush(stdout);
    cl_kernel centers_mean_kernel = clCreateKernel(program, "centers_mean", &clStatus);
    clStatus = clSetKernelArg(centers_mean_kernel, 0, sizeof(cl_mem), (void *) &data_);
    clStatus |= clSetKernelArg(centers_mean_kernel, 1, sizeof(cl_mem), (void *) &centers_);
    clStatus |= clSetKernelArg(centers_mean_kernel, 2, sizeof(cl_mem), (void *) &distances_);
	clStatus |= clSetKernelArg(centers_mean_kernel, 3, sizeof(cl_mem), (void *) &counts_);
    clStatus |= clSetKernelArg(centers_mean_kernel, 4, sizeof(cl_int), (void *) &n_pixels);
    clStatus |= clSetKernelArg(centers_mean_kernel, 5, sizeof(cl_int), (void *) &n_channels);
    clStatus |= clSetKernelArg(centers_mean_kernel, 6, sizeof(cl_int), (void *) &n_clusters);
    // LOCAL
    // clStatus |= clSetKernelArg(centers_mean_kernel, 7, local_item_size * sizeof(double), NULL);
    // clStatus |= clSetKernelArg(centers_mean_kernel, 8, local_item_size * sizeof(int), NULL);
    // printf("%s\n", clStatus);
    // fflush(stdout);

    // printf("\t[+] Creating centers_mean_new kernel: ");
    // fflush(stdout);
    // cl_kernel centers_mean_kernel_new = clCreateKernel(program, "centers_mean_new", &clStatus);
    // clStatus = clSetKernelArg(centers_mean_kernel_new, 0, sizeof(cl_mem), (void *) &data_);
    // clStatus |= clSetKernelArg(centers_mean_kernel_new, 1, sizeof(cl_mem), (void *) &centers_);
    // clStatus |= clSetKernelArg(centers_mean_kernel_new, 2, sizeof(cl_mem), (void *) &distances_);
	// clStatus |= clSetKernelArg(centers_mean_kernel_new, 3, sizeof(cl_mem), (void *) &counts_);
    // clStatus |= clSetKernelArg(centers_mean_kernel_new, 4, sizeof(cl_int), (void *) &n_pixels);
    // clStatus |= clSetKernelArg(centers_mean_kernel_new, 5, sizeof(cl_int), (void *) &n_channels);
    // clStatus |= clSetKernelArg(centers_mean_kernel_new, 6, sizeof(cl_int), (void *) &n_clusters);
    // printf("%s\n", clStatus);
    // fflush(stdout);

    // printf("\t[+] Creating update_data kernel: ");
    // fflush(stdout);
    cl_kernel update_data_kernel = clCreateKernel(program, "update_data", &clStatus);
    clStatus = clSetKernelArg(update_data_kernel, 0, sizeof(cl_mem), (void *) &data_);
    clStatus |= clSetKernelArg(update_data_kernel, 1, sizeof(cl_mem), (void *) &centers_);
    clStatus |= clSetKernelArg(update_data_kernel, 2, sizeof(cl_mem), (void *) &labels_);
    clStatus |= clSetKernelArg(update_data_kernel, 3, sizeof(cl_int), (void *) &n_pixels);
    clStatus |= clSetKernelArg(update_data_kernel, 4, sizeof(cl_int), (void *) &n_channels);
    // printf("%s\n", clStatus);
    // fflush(stdout);

    double transfer_data_time = omp_get_wtime() - start_time;

    double assign_pixels_time = 0;
    double read_changed_time = 0;
    double partial_sum_centers_time = 0;
    double partial_sum_centers_time_new = 0;
    double centers_mean_time = 0;
    double update_data_time = 0;
    double read_updated_data_time = 0;

    for (int i = 0; i < max_iterations; i++) {

        // printf("### Loop %d ###\n", i);
        // printf("[+] Starting assign_pixels kernel: ");
        // fflush(stdout);
        start_time = omp_get_wtime();
        clStatus = clEnqueueNDRangeKernel(command_queue, assign_pixels_kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
        clStatus = clFinish(command_queue);
        assign_pixels_time += omp_get_wtime() - start_time;
        // printf("%s\n", clStatus);
        // fflush(stdout);
        
        // printf("[+] Copying changed to host: ");
        // fflush(stdout);
        start_time = omp_get_wtime();
        clStatus = clEnqueueReadBuffer(command_queue, changed_, CL_TRUE, 0, sizeof(int), &changed, 0, NULL, NULL);
        read_changed_time += omp_get_wtime() - start_time;
        // printf("%s\n", clStatus);
        // // printf("\t[+] Changed: %d\n", changed);
        // fflush(stdout);

        // if clusters haven't changed, they won't change in the next iteration as well, so just stop early
        if (!changed) {
            break;
        }

        // printf("[+] Starting partial_sum_centers kernel: ");
        // fflush(stdout);
        // start_time = omp_get_wtime();
        // clStatus = clEnqueueNDRangeKernel(command_queue, partial_sum_centers_kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
        // clStatus = clFinish(command_queue);
        // partial_sum_centers_time += omp_get_wtime() - start_time;
        // printf("%s\n", clStatus);
        // fflush(stdout);

        // printf("[+] Starting partial_sum_centers_new kernel: ");
        // fflush(stdout);
        start_time = omp_get_wtime();
        clStatus = clEnqueueNDRangeKernel(command_queue, partial_sum_centers_kernel_new, 1, NULL, &global_item_size_partial_sum_centers, &local_item_size_partial_sum_centers, 0, NULL, NULL);
        clStatus = clFinish(command_queue);
        partial_sum_centers_time_new += omp_get_wtime() - start_time;
        // printf("%s\n", clStatus);
        // fflush(stdout);

        // printf("[+] Starting centers_mean kernel: ");
        // fflush(stdout);
        start_time = omp_get_wtime();
        clStatus = clEnqueueNDRangeKernel(command_queue, centers_mean_kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
        clStatus = clFinish(command_queue);
        centers_mean_time += omp_get_wtime() - start_time;
        // printf("%s\n", clStatus);
        // fflush(stdout);
    }

    // printf("[+] Starting update_data kernel: ");
    // fflush(stdout);
    start_time = omp_get_wtime();
    clStatus = clEnqueueNDRangeKernel(command_queue, update_data_kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
    clStatus = clFinish(command_queue);
    update_data_time += omp_get_wtime() - start_time;
    // printf("%s\n", clStatus);
    // fflush(stdout);


    // printf("[+] Copying update data to host: ");
    // fflush(stdout);
    start_time = omp_get_wtime();
    clStatus = clEnqueueReadBuffer(command_queue, data_, CL_TRUE, 0, n_pixels * n_channels * sizeof(byte_t), data, 0, NULL, NULL);
    clStatus = clFinish(command_queue);
    read_updated_data_time = omp_get_wtime() - start_time;
    // printf("%s\n", clStatus);

    // printf("\t[+] Updated data: ");
    // for (int pixel = 0; pixel < 20; pixel++) {
    //     for (int channel = 0; channel < n_channels; channel++) {
    //         printf("%d ", data[pixel * n_channels + channel]);
    //     }
    //     printf("| ");
    // }
    // printf("\n");
    // fflush(stdout);

    printf("[+] Printing times: \n");
    printf("\t[+] transfer_data_time: %f\n", transfer_data_time);
    printf("\t[+] assign_pixels_time: %f\n", assign_pixels_time);
    printf("\t[+] read_changed_time: %f\n", read_changed_time);
    // printf("\t[+] partial_sum_centers_time: %f\n", partial_sum_centers_time);
    printf("\t[+] partial_sum_centers_time_new: %f\n", partial_sum_centers_time_new);
    printf("\t[+] centers_mean_time: %f\n", centers_mean_time);
    printf("\t[+] update_data_time: %f\n", update_data_time);
    printf("\t[+] read_updated_data_time: %f\n", read_updated_data_time);
    fflush(stdout);

    clStatus = clReleaseKernel(assign_pixels_kernel);
    clStatus = clReleaseKernel(partial_sum_centers_kernel);
    clStatus = clReleaseKernel(centers_mean_kernel);
    clStatus = clReleaseKernel(update_data_kernel);

    clStatus = clReleaseProgram(program);

    clStatus = clReleaseMemObject(data_);
    clStatus = clReleaseMemObject(centers_);
    clStatus = clReleaseMemObject(labels_);
    clStatus = clReleaseMemObject(distances_);
    clStatus = clReleaseMemObject(counts_);
    
    clStatus = clReleaseCommandQueue(command_queue);
    clStatus = clReleaseContext(context);

    free(devices);
    free(platforms);

    free(counts);
    free(centers);
    free(labels);
    free(distances);

}

void initialise_centers(byte_t *data, long *centers, int n_pixels, int n_channels, int n_clusters)
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
