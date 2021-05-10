#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

#include "image_io.h"
#include "compression.h"

#define DEFAULT_N_CLUSTERS 4
#define DEFAULT_MAX_ITERATIONS 150
#define DEFAULT_OUT_PATH "result.jpg"
#define DEFAULT_N_THREADS 2

double get_time();

int main(int argc, char **argv)
{
    char *in_path = NULL;
    char *out_path = DEFAULT_OUT_PATH;

    int n_clusters = DEFAULT_N_CLUSTERS;
    int max_iterations = DEFAULT_MAX_ITERATIONS;

    int seed = time(NULL);
    int n_threads = DEFAULT_N_THREADS;
    
    // Parse arguments and optional parameters
    char optchar;
    while ((optchar = getopt(argc, argv, "k:m:o:s:t:h")) != -1) {
        switch (optchar)
        {
        case 'k':
            n_clusters = strtol(optarg, NULL, 10);
            break;
        case 'm':
            max_iterations = strtol(optarg, NULL, 10);
            break;
        case 'o':
            out_path = optarg;
            break;
        case 's':
            seed = strtol(optarg, NULL, 10);
            break;
        case 't':
            n_threads = strtol(optarg, NULL, 10);
            break;
        case 'h':
        default:
            // TODO @blarc print_usage(argv[0])
            exit(EXIT_FAILURE);
            break;
        }
    }

    in_path = argv[optind];

    // Validate input parameters
    if (in_path == NULL) {
        // TODO @blarc print_usage(argv[0])
        fprintf(stderr, "INPUT ERROR: << Parameter 'in_path' not defined >> \n");
        exit(EXIT_FAILURE);
    }

    if (n_clusters < 2) {
        fprintf(stderr, "INPUT ERROR: << Invalid number of clusters >> \n");
        exit(EXIT_FAILURE);    
    }

    if (max_iterations < 1) {
        fprintf(stderr, "INPUT ERROR: << Invalid maximum number of iterations >> \n");
        exit(EXIT_FAILURE);    
    }

    if (n_threads < 2) {
        fprintf(stderr, "INPUT ERROR: << Invalid number of threads >> \n");
        exit(EXIT_FAILURE);    
    }

    // Initialise the random seed
    srand(seed);

    // Scan input image
    int width, height, n_channels;
    byte_t *data = img_load(in_path, &width, &height, &n_channels);

    // Execute k-means compression
    double start_time = omp_get_wtime();
    kmeans_compression_omp(data, width, height, n_channels, n_clusters, max_iterations, n_threads);
    double execution_time = omp_get_wtime() - start_time;

    // Save the result
    img_save(out_path, data, width, height, n_channels);
    printf("Execution time: %f\n", execution_time);

    free(data);

    return EXIT_SUCCESS;
}