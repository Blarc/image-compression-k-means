#ifndef COMPRESSION_H
#define COMPRESSION_H

void kmeans_compression(byte_t *data, int width, int height, int n_channels, int n_clusters, int *max_iterations, double *sse);
void kmeans_compression_omp(byte_t *data, int width, int height, int n_channels, int n_clusters, int *max_iterations, double *sse, int n_threads);

#endif