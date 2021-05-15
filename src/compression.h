#ifndef COMPRESSION_H
#define COMPRESSION_H

void kmeans_compression(byte_t *data, int width, int height, int n_channels, int n_clusters, int max_iterations);
void kmeans_compression_omp(byte_t *data, int width, int height, int n_channels, int n_clusters, int max_iterations, int n_threads);
void kmeans_compression_gpu(byte_t *data, int width, int height, int n_channels, int n_clusters, int max_iterations);

#endif