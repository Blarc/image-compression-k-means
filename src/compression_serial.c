#include <stdlib.h>
#include <float.h>
#include <math.h>

#include "image_io.h"
#include "compression.h"

void initialise_centers(byte_t *data, double *centers, int n_pixels, int n_channels, int n_clusters);
void assign_pixels(byte_t *data, double *centers, int *labels, double *distances, int *changed, int n_pixels, int n_channels, int n_clusters);
void update_centers(byte_t *data, double *centers, int *labels, double *distances, int n_pixels, int n_channels, int n_clusters);
void update_data(byte_t *data, double *centers, int *labels, int n_pixels, int n_channels);


void kmeans_compression(byte_t *data, int width, int height, int n_channels, int n_clusters, int max_iterations) 
{
    int n_pixels = width * height;

    int *labels = malloc(n_pixels * sizeof(int));
    double *centers = malloc(n_clusters * n_channels * sizeof(double));
    double *distances = malloc(n_pixels * sizeof(double));

    initialise_centers(data, centers, n_pixels, n_channels, n_clusters);

    int have_clusters_changed = 0;
    for (int i = 0; i < max_iterations; i++) {
        assign_pixels(data, centers, labels, distances, &have_clusters_changed, n_pixels, n_channels, n_clusters);

        // if clusters haven't changed, they won't change in the next iteration as well, so just stop early
        if (!have_clusters_changed) {
            break;
        }

        update_centers(data, centers, labels, distances, n_pixels, n_channels, n_clusters);
    }

    update_data(data, centers, labels, n_pixels, n_channels);

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

void assign_pixels(byte_t *data, double *centers, int *labels, double *distances, int *changed, int n_pixels, int n_channels, int n_clusters)
{
    int have_clusters_changed = 0;
    int min_cluster;

    for (int pixel = 0; pixel < n_pixels; pixel++) {
        double min_distance = DBL_MAX;

        // calculate the distance between the pixel and each of the centers
        for (int cluster = 0; cluster < n_clusters; cluster++) {
            double distance = 0;

            for (int channel = 0; channel < n_channels; channel++) {
                // calculate euclidean distance between the pixel's channels and the center's channels
                double tmp = (double)(data[pixel * n_channels + channel] - centers[cluster * n_channels + channel]);
                distance += (tmp * tmp);
            }

            if (distance < min_distance) {
                min_distance = distance;
                min_cluster = cluster;
            }
        }

        distances[pixel] = min_distance;

        // if pixel's cluster has changed, update it and set 'has_changed' to True
        if (labels[pixel] != min_cluster) {
            labels[pixel] = min_cluster;
            have_clusters_changed = 1;
        }
    }

    // set the outside flag
    *changed = have_clusters_changed;
}

void update_centers(byte_t *data, double *centers, int *labels, double *distances, int n_pixels, int n_channels, int n_clusters)
{
    int *counts = malloc(n_clusters * sizeof(int));

    // reset centers and initialise clusters' counters
    for (int cluster = 0; cluster < n_clusters; cluster++) {
        for (int channel = 0; channel < n_channels; channel++) {
            centers[cluster * n_channels + channel] = 0;
        }

        counts[cluster] = 0;
    }

    // compute partial sums of the centers and update clusters counters
    for (int pixel = 0; pixel < n_pixels; pixel++) {
        int min_cluster = labels[pixel];

        // sum without division
        for (int channel = 0; channel < n_channels; channel++) {
            centers[min_cluster * n_channels + channel] += data[pixel * n_channels + channel];
        }

        counts[min_cluster] += 1;
    }

    // obtain the centers mean
    for (int cluster = 0; cluster < n_clusters; cluster++) {
        if (counts[cluster]) {
            for (int channel = 0; channel < n_channels; channel++) {
                centers[cluster * n_channels + channel] /= counts[cluster];
            }
        } else {
            // if the cluster is empty, we find the farthest pixel from its cluster's center
            double max_distance = 0;
            int farthest_pixel = 0;

            // find the farthest pixel
            for (int pixel = 0; pixel < n_pixels; pixel++) {
                if (distances[pixel] > max_distance) {
                    max_distance = distances[pixel];
                    farthest_pixel = pixel;
                }
            }

            // set the centers channels to the farthest pixel's channels
            for (int channel = 0; channel < n_channels; channel++) {
                centers[cluster * n_channels + channel] = data[farthest_pixel * n_channels + channel];
            }

            // TODO @jakobm why?
            distances[farthest_pixel] = 0;
        }
    }

    free(counts);

}

void update_data(byte_t *data, double *centers, int *labels, int n_pixels, int n_channels)
{
    for (int pixel = 0; pixel < n_pixels; pixel++) {
        int min_cluster = labels[pixel];

        for (int channel = 0; channel < n_channels; channel++) {
            data[pixel * n_channels + channel] = (byte_t)round(centers[min_cluster * n_channels + channel]);
        }
    }
}