#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

__kernel void assign_pixels(__global unsigned char *data,
                            __global long *centers,
                            __global int *labels,
                            __global double *distances,
                            __global int *changed,
                            int n_pixels,
                            int n_channels,
                            int n_clusters
)
{

    int lid = (int) get_local_id(0);

    // gid = pixel
    int gid = (int) get_global_id(0);

    int min_cluster = 0;
    int have_clusters_changed = 0;

    while( gid < n_pixels )
    {
        double min_distance = DBL_MAX;
        
        // calculate the distance between the pixel and each of the centers
        for (int cluster = 0; cluster < n_clusters; cluster++) {
            long distance = 0.0f;

            for (int channel = 0; channel < n_channels; channel++) {
                // calculate euclidean distance between the pixel's channels and the center's channels
                double tmp = (double) (data[gid * n_channels + channel] - centers[cluster * n_channels + channel]);
                distance += (tmp * tmp);
            }

            if (distance < min_distance) {
                min_distance = distance;
                min_cluster = cluster;
            }
        }

        distances[gid] = min_distance;

        // if pixel's cluster has changed, update it and set 'has_changed' to True
        if (labels[gid] != min_cluster) {
            labels[gid] = min_cluster;
            have_clusters_changed = 1;
        }


        gid += get_global_size(0);
    }

    // set the outside flag
    if (have_clusters_changed) {
        *changed = 1;
    }
}

__kernel void partial_sum_centers(__global unsigned char *data,
                                  __global long *centers,
                                  __global int *labels,
                                  __global double *distances,
                                  int n_pixels,
                                  int n_channels,
                                  int n_clusters,
                                  __global int *counts
)
{
    int gid = (int) get_global_id(0); 

    // TODO @jakobm This could probably be optimized with more threads
    if (gid == 0) {
        // reset centers and initialise clusters' counters
        for (int cluster = 0; cluster < n_clusters; cluster++) {
            for (int channel = 0; channel < n_channels; channel++) {
                centers[cluster * n_channels + channel] = 0.0f;
            }
            counts[cluster] = 0;
        }
    }

    // Wait for all threads
	barrier(CLK_GLOBAL_MEM_FENCE);

    // TODO @jakobm This could probably be optimized by sequential memory access
    int min_cluster, channel;
    while(gid < n_pixels)
    {
        min_cluster = labels[gid];

        for (channel = 0; channel < n_channels; channel++) {
            atom_add(&centers[min_cluster * n_channels + channel], ((long) data[gid * n_channels + channel]));
        }

        atomic_inc(&counts[min_cluster]);

        gid += get_global_size(0);
    }
}

__kernel void centers_mean(__global unsigned char *data,
                           __global long *centers,
                           __global double *distances,
                           __global int *counts,
                           int n_pixels,
                           int n_channels,
                           int n_clusters

)
{
    int gid = (int) get_global_id(0); 

    // TODO @jakobm This could probably be optimized with more threads
    if (gid == 0) {
        for (int cluster = 0; cluster < n_clusters; cluster++) {
            if (counts[cluster]) {
                for (int channel = 0; channel < n_channels; channel++) {
                    centers[cluster * n_channels + channel] /= counts[cluster];
                }
            }
            else {
                // if the cluster is empty, we find the farthest pixel from its cluster's center
                long max_distance = 0;
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

                distances[farthest_pixel] = 0;
            }
        }
    }
}

__kernel void update_data(__global unsigned char *data,
                          __global long *centers,
                          __global int *labels,
                          int n_pixels,
                          int n_channels
)
{
    int gid = (int) get_global_id(0);

    int min_cluster, channel;
    while(gid < n_pixels)
    {
        min_cluster = labels[gid];

        for (channel = 0; channel < n_channels; channel++) {
            // data[gid * n_channels + channel] = (unsigned char) round(centers[min_cluster * n_channels + channel]);
            data[gid * n_channels + channel] = (unsigned char) (centers[min_cluster * n_channels + channel]);
        }

        gid += get_global_size(0);
    }
}