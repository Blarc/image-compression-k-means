#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

__kernel void assign_pixels_old(__global unsigned char *data,
                            __global long *centers,
                            __global int *labels,
                            __global double *distances,
                            int n_pixels,
                            int n_channels,
                            int n_clusters
)
{

    int lid = (int) get_local_id(0);

    // gid = pixel
    int gid = (int) get_global_id(0);

    int min_cluster = 0;
    // int have_clusters_changed = 0;

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
            // have_clusters_changed = 1;
        }


        gid += get_global_size(0);
    }

    // set the outside flag
    // if (have_clusters_changed) {
    //     *changed = 1;
    // }
}

__kernel void assign_pixels(__global unsigned char *data,
                            __global long *centers,
                            __global int *labels,
                            __global double *distances,
                            int n_pixels,
                            int n_channels,
                            int n_clusters,
                            __local double *distances_loc,
                            int size
)
{
    int gid = (int) get_global_id(0);
    int lid = (int) get_local_id(0);

    // split
    // thread = channel
    int fixed_size = size - (size % (n_channels * n_clusters));
    if(lid < fixed_size)
    {
        int pixel = gid / (n_clusters * n_channels);
        int channel_index = gid % n_channels;
        int cluster_index = gid / (n_channels * n_pixels);

        int channel = (int) data[pixel * n_channels + channel_index];
        int center_channel = (int) centers[cluster_index * n_channels + channel_index];

        double tmp = channel - center_channel;
        distances_loc[lid] = tmp * tmp;
    }

	barrier(CLK_LOCAL_MEM_FENCE);

    // if (lid == 0) {
    //     for (int i = 0; i < fixed_size; i++) {
    //         if (distances_loc[i] > 0) {
    //             printf("%lf\n", distances_loc[i]);
    //         }
    //     }
    // }

    // barrier(CLK_LOCAL_MEM_FENCE);

    // sum
    // thread = cluster
    int i = 0;
    int num_clusters = fixed_size / n_channels;
    if(lid < num_clusters)
    {
        for (i = 1; i < n_channels; i++) {
            distances_loc[lid * n_channels] += distances_loc[lid * n_channels + i];
        }
    }

	// barrier(CLK_LOCAL_MEM_FENCE);

    // minimum
    // thread = pixel
    i = 0;
    int num_pixels = fixed_size / (n_channels * n_clusters);
    if(lid < num_pixels)
    {
        int min_cluster = 0;
        double min_distance = distances_loc[lid * n_clusters * n_channels];

        for (i = 1; i < n_clusters; i++) {
            if (distances_loc[lid * n_clusters * n_channels + i * n_channels] < min_distance) {
                min_distance = distances_loc[lid * n_clusters * n_channels + i * n_channels];
                min_cluster = i;
            }
        }

        distances[gid % n_pixels] = min_distance;
        labels[gid % n_pixels] = min_cluster;
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