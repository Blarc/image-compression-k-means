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

__kernel void partial_sum_centers_new(__global unsigned char *data,
                                  __global long *centers,
                                  __global int *labels,
                                  __global double *distances,
                                  int n_pixels,
                                  int n_channels,
                                  int n_clusters,
                                  __global int *counts,
                                  __local int* loc
)
{
    int gid = (int) get_global_id(0);
    int lid = (int) get_local_id(0);
    int group_id = (int) get_group_id(0);
    int local_size = (int) get_local_size(0);

    if (gid == 0) {
        // reset centers and initialise clusters' counters
        for (int cluster = 0; cluster < n_clusters; cluster++) {
            for (int channel = 0; channel < n_channels; channel++) {
                centers[cluster * n_channels + channel] = 0;
            }
            counts[cluster] = 0;
        }
    }

    // Wait for all threads
	barrier(CLK_GLOBAL_MEM_FENCE);

    int clusters_channels = n_clusters * (n_channels + 1);
    int all = n_pixels * clusters_channels;
    int pixel = gid / clusters_channels;
    int channel = gid % (n_channels + 1);
    int cluster = (gid / (n_channels + 1)) % n_clusters;
    int label = labels[pixel];

    if (gid < all) {
        if (label == cluster) {
            if (channel < n_channels) {
                loc[lid] = data[pixel * n_channels + channel];
            }
            else {
                loc[lid] = 1;
            }
        }
        else {
            loc[lid] = 0;
        }
    }

	barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction

    int log2a = log2((float) local_size / clusters_channels);
    int floorPow2 = exp2((float)log2a) * clusters_channels;

    if (local_size != floorPow2) {
        if (lid >= floorPow2) {
            loc[lid - floorPow2] += loc[lid];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int i = (floorPow2 >> 1); i >= clusters_channels; i >>= 1) {
        if (lid < i) {
            loc[lid] += loc[lid + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // if (gid == 0) {
    //     printf("\n");
    //     for (int i = 0; i < 1; i++) {
    //         for (int j = 0; j < 16; j++) {
    //             printf("%5d ", loc[i * 16 + j]);
    //         }
    //         printf("\n");
    //     }
    // }

    barrier(CLK_GLOBAL_MEM_FENCE);

    // Sum the results

    if (lid < clusters_channels)
    {
        if (channel < n_channels) {
            atomic_add(&centers[cluster * n_channels + channel], loc[lid]);
        }
        else {
            atomic_add(&counts[cluster], loc[lid]);
        }
    }
}

// deprecated
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
    int lid = (int) get_local_id(0);

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

    // for (int cluster = 0; cluster < n_clusters; cluster++) {
    //     if (counts[cluster] == 0) {
    //         if (gid < n_pixels) {
    //             loc[lid] = distances[gid];
    //             loc_index[lid] = gid;
    //         }
    //         else {
    //             loc[lid] = DBL_MAX;
    //             loc_index[lid] = -1;
    //         }

    //         barrier(CLK_LOCAL_MEM_FENCE);

    //         // Reduction

    //         int log2a = log2((float) get_local_size(0));
    //         int floorPow2 = exp2((float)log2a);

    //         if (get_local_size(0) != floorPow2) {
    //             if (lid >= floorPow2) {
    //                 if (loc[lid - floorPow2] < loc[lid]) {
    //                     loc[lid - floorPow2] = loc[lid];
    //                     loc_index[lid - floorPow2] = gid;
    //                 }
    //                 barrier(CLK_LOCAL_MEM_FENCE);
    //             }
    //         }

    //         for (int i = (floorPow2 >> 1); i > 0; i >>= 1) {
    //             if (lid < i) {
    //                 if (loc[lid] < loc[lid + i]) {
    //                     loc[lid] = loc[lid + i];
    //                     loc_index[lid] = gid + i;
    //                 }
    //             }
    //             barrier(CLK_LOCAL_MEM_FENCE);
    //         }

    //         // write to global
            
    //         // if (gid == 0) {
    //         //     // set the centers channels to the farthest pixel's channels
    //         //     int farthest_pixel = loc_index[0];
    //         //     for (int channel = 0; channel < n_channels; channel++) {
    //         //         centers[cluster * n_channels + channel] = data[farthest_pixel * n_channels + channel];
    //         //     }
    //         //     distances[farthest_pixel] = 0;
    //         // }

    //         // if (gid == 0) {
    //         //     printf("\n\n");
    //         //     for (int j = 0; j < 64; j++) {
    //         //         for (int i = 0; i < 16; i++) {
    //         //             printf("%6.2lf ", loc[j * 16 + i]);
    //         //         }
    //         //         printf("\n");
    //         //     }
    //         //     printf("\n\n");
    //         //     for (int j = 0; j < 64; j++) {
    //         //         for (int i = 0; i < 16; i++) {
    //         //             printf("%5d ", loc_index[j * 16 + i]);
    //         //         }
    //         //         printf("\n");
    //         //     }
    //         // }
    //     }
    // }
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