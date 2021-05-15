__kernel void assign_pixels(__global char *data,
                            __global double *centers,
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

    int min_cluster;
    int have_clusters_changed = 0;

    while( gid < n_pixels )
    {
        double min_distance = DBL_MAX;
        
        // calculate the distance between the pixel and each of the centers
        for (int cluster = 0; cluster < n_clusters; cluster++) {
            double distance = 0;

            for (int channel = 0; channel < n_channels; channel++) {
                // calculate euclidean distance between the pixel's channels and the center's channels
                double tmp = (double)(data[gid * n_channels + channel] - centers[cluster * n_channels + channel]);
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