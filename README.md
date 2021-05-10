# Image compression using K-means clustering

## Description

The ubiquity of images, videos, and other multimedia on the web nowadays causes a huge strain on the
internet infrastructure. To alleviate this problems numerous lossy and lossless compression methods 
for multimedia where invented that greatly reduce the amount of data transferred between users
who are always eager to watch new and fascinating images and videos of cats. Here we will look into image
compression using K-means clustering, which is a simple unsupervised learning algorithm

## Getting started

### Compiling
Serial
```bash
gcc -o main_serial main_serial.c image_io.c compression_serial.c -O2 -lm -fopenmp
```
Parallel
```bash
gcc -o main_omp main_omp.c image_io.c compression_omp.c -O2 -lm -fopenmp
```

## Acknowledgments

External libraries have been used for handling I/O of the images:
- **stb_image** (v2.20): for loading the color values of the pixels of the inptu images
- **stb_image_write** (v1.14): for creating images from pixel values

These libraries are part of an amazing project that can be found [here](https://github.com/nothings/stb).