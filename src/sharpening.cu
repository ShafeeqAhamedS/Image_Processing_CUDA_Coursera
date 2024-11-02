#include <opencv2/opencv.hpp>  // OpenCV header for image processing
#include <cuda_runtime.h>      // CUDA Runtime API
#include <device_launch_parameters.h> // CUDA device launch parameters
#include <cstdio>              // For printf and fprintf
#include <cstdlib>             // For exit and malloc
#include <vector>              // For std::vector
#include <string>              // For std::string
#include <sstream>             // For std::ostringstream

using namespace cv;

// CUDA kernel for Canny edge detection
__global__ void cannyEdgeDetection(unsigned char *srcImage, unsigned char *dstImage, unsigned int width, unsigned int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        // Compute gradient in the x and y direction
        float Gx = -1 * srcImage[(y - 1) * width + (x - 1)] + 1 * srcImage[(y - 1) * width + (x + 1)]
                  - 2 * srcImage[y * width + (x - 1)] + 2 * srcImage[y * width + (x + 1)]
                  - 1 * srcImage[(y + 1) * width + (x - 1)] + 1 * srcImage[(y + 1) * width + (x + 1)];

        float Gy = -1 * srcImage[(y - 1) * width + (x - 1)] - 2 * srcImage[(y - 1) * width + x]
                  - 1 * srcImage[(y - 1) * width + (x + 1)] + 1 * srcImage[(y + 1) * width + (x - 1)]
                  + 2 * srcImage[(y + 1) * width + x] + 1 * srcImage[(y + 1) * width + (x + 1)];

        // Calculate gradient magnitude
        float magnitude = sqrt(Gx * Gx + Gy * Gy);
        
        // Apply a threshold to get binary edge image
        dstImage[y * width + x] = (magnitude > 100) ? 255 : 0; // Change 100 to adjust sensitivity
    }
}

void checkCudaErrors(cudaError_t r)
{
    if (r != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(r));
        exit(EXIT_FAILURE);
    }
}

int main()
{
    // Specify the input and output directories
    std::string inputDir = "data/*.jpeg"; // Input directory with image pattern
    std::string outputDir = "output/";     // Output directory for processed images

    // Get a list of images from the input directory
    std::vector<cv::String> imagePaths;
    cv::glob(inputDir, imagePaths); // Get the image paths

    // Initialize a counter for the output images
    int outputCounter = 0;

    for (const auto &imagePath : imagePaths) {
        // Read input image
        Mat image = imread(imagePath, IMREAD_GRAYSCALE);
        if (image.empty())
        {
            printf("Error: Image not found: %s\n", imagePath.c_str());
            continue; // Skip to the next image
        }

        int width = image.cols;
        int height = image.rows;
        size_t imageSize = width * height * sizeof(unsigned char);

        // Allocate host memory for output image
        unsigned char *h_outputImage = (unsigned char *)malloc(imageSize);
        if (h_outputImage == nullptr)
        {
            fprintf(stderr, "Failed to allocate host memory\n");
            continue; // Skip to the next image
        }

        // Allocate device memory
        unsigned char *d_inputImage, *d_outputImage;
        checkCudaErrors(cudaMalloc(&d_inputImage, imageSize));
        checkCudaErrors(cudaMalloc(&d_outputImage, imageSize));
        checkCudaErrors(cudaMemcpy(d_inputImage, image.data, imageSize, cudaMemcpyHostToDevice));

        // Define CUDA events for timing
        cudaEvent_t start, stop;
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));

        // Launch kernel
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

        checkCudaErrors(cudaEventRecord(start));
        cannyEdgeDetection<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, width, height);
        checkCudaErrors(cudaEventRecord(stop));

        // Synchronize events
        checkCudaErrors(cudaEventSynchronize(stop));

        // Calculate elapsed time
        float milliseconds = 0;
        checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

        // Copy result back to host
        checkCudaErrors(cudaMemcpy(h_outputImage, d_outputImage, imageSize, cudaMemcpyDeviceToHost));

        // Write output image
        std::ostringstream outputImagePath;
        outputImagePath << outputDir << "canny_edges_" << outputCounter++ << ".jpeg"; // Use a counter for unique filenames
        Mat outputImage(height, width, CV_8UC1, h_outputImage);
        imwrite(outputImagePath.str(), outputImage);

        // Free memory
        free(h_outputImage);
        cudaFree(d_inputImage);
        cudaFree(d_outputImage);

        // Destroy CUDA events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // Print elapsed time for each image
        printf("Processed %s in %f milliseconds - stored in %s\n", imagePath.c_str(), milliseconds, outputImagePath.str().c_str());
    }

    return 0;
}
