# Canny Edge Detection with CUDA

## Overview
This project implements a Canny edge detection algorithm using CUDA to leverage the parallel processing capabilities of GPUs. The application reads grayscale images from a specified input directory, processes them using a CUDA kernel, and outputs the results to a specified output directory. This implementation aims to enhance the performance of edge detection tasks, particularly for large image datasets.

## Code Organization
- **bin/**: This folder holds all binary/executable code that is built automatically or manually. Executable code should use the `.exe` extension or a programming language-specific extension.
  
- **data/**: This folder holds example data in any format. If the original data is large or can be brought in via scripts, this can be left blank in the repository to avoid major downloads when only the code/structure is desired.

- **src/**: The source code is placed here in a hierarchical fashion, as appropriate.

- **README.md**: This file holds the description of the project so that anyone cloning or deciding if they want to clone this repository can understand its purpose to help with their decision.

- **Makefile**: This file provides instructions for building the project's code automatically.

- **run.sh**: An optional script used to run your executable code, either with or without command-line arguments.

## Key Concepts
- CUDA Programming
- Image Processing
- Edge Detection Algorithms
- Memory Management in CUDA

## Supported OS
- Linux
- Windows (with appropriate modifications)

## Supported CPU Architecture
- x86_64

## CUDA APIs Involved
- `cudaMalloc()`
- `cudaMemcpy()`
- `cudaFree()`
- `cudaEventCreate()`
- `cudaEventRecord()`
- `cudaEventSynchronize()`
- `cudaEventElapsedTime()`

## Dependencies Needed to Build/Run
- CUDA Toolkit
- OpenCV (version 4 or later)

## Prerequisites
- Install the CUDA Toolkit on your machine. Ensure that your GPU is compatible with CUDA.
- Install OpenCV using the appropriate package manager for your operating system. 

## Build and Run
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required dependencies:
   - For OpenCV, you can use:
     ```bash
     sudo apt-get install libopencv-dev  # For Debian/Ubuntu
     ```

3. Build the project:
   ```bash
   make all
   ```

4. Run the application:
   ```bash
   ./run.sh
   ```

5. The processed images will be stored in the `output/` directory.

## Sample Output

|Input|Output|
|:-:|:-:|
|![image4885](https://github.com/user-attachments/assets/5f01939e-a07a-4b5d-ba97-5dad20abc952) |![canny_edges_7](https://github.com/user-attachments/assets/98a54f9d-528b-41bd-ad14-73a717963e16) |
|![image4944](https://github.com/user-attachments/assets/12a7beb7-b870-49f1-b3b6-cf78e7df16a5) |![canny_edges_45](https://github.com/user-attachments/assets/79d0a0e8-f540-4cf0-a62a-c886f3266887) |
