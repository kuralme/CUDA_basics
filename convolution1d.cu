#include "cuda_tools.h"

// Length of our convolution mask
#define MASK_LENGTH 7

// Allocate space for the mask in constant memory
__constant__ int shmask[MASK_LENGTH];

// 1-D convolution kernel
//  Arguments:
//      array   = padded array
//      mask    = convolution mask
//      result  = result array
//      n       = number of elements in array
//      m       = number of elements in the mask
__global__ void convolution_1d(int *array, int *mask, int *result, int n, int m) {
  // Global thread ID calculation
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Calculate radius of the mask
  int r = m / 2;

  // Calculate the starting point for the element
  int start = tid - r;

  // Temp value for calculation
  int temp = 0;

  // Go over each element of the mask
  for (int j = 0; j < m; j++) {
    // Ignore elements that hang off (0s don't contribute)
    if (((start + j) >= 0) && (start + j < n)) {
      // accumulate partial results
      temp += array[start + j] * mask[j];
    }
  }

  // Write-back the results
  result[tid] = temp;
}
__global__ void cmm_convolution_1d(int *array, int *result, int n) {
  // Global thread ID calculation
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Calculate radius of the mask
  int r = MASK_LENGTH / 2;

  // Calculate the starting point for the element
  int start = tid - r;

  // Temp value for calculation
  int temp = 0;

  // Go over each element of the mask
  for (int j = 0; j < MASK_LENGTH; j++) {
    // Ignore elements that hang off (0s don't contribute)
    if (((start + j) >= 0) && (start + j < n)) {
      // accumulate partial results
      temp += array[start + j] * shmask[j];
    }
  }

  // Write-back the results
  result[tid] = temp;
}
__global__ void tiled_convolution_1d(int *array, int *result, int n) {
  // Global thread ID calculation
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Store all elements needed to compute output in shared memory
  extern __shared__ int s_array[];

  // r: The number of padded elements on either side
  int r = MASK_LENGTH / 2;

  // d: The total number of padded elements
  int d = 2 * r;

  // Size of the padded shared memory array
  int n_padded = blockDim.x + d;

  // Offset for the second set of loads in shared memory
  int offset = threadIdx.x + blockDim.x;

  // Global offset for the array in DRAM
  int g_offset = blockDim.x * blockIdx.x + offset;

  // Load the lower elements first starting at the halo
  // This ensure divergence only once
  s_array[threadIdx.x] = array[tid];

  // Load in the remaining upper elements
  if (offset < n_padded) {
    s_array[offset] = array[g_offset];
  }
  __syncthreads();

  // Temp value for calculation
  int temp = 0;

  // Go over each element of the mask
  for (int j = 0; j < MASK_LENGTH; j++) {
    temp += s_array[threadIdx.x + j] * shmask[j];
  }

  // Write-back the results
  result[tid] = temp;
}

// Verify the result on the CPU
void verify_result(int *array, int *mask, int *result, int n) {
  int radius = MASK_LENGTH / 2;
  int temp;
  int start;
  for (int i = 0; i < n; i++) {
    start = i - radius;
    temp = 0;
    for (int j = 0; j < MASK_LENGTH; j++) {
      if ((start + j >= 0) && (start + j < n)) {
        temp += array[start + j] * mask[j];
      }
    }
    assert(temp == result[i]);
  }
}
void verify_result_tiled(int *array, int *mask, int *result, int n) {
  int temp;
  for (int i = 0; i < n; i++) {
    temp = 0;
    for (int j = 0; j < MASK_LENGTH; j++) {
      temp += array[i + j] * mask[j];
    }
    assert(temp == result[i]);
  }
}

int main() {
    // Number of elements and size in result array
    int n = 1 << 20;
    int bytes_result = n * sizeof(int);

    // Size of mask in bytes
    int m = MASK_LENGTH;
    int bytes_m = m * sizeof(int);

    // Allocate the array (include edge elements)...
    std::vector<int> h_array(n);

    // ... and initialize it
    std::generate(begin(h_array), end(h_array), [](){ return rand() % 100; });

    // Allocate the mask and initialize it
    std::vector<int> h_mask(m);
    std::generate(begin(h_mask), end(h_mask), [](){ return rand() % 10; });

    // Allocate space for the result
    std::vector<int> h_result(n);

    // Allocate space on the device
    int *d_array, *d_mask, *d_result;
    cudaMalloc(&d_array, bytes_result);
    cudaMalloc(&d_mask, bytes_m);
    cudaMalloc(&d_result, bytes_result);

    // Copy the data to the device
    cudaMemcpy(d_array, h_array.data(), bytes_result, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask.data(), bytes_m, cudaMemcpyHostToDevice);

    // Threads per TB
    int THREADS = 256;
    // Number of TBs
    int GRID = (n + THREADS - 1) / THREADS;

    // Call the kernel
    auto begin1 = std::chrono::high_resolution_clock::now();
    convolution_1d<<<GRID, THREADS>>>(d_array, d_mask, d_result, n, m);
    cudaMemcpy(h_result.data(), d_result, bytes_result, cudaMemcpyDeviceToHost);

    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>
        (std::chrono::high_resolution_clock::now() - begin1);
    verify_result(h_array.data(), h_mask.data(), h_result.data(), n);
    printf("Naive conv took\t\t%ld[us]\n", duration1.count());

    // Free allocated memory on the device and host
    cudaFree(d_result);
    cudaFree(d_mask);
    cudaFree(d_array);

    // ==================== Mask in constant memory =============================
    // Size of the mask in bytes
    size_t bytes_mask = MASK_LENGTH * sizeof(int);

    // Allocate the host array (include edge elements)...
    int *hostArr = new int[n];
    for (int i = 0; i < n; i++) {
        hostArr[i] = rand() % 100;
    }

    // Allocate the mask and initialize it
    int *hostMask = new int[MASK_LENGTH];
    for (int i = 0; i < MASK_LENGTH; i++) {
        hostMask[i] = rand() % 10;
    }

    // Allocate space for the result
    int *hostResult = new int[n];

    // Allocate space on the device
    int *deviceArr, *deviceResult;
    cudaMalloc(&deviceArr, bytes_result);
    cudaMalloc(&deviceResult, bytes_result);

    // Copy the data to the device
    cudaMemcpy(deviceArr, hostArr, bytes_result, cudaMemcpyHostToDevice);

    // Copy the data directly to the symbol
    // Would require 2 API calls with cudaMemcpy
    cudaMemcpyToSymbol(shmask, hostMask, bytes_mask);

    // Threads per TB
    THREADS = 256;
    GRID = (n + THREADS - 1) / THREADS;

    // Call the kernel
    auto begin2 = std::chrono::high_resolution_clock::now();
    cmm_convolution_1d<<<GRID, THREADS>>>(deviceArr, deviceResult, n);
    cudaMemcpy(hostResult, deviceResult, bytes_result, cudaMemcpyDeviceToHost);

    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>
        (std::chrono::high_resolution_clock::now() - begin2);
    verify_result(hostArr, hostMask, hostResult, n);
    printf("Const mem mask took\t%ld[us]\n", duration2.count());

    // Free allocated memory on the device and host
    delete[] hostArr;
    delete[] hostResult;
    cudaFree(deviceArr);
    cudaFree(deviceResult);
    CHECK_LAST_CUDA_ERROR();

    // ==================== Tiled array =============================
    // Radius for padding the array
    int r = MASK_LENGTH / 2;
    int n_p = n + r * 2;

    // Size of the padded array in bytes
    size_t bytes_p = n_p * sizeof(int);

    // Allocate the array (include edge elements)...
    int *hostArr1 = new int[n_p];

    // ... and initialize it
    for (int i = 0; i < n_p; i++) {
        if ((i < r) || (i >= (n + r))) {
            hostArr1[i] = 0;
        } else {
            hostArr1[i] = rand() % 100;
        }
    }

    // Allocate space for the result
    int *hostResult1 = new int[n];

    // Allocate space on the device
    int *deviceArr1, *deviceResult1;
    cudaMalloc(&deviceArr1, bytes_p);
    cudaMalloc(&deviceResult1, bytes_result);

    // Copy the data to the device
    cudaMemcpy(deviceArr1, hostArr1, bytes_p, cudaMemcpyHostToDevice);

    //  ---- Will use the same mask as previous ----
    // cudaMemcpyToSymbol(mask, hostMask, bytes_m);

    // Threads per TB
    THREADS = 256;
    GRID = (n + THREADS - 1) / THREADS;

    // Amount of space per-block for shared memory
    // This is padded by the overhanging radius on either side
    size_t SHMEM = (THREADS + r * 2) * sizeof(int);

    // Call the kernel
    auto begin3 = std::chrono::high_resolution_clock::now();
    tiled_convolution_1d<<<GRID, THREADS, SHMEM>>>(deviceArr1, deviceResult1, n);
    cudaMemcpy(hostResult1, deviceResult1, bytes_result, cudaMemcpyDeviceToHost);

    auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>
        (std::chrono::high_resolution_clock::now() - begin3);
    verify_result_tiled(hostArr1, hostMask, hostResult1, n);
    printf("Tiled mem took\t\t%ld[us]\n", duration3.count());

    // Free allocated memory on the device and host
    delete[] hostArr1;
    delete[] hostResult1;
    delete[] hostMask;
    cudaFree(deviceArr1);
    cudaFree(deviceResult1);
    return 0;
}