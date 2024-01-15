#include "cuda_tools.h"

// Number of bins for our plot
constexpr int BINS = 7;
constexpr int DIV = ((26 + BINS - 1) / BINS);

// GPU kernel for computing a histogram
// Takes:
//  a: Problem array in global memory
//  result: result array
//  N: Size of the array
__global__ void histogram(char *a, int *result, int N) {
  // Calculate global thread ID
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Calculate the bin positions where threads are grouped together
  int alpha_position;
  for (int i = tid; i < N; i += (gridDim.x * blockDim.x)) {
    // Calculate the position in the alphabet
    alpha_position = a[i] - 'a';
    atomicAdd(&result[alpha_position / DIV], 1);
  }
}
__global__ void histogram_shmem(char *a, int *result, int N) {
  // Calculate global thread ID
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Allocate a local histogram for each TB
  __shared__ int s_result[BINS];

  // Initalize the shared memory to 0
  if (threadIdx.x < BINS) {
    s_result[threadIdx.x] = 0;
  }

  // Wait for shared memory writes to complete
  __syncthreads();

  // Calculate the bin positions locally
  int alpha_position;
  for (int i = tid; i < N; i += (gridDim.x * blockDim.x)) {
    // Calculate the position in the alphabet
    alpha_position = a[i] - 'a';
    atomicAdd(&s_result[(alpha_position / DIV)], 1);
  }

  // Wait for shared memory writes to complete
  __syncthreads();

  // Combine the partial results
  if (threadIdx.x < BINS) {
    atomicAdd(&result[threadIdx.x], s_result[threadIdx.x]);
  }
}

int main() {
  // Declare our problem size
  int N = 1 << 24;

  // Allocate memory on the host
  std::vector<char> hostInput(N);

  // Allocate space for the binned result
  std::vector<int> hostResult(BINS);
  std::vector<int> hostResult2(BINS);

  // Initialize the array
  srand(1);
  generate(begin(hostInput), end(hostInput), []() { return 'a' + (rand() % 26); });

  // Allocate memory on the device
  char *deviceInput;
  int *deviceResult, *deviceResult2;
  cudaMalloc(&deviceInput, N);
  cudaMalloc(&deviceResult, BINS * sizeof(int));
  cudaMalloc(&deviceResult2, BINS * sizeof(int));

  // Copy the array to the device
  cudaMemcpy(deviceInput, hostInput.data(), N, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceResult, hostResult.data(), BINS * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceResult2, hostResult2.data(), BINS * sizeof(int), cudaMemcpyHostToDevice);

  // Number of threads per threadblock
  int THREADS = 512;
  // Calculate the number of threadblocks
  int BLOCKS = N / THREADS / 4;

  printf("Histogram calculation 16777216 letter to 7 bins\n");
  // ===================== With global memory ============================
  auto begin1 = std::chrono::high_resolution_clock::now();
  histogram<<<BLOCKS, THREADS>>>(deviceInput, deviceResult, N);
  cudaMemcpy(hostResult.data(), deviceResult, BINS * sizeof(int), cudaMemcpyDeviceToHost);

  auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>
      (std::chrono::high_resolution_clock::now() - begin1);
  printf("Global mem took\t%ld[us]\n", duration1.count());
  
  // Functional test (the sum of all bins == N)
  assert(N == std::accumulate(begin(hostResult), end(hostResult), 0));

  // ===================== With shared memory ============================
  auto begin2 = std::chrono::high_resolution_clock::now();
  histogram_shmem<<<BLOCKS, THREADS>>>(deviceInput, deviceResult2, N);
  cudaMemcpy(hostResult2.data(), deviceResult2, BINS * sizeof(int), cudaMemcpyDeviceToHost);

  auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>
      (std::chrono::high_resolution_clock::now() - begin2);
  printf("Shared mem took\t%ld[us]\n", duration2.count());

  // Functional test (the sum of all bins == N)
  assert(N == std::accumulate(begin(hostResult2), end(hostResult2), 0));

  // Write the data out for gnuplot
  std::fstream output_file;
  output_file.open("histogram.dat", std::fstream::ios_base::out | std::fstream::ios_base::trunc);
  for (auto i : hostResult) {
    output_file << i << " \n\n";
  }
  output_file.close();

  // Free memory
  cudaFree(deviceInput);
  cudaFree(deviceResult);

  return 0;
}