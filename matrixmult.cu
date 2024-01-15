#include "cuda_tools.h"

const int N = 1 << 10;              // Matrix dim
const int SHMEM_SIZE = 1 << 10;     // Shared memory tile size
size_t bytes = N * N * sizeof(int); // Size of matrix in bytes
const int BLOCK_SIZE = 32;          // Threads per CTA dimension
const int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;;     // Blocks per grid dimension (assumes THREADS divides N evenly)


__global__ void matrixMul(const int *a, const int *b, int *c) {
  // Compute each thread's global row and column index
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Iterate over row, and down column
  c[row * N + col] = 0;
  for (int k = 0; k < N; k++) {
    // Accumulate results for a single element
    c[row * N + col] += a[row * N + k] * b[k * N + col];
  }
}
__global__ void tiledMatrixMul(const int *a, const int *b, int *c) {
  // Compute each thread's global row and column index
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Statically allocated shared memory
  __shared__ int sharedA[SHMEM_SIZE];
  __shared__ int sharedB[SHMEM_SIZE];

  // Accumulate in temporary variable
  int tmp = 0;

  // Sweep tile across matrix
  for (int i = 0; i < N; i += blockDim.x) {
    // Load in elements for this tile
    sharedA[threadIdx.y * blockDim.x + threadIdx.x] = a[row * N + i + threadIdx.x];
    sharedB[threadIdx.y * blockDim.x + threadIdx.x] = b[i * N + threadIdx.y * N + col];

    // Wait for both tiles to be loaded in before doing computation
    __syncthreads();

    // Do matrix multiplication on the small matrix
    for (int j = 0; j < blockDim.x; j++) {
      tmp += sharedA[threadIdx.y * blockDim.x + j] * sharedB[j * blockDim.x + threadIdx.x];
    }

    // Wait for all threads to finish using current tiles before loading in new ones
    __syncthreads();
  }

  // Write back results
  c[row * N + col] = tmp;
}

// Check result with the CPU
void verify_result(std::vector<int> &a, std::vector<int> &b, std::vector<int> &c) {
  for (int i = 0; i < N; i++) { // Rows
    for (int j = 0; j < N; j++) { // Columns
      // For every element in the pair
      int tmp = 0;
      for (int k = 0; k < N; k++) {
        // Accumulate the partial results
        tmp += a[i * N + k] * b[k * N + j];
      }
      // Check against the CPU result
      assert(tmp == c[i * N + j]);
    }
  }
}
void verify_cublas_result(float *a, float *b, float *c, int n) {
  float epsilon = .001;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      float tmp = 0;
      for (int k = 0; k < N; k++) {
        tmp += a[k * N + i] * b[j * N + k];
      }
      assert(fabs(c[j * N + i] - tmp) < epsilon);
    }
  }
}

int main() {
  printf("Beginning matrix multiplications of two %dx%d matrices\n", N, N);

  // Host vectors
  std::vector<int> hostA(N * N);
  std::vector<int> hostB(N * N);
  std::vector<int> hostC(N * N);
  std::vector<int> hostD(N * N);

  // Initialize matrices
  std::generate(hostA.begin(), hostA.end(), []() { return rand() % 100; });
  std::generate(hostB.begin(), hostB.end(), []() { return rand() % 100; });

  // Allocate device memory
  int *deviceA, *deviceB, *deviceC, *deviceD;
  cudaMalloc(&deviceA, bytes);
  cudaMalloc(&deviceB, bytes);
  cudaMalloc(&deviceC, bytes);
  cudaMalloc(&deviceD, bytes);

  // Copy data to the device
  cudaMemcpy(deviceA, hostA.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB.data(), bytes, cudaMemcpyHostToDevice);

  // Use dim3 structs for block and grid dimensions
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 blocks(GRID_SIZE, GRID_SIZE);

  // Launch kernel
  auto begin = std::chrono::high_resolution_clock::now();  
  matrixMul<<<blocks, threads>>>(deviceA, deviceB, deviceC);
  cudaMemcpy(hostC.data(), deviceC, bytes, cudaMemcpyDeviceToHost);

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>
        (std::chrono::high_resolution_clock::now() - begin);
  verify_result(hostA, hostB, hostC);
  printf("Baseline took\t\t%ld[us]\n", duration.count());

  cudaFree(deviceC);
  CHECK_LAST_CUDA_ERROR();

  // =================== Cache Tiled kernel ==================================
  auto begin1 = std::chrono::high_resolution_clock::now();

  tiledMatrixMul<<<blocks, threads>>>(deviceA, deviceB, deviceD);
  cudaMemcpy(hostD.data(), deviceD, bytes, cudaMemcpyDeviceToHost);

  auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>
        (std::chrono::high_resolution_clock::now() - begin1);
  verify_result(hostA, hostB, hostD);
  printf("Cache tiled took\t%ld[us]\n", duration1.count());

  // Free memory on device
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceD);
  CHECK_LAST_CUDA_ERROR();

  // ====================== Cublas ============================================
  // Allocate memory
  float *hostfA, *hostfB, *hostfC;
  float *device_cublA, *device_cublB, *device_cublC;
  hostfA = (float*)malloc(bytes);
  hostfB = (float*)malloc(bytes);
  hostfC = (float*)malloc(bytes);
  cudaMalloc(&device_cublA, bytes);
  cudaMalloc(&device_cublB, bytes);
  cudaMalloc(&device_cublC, bytes);

  // Pseudo random number generator
  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

  // Set the seed
  curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());

  // Fill the matrixes with random numbers on the device
  curandGenerateUniform(prng, device_cublA, N*N);
  curandGenerateUniform(prng, device_cublB, N*N);

	// Create and initialize a new context
	cublasHandle_t handle;
	cublasCreate_v2(&handle);

  // Scaling factors
  float alpha = 1.0f;
  float beta = .0f;

  // Calculate c = (alpha*A) * B + (beta*C)
  // (m X n) * (n x k) = (m X k)
  // Signature: handle, operation, operation, m, n, k, alpha, A, lda, beta, ldb, beta, C, ldc
  auto begin2 = std::chrono::high_resolution_clock::now();
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, device_cublA, N, device_cublB, N, &beta, device_cublC, N);

  // Copy back to host
  cudaMemcpy(hostfA, device_cublA, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(hostfB, device_cublB, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(hostfC, device_cublC, bytes, cudaMemcpyDeviceToHost);

  auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>
        (std::chrono::high_resolution_clock::now() - begin2);
  verify_cublas_result(hostfA, hostfB, hostfC, N);
  printf("Cublas lib took\t\t%ld[us]\n", duration2.count());
  
  // Free memory on device
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  CHECK_LAST_CUDA_ERROR();
  return 0;
}