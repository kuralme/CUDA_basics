#include "cuda_tools.h"

// CUDA Kernel for vector addition
__global__ void vectorAdd(int *a, int *b, int *c, int size)
{
    // Calculate global thread ID
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    // Vector boundary guard
    if(tid < size)
    { // Each thread adds a single element
        c[tid] = a[tid] + b[tid];
    }
}

// Initialize vector
void vector_init(int* vec, int size)
{
    for(int i=0; i < size; i++)
        vec[i] = rand() % 100;
}
void fvector_init(float* vec, int size)
{
	for (int i = 0; i < size; i++) {
		vec[i] = (float)(rand() % 100);
	}
}

// Check vector add result
void verify_result(int *a, int *b, int *c, int n) {
  for (int i = 0; i < n; i++) {
    assert(c[i] == a[i] + b[i]);
  }
}
void verify_fresult(float *a, float *b, float *c, float factor, int n) {
  for (int i = 0; i < n; i++) {
	assert(c[i] == factor * a[i] + b[i]);
  }
}


int main()
{
    printf("Vector adding with CPU vs CUDA cores\n");

    // Vector size of 2^16 (65536)
    int n = 1 << 16;
    // Allocation size for vectors
    size_t bytes = sizeof(int) * n;

    // Host vector pointers
    int *hostA, *hostB, *hostC, *f;
    hostA = (int*)malloc(bytes);
    hostB = (int*)malloc(bytes);
    hostC = (int*)malloc(bytes);
    f = (int*)malloc(bytes);

    // Init vectors with random values 0 to 100
    vector_init(hostA, n);
    vector_init(hostB, n);
    
    // =============  CPU  ==============================
    auto begin = std::chrono::high_resolution_clock::now();
    for(int i=0; i < n; i++)
    {
        f[i] = hostA[i] + hostB[i];
    }
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>
        (std::chrono::high_resolution_clock::now() - begin);
    printf("CPU loop took\t\t\t%ld[us]\n", duration.count());
    
    // =============  CUDA  ======================================
    // Device vector pointers
    int *cudaA, *cudaB, *cudaC;
    cudaMalloc(&cudaA, bytes);
    cudaMalloc(&cudaB, bytes);
    cudaMalloc(&cudaC, bytes);

    cudaMemcpy(cudaA, hostA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaB, hostB, bytes, cudaMemcpyHostToDevice);

    // Threadblock and grid size
    // if num of threads is an integer, block is one dimentional
    int NUM_THREADS = 1 << 10;
    int NUM_BLOCKS = (n + NUM_THREADS - 1) / NUM_THREADS;

    // Launch kernel on default stream without shmem
    auto begin1 = std::chrono::high_resolution_clock::now();
    vectorAdd <<< NUM_BLOCKS, NUM_THREADS>>> (cudaA, cudaB, cudaC, n);
    cudaMemcpy(hostC, cudaC, bytes, cudaMemcpyDeviceToHost);

    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>
        (std::chrono::high_resolution_clock::now() - begin1);
    verify_result(hostA, hostB, hostC, n);
    printf("CUDA baseline took\t\t%ld[us]\n", duration1.count());

    // Free unified memory
    cudaFree(cudaA);
    cudaFree(cudaB);
    cudaFree(cudaC);


    // ============= Unified memory prefetch CUDA  ======================================

    // Declare unified memory pointers
    int *uniX, *uniY, *uniZ;

    // Allocation memory for these pointers
    cudaMallocManaged(&uniX, bytes);
    cudaMallocManaged(&uniY, bytes);
    cudaMallocManaged(&uniZ, bytes);
    
    // Get the device ID for prefetching calls
    int id = cudaGetDevice(&id);

    // Set some hints about the data and do some prefetching
    cudaMemAdvise(uniX, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise(uniY, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemPrefetchAsync(uniZ, bytes, id);

    // Initialize vectors
    vector_init(uniX, bytes);
    vector_init(uniY, bytes);
    
    // Pre-fetch 'a' and 'b' arrays to the specified device (GPU)
    cudaMemAdvise(uniX, bytes, cudaMemAdviseSetReadMostly, id);
    cudaMemAdvise(uniY, bytes, cudaMemAdviseSetReadMostly, id);
    cudaMemPrefetchAsync(uniX, bytes, id);
    cudaMemPrefetchAsync(uniY, bytes, id);
    
    // Threads per CTA (65536 threads per CTA)
    int BLOCK_SIZE = 1 << 10;
    int GRID_SIZE = (n + BLOCK_SIZE - 1) / BLOCK_SIZE; // CTAs per Grid

    // Call CUDA kernel
    auto begin2 = std::chrono::high_resolution_clock::now();
    vectorAdd<<<GRID_SIZE, BLOCK_SIZE>>>(uniX, uniY, uniZ, n);

    // Wait for all previous operations before using values
    // We need this because we don't get the implicit synchronization 
    // of cudaMemcpy like in the original example
    cudaDeviceSynchronize();

    // Prefetch to the host (CPU)
    cudaMemPrefetchAsync(uniX, bytes, cudaCpuDeviceId);
    cudaMemPrefetchAsync(uniY, bytes, cudaCpuDeviceId);
    cudaMemPrefetchAsync(uniZ, bytes, cudaCpuDeviceId);

    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>
        (std::chrono::high_resolution_clock::now() - begin2);
    verify_result(uniX, uniY, uniZ, n);
    printf("Prefetched unified memory took\t%ld[us]\n", duration2.count());

    // Free unified memory
    cudaFree(uniX);
    cudaFree(uniY);
    cudaFree(uniZ);

    // ============= CUDA Cublas =================================================
    float *hostcublA, *hostcublB, *hostcublC, *cublA, *cublB;
    hostcublA = (float*)malloc(bytes);
    hostcublB = (float*)malloc(bytes);
    hostcublC = (float*)malloc(bytes);
    cudaMalloc(&cublA, bytes);
	cudaMalloc(&cublB, bytes);

	fvector_init(hostcublA, n);
	fvector_init(hostcublB, n);

	// Create and initialize a new context
	cublasHandle_t handle;
	cublasCreate_v2(&handle);

	// Copy the vectors over to the device
	cublasSetVector(n, sizeof(float), hostcublA, 1, cublA, 1);
	cublasSetVector(n, sizeof(float), hostcublB, 1, cublB, 1);

	// Launch simple saxpy kernel (single precision a * x + y)
    // Function signature: handle, # elements n, A, increment, B, increment
    auto begin3 = std::chrono::high_resolution_clock::now();
	const float scale = 2.0f;
	cublasSaxpy(handle, n, &scale, cublA, 1, cublB, 1);
	cublasGetVector(n, sizeof(float), cublB, 1, hostcublC, 1);

    auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>
        (std::chrono::high_resolution_clock::now() - begin3);
	verify_fresult(hostcublA, hostcublB, hostcublC, scale, n);
    printf("Cublas kernel took\t\t%ld[us]\n", duration3.count());

	// Clean up the created handle
	cublasDestroy(handle);

    // Free memory
    cudaFree(cublA);
    cudaFree(cublB);
    return 0;
}