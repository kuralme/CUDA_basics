#include "cuda_tools.h"

#define SIZE 256
#define SHMEM_SIZE 256 * 4


// ===================== Coop Group Fcn ===============================
void vector_init(int* vec, int size)
{
    for(int i=0; i < size; i++)
		vec[i] = 1; //rand() % 10;
}

// Reduces a thread group to a single element
__device__ int reduce_sum(cooperative_groups::thread_group g, int *temp, int val){
	int lane = g.thread_rank();

	// Each thread adds its partial sum[i] to sum[lane+i]
	for (int i = g.size() / 2; i > 0; i /= 2){
		temp[lane] = val;
		// wait for all threads to store
		g.sync();
		if (lane < i) {
			val += temp[lane + i];
		}
		// wait for all threads to load
		g.sync();
	}
	// note: only thread 0 will return full sum
	return val; 
}

// Creates partials sums from the original array
__device__ int thread_sum(int *input, int n){
	int sum = 0;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = tid; i < n / 4; i += blockDim.x * gridDim.x){
		// Cast as int4 
		int4 in = ((int4*)input)[i];
		sum += in.x + in.y + in.z + in.w;
	}
	return sum;
}

__global__ void sum_reductionCG(int *sum, int *input, int n){
	// Create partial sums from the array
	int my_sum = thread_sum(input, n);

	// Dynamic shared memory allocation
	extern __shared__ int temp[];
	
	// Identifier for a TB
	auto g = cooperative_groups::this_thread_block();
	
	// Reudce each TB
	int block_sum = reduce_sum(g, temp, my_sum);

	// Collect the partial result from each TB
	if (g.thread_rank() == 0) {
		atomicAdd(sum, block_sum);
	}
}

// ======================== Warp Reduction ================================
// For last iteration (saves useless work)
// Use volatile to prevent caching in registers (compiler optimization)
// No __syncthreads() necessary!
__device__ void warpReduce(volatile int* shmem_ptr, int t) {
	shmem_ptr[t] += shmem_ptr[t + 32];
	shmem_ptr[t] += shmem_ptr[t + 16];
	shmem_ptr[t] += shmem_ptr[t + 8];
	shmem_ptr[t] += shmem_ptr[t + 4];
	shmem_ptr[t] += shmem_ptr[t + 2];
	shmem_ptr[t] += shmem_ptr[t + 1];
}
__global__ void sumReduction3(int *v, int *vsum) {
	// Allocate shared memory
	__shared__ int partial_sum[SHMEM_SIZE];

	// Calculate thread ID
	// int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Load elements AND do first add of reduction
	// Vector now 2x as long as number of threads, so scale i
	int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	// Store first partial result instead of just the elements
	partial_sum[threadIdx.x] = v[i] + v[i + blockDim.x];
	__syncthreads();

	// Start at 1/2 block stride and divide by two each iteration
	// Stop early (call device function instead)
	for (int s = blockDim.x / 2; s > 32; s >>= 1) {
		// Each thread does work unless it is further than the stride
		if (threadIdx.x < s) {
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		}
		__syncthreads();
	}

	if (threadIdx.x < 32) {
		warpReduce(partial_sum, threadIdx.x);
	}
	
	// Let the thread 0 for this block write it's result to main memory
	// Result is inexed by this block
	if (threadIdx.x == 0) {
		vsum[blockIdx.x] = partial_sum[0];
	}
}
// =========================================================================

__global__ void sumReduction2(int *v, int *vsum) {
	// Allocate shared memory
	__shared__ int partial_sum[SHMEM_SIZE];

	// Calculate thread ID
	// int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Load elements AND do first add of reduction
	// Vector now 2x as long as number of threads, so scale i
	int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	// Store first partial result instead of just the elements
	partial_sum[threadIdx.x] = v[i] + v[i + blockDim.x];
	__syncthreads();

	// Start at 1/2 block stride and divide by two each iteration
	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		// Each thread does work unless it is further than the stride
		if (threadIdx.x < s) {
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		}
		__syncthreads();
	}
	

	// Let the thread 0 for this block write it's result to main memory
	// Result is inexed by this block
	if (threadIdx.x == 0) {
		vsum[blockIdx.x] = partial_sum[0];
	}
}

__global__ void sumReduction(int *v, int *vsum) {
	// Allocate shared memory
	__shared__ int partial_sum[SHMEM_SIZE];

	// Calculate thread ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Load elements into shared memory
	partial_sum[threadIdx.x] = v[tid];
	__syncthreads();

	// Method1 - Iterate of log base 2 the block dimension
	// Slowest due to modulo and bank conflicts
	// for (int s = 1; s < blockDim.x; s *= 2) {
	// 	// Reduce the threads performing work by half previous iteration each cycle
	// 	if (threadIdx.x % (2 * s) == 0) {
	// 		partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
	// 	}
	// 	__syncthreads();
	// }

	// Method 2 - Increase the stride of the access until we exceed the CTA dimensions
	// Slow due to bank conflicts
	// for (int s = 1; s < blockDim.x; s *= 2) {
	// 	// Change the indexing to be sequential threads
	// 	int index = 2 * s * threadIdx.x;

	// 	// Each thread does work unless the index goes off the block
	// 	if (index < blockDim.x) {
	// 	partial_sum[index] += partial_sum[index + s];
	// 	}
	// 	__syncthreads();
	// }

	// Method 3 - Start at 1/2 block stride and divide by two each iteration
	// Fast and no bank conflicts
	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		// Each thread does work unless it is further than the stride
		if (threadIdx.x < s) {
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		}
		__syncthreads();
	}

	// Let the thread 0 for this block write it's result to main memory
	// Result is inexed by this block
	if (threadIdx.x == 0) {
		vsum[blockIdx.x] = partial_sum[0];
	}
}


int main() {
	// Vector size
	int N = 1 << 16;
	size_t bytes = N * sizeof(int);

	// Host data
	std::vector<int> hostV(N);
	std::vector<int> hostVsum(N);

    // Initialize the input data
    std::generate(begin(hostV), end(hostV), [](){ return rand() % 10; });

	// Allocate device memory
	int *deviceV, *deviceVsum;
	cudaMalloc(&deviceV, bytes);
	cudaMalloc(&deviceVsum, bytes);

	// Copy to device
	cudaMemcpy(deviceV, hostV.data(), bytes, cudaMemcpyHostToDevice);
	
	// Threadblock Size
	int TB_SIZE = 256;
	// Grid Size (No padding)
	int GRID_SIZE = (N + TB_SIZE - 1) / TB_SIZE;

	// Call kernels
    auto begin1 = std::chrono::high_resolution_clock::now();
	sumReduction<<<GRID_SIZE, TB_SIZE>>>(deviceV, deviceVsum);
	sumReduction<<<1, TB_SIZE>>> (deviceVsum, deviceVsum);
	cudaMemcpy(hostVsum.data(), deviceVsum, bytes, cudaMemcpyDeviceToHost);

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>
        (std::chrono::high_resolution_clock::now() - begin1);
	assert(hostVsum[0] == std::accumulate(begin(hostV), end(hostV), 0));
    printf("Vector reduction Sum: %d\nNo bank conf took\t%ld[us]\n", *hostVsum.data(), duration.count());
    
    cudaFree(deviceVsum);

	// =============== Reduced idle ====================================
	// Grid Size (cut in half) (No padding)
	GRID_SIZE = N / TB_SIZE / 2;

	// Call kernels
    auto begin2 = std::chrono::high_resolution_clock::now();
	sumReduction2<<<GRID_SIZE, TB_SIZE>>>(deviceV, deviceVsum);
	sumReduction2<<<1, TB_SIZE>>> (deviceVsum, deviceVsum);
	cudaMemcpy(hostVsum.data(), deviceVsum, bytes, cudaMemcpyDeviceToHost);

    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>
        (std::chrono::high_resolution_clock::now() - begin2);
	assert(hostVsum[0] == std::accumulate(begin(hostV), end(hostV), 0));
    printf("Reduced idle took\t%ld[us]\n", duration2.count());
    
    cudaFree(deviceVsum);

	// =============== Warp reduction ==================================
    auto begin3 = std::chrono::high_resolution_clock::now();
	sumReduction3<<<GRID_SIZE, TB_SIZE>>>(deviceV, deviceVsum);
	sumReduction3<<<1, TB_SIZE>>> (deviceVsum, deviceVsum);
	cudaMemcpy(hostVsum.data(), deviceVsum, bytes, cudaMemcpyDeviceToHost);

    auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>
        (std::chrono::high_resolution_clock::now() - begin3);
	assert(hostVsum[0] == std::accumulate(begin(hostV), end(hostV), 0));
    printf("Warp reduction took\t%ld[us]\n", duration3.count());
    
    cudaFree(deviceVsum);
    cudaFree(deviceV);

	// ================== Coop Group ====================================
	N = 1 << 13;
	bytes = N * sizeof(int);

	// Original vector and result vector
	int *sum, *data;

	// Allocate using unified memory
	cudaMallocManaged(&sum, sizeof(int));
	cudaMallocManaged(&data, bytes);

	// Initialize vector
	vector_init(data, N);

	TB_SIZE = 256;
	GRID_SIZE = (N + TB_SIZE - 1) / TB_SIZE;

	// Call kernel with dynamic shared memory (Could decrease this to fit larger data)
    auto begin4 = std::chrono::high_resolution_clock::now();
	sum_reductionCG<<<GRID_SIZE, TB_SIZE, N * sizeof(int)>>> (sum, data, N);

	// Synchronize the kernel
	cudaDeviceSynchronize();

    auto duration4 = std::chrono::duration_cast<std::chrono::microseconds>
        (std::chrono::high_resolution_clock::now() - begin4);
	assert(*sum == 8192);
    printf("Coop groups took\t%ld[us]\n", duration4.count());

    // Free memory
    cudaFree(sum);
    cudaFree(data);
	return 0;
}