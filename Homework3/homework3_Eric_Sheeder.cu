#include <stdio.h>
#include <cstdlib>
#include <time.h>
#include <stdlib.h>
#include <math.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
//#include <helper_functions.h>
	
//This function takes 32 elements and computers their all-prefix-sum
//Also, stores the sum of the block in the partial_sums array
//Algorithm taken from here: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
__global__ void scan_block(int *A_gpu, int *data, int *partial_sums) {
	// Block index
    int bx = blockIdx.x;

    // Thread index
    int tx = threadIdx.x;
	
	//Thread ID
	int id = bx * blockDim.x + tx;
	
	//Chunk of 32 values that will be loaded from products
	__shared__ float values[32];
	
	values[tx] = data[id];
	
	//Up-sweep
	//Run through 5 times since log2(32) = 5
	for (int i = 1; i <= 5; i++) {
		int good_lane = 1<<i;
		int back = 1<<i-1;
		if ((tx + 1) % good_lane == 0) {
			values[tx] += values[tx - back];
		}
		__syncthreads();
	}
	
	//Set the last value to 0
	values[blockDim.x - 1] = 0;
	
	//Down-sweep
	for (int i = 5; i >= 1; i--) {
		int good_lane = 1<<i;
		int back = 1<<i-1;
		if ((tx + 1) % good_lane == 0) {
			int temp = values[tx];
			values[tx] += values[tx - back];
			values[tx - back] = temp;
		}
		__syncthreads();
	}
	
	//Store the values in their proper place in A_gpu
	A_gpu[id] = values[tx];
	//Keep track of the sum of the block of 32 values for later use
	partial_sums[bx] = values[blockDim.x - 1] + data[bx * blockDim.x + blockDim.x - 1];
}

//This function takes each scanned block and adds to each element in it the sum of the previous scanned blocks
__global__ void add_sums(int *A_gpu, int *data, int *partial_sums) {
	// Block index
    int bx = blockIdx.x;
	int block_sum = 0;
	
	// Thread index
    int tx = threadIdx.x;
	
	//Thread ID
	int id = bx * blockDim.x + tx;
	
	//Find the total sum of the data up to this block
	for (int i = 0; i < bx; i++) {
		block_sum += partial_sums[i];
	}
	
	//Add this sum to all 32 elements in our output array that the block corresponds to
	A_gpu[id] += block_sum;
}

int main (int argc, char *argv[]) {
  int *A_cpu, *A_gpu, *data;
  int n; // size of array
  srand(time(NULL)); //random numbers each time
  
  //Make sure user puts in right parameters
  if (argc !=2) {
	printf("Usage: executable.exe {arraySize}, where arraySize is a positive integer");
	exit(1);
  }
  
  n = atoi(argv[1]);

  // Read number of rows (nr), number of columns (nc) and
  // number of elements and allocate memory for row_ptr, indices, data, b and c.
  unsigned int mem_size_matrices = (n)*sizeof(int);
  data = (int *) malloc (mem_size_matrices);
  A_cpu = (int *) malloc (mem_size_matrices);
  A_gpu = (int *) malloc (mem_size_matrices);

  // Read data in coordinate format and initialize sparse matrix
  for (int i=0; i<n; i++) {
	int someInt = 1 + rand() % 1000;
	data[i] = someInt;
	A_cpu[i] = 0;
	A_gpu[i] = 0;
  }

  // MAIN COMPUTATION, SEQUENTIAL VERSION
  for (int i=0; i<n; i++) {
	int sum = 0;
    for (int j = 0; j<i; j++) {
      sum += data[j];
    }
	A_cpu[i] = sum;
  }
  
  // Allocate device memory
  int *d_A_gpu, *d_data, *d_partial_sums;
  
  cudaError_t error;

  error = cudaMalloc((void **) &d_A_gpu, mem_size_matrices);
  error = cudaMalloc((void **) &d_data, mem_size_matrices);
  error = cudaMalloc((void **) &d_partial_sums, (n/32 + 1)*sizeof(int));
  
  // copy host memory to device
  error = cudaMemcpy(d_A_gpu, A_gpu, mem_size_matrices, cudaMemcpyHostToDevice);
  error = cudaMemcpy(d_data, data, mem_size_matrices, cudaMemcpyHostToDevice);
  
  // Setup execution parameters
  int block_size = 32;
  int num_blocks = n/block_size + (n % block_size != 0);
  
  //Execute parallel code
  scan_block<<<num_blocks, block_size>>>(d_A_gpu, d_data, d_partial_sums);
  add_sums<<<num_blocks, block_size>>>(d_A_gpu, d_data, d_partial_sums);
  
  // Copy result from device to host
  error = cudaMemcpy(A_gpu, d_A_gpu, mem_size_matrices, cudaMemcpyDeviceToHost);
  
  cudaDeviceSynchronize();
  
  //Print values in our 2 c vectors to output.txt
  //Code taken from http://www.tutorialspoint.com/cprogramming/c_file_io.htm
  FILE *fpout;
  fpout = fopen("output.txt", "w+");
  fprintf(fpout, "index\tCPU\tGPU\tDifference\n");
  for (int i = 0; i < n; i++) {
	  int difference = A_cpu[i] - A_gpu[i];
	  fprintf(fpout, "%i\t%i\t%i\t%i\n", i, A_cpu[i], A_gpu[i], difference);
  }
  fclose(fpout);
  
  /*for (int i = 0; i < n; i++) {
	  printf("CPU[%i] = %i\n", i, A_cpu[i]);
	  printf("GPU[%i] = %i\n\n", i, A_gpu[i]);
  }*/
  
  // Clean up memory
  free(data);
  free(A_cpu);
  free(A_gpu);

  cudaFree(d_data);
  cudaFree(d_A_gpu);
  cudaFree(d_partial_sums);
  
  cudaDeviceReset();
  
  return 0;
}
