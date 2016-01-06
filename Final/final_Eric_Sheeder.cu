#include <stdio.h>
#include <cstdlib>
#include <time.h>
#include <stdlib.h>
#include <math.h>

/*
	Authors: Eric Sheeder, Gokul Natesan, Jacob Hollister
	Parallel Computing Final Project
	
	This code generates 2 large matrices and multiplies them, once on the GPU and once on the CPU
	It expects 3 variables on the command line, n, m, and p, where n and m are the dimensions of A and m and p are the dimensions of B
	This code currently only works accurately when n, m, and p are all multiples of 32
	
	Algorithm was learned and taken from this paper:
	https://webs.um.es/jmgarcia/miwiki/lib/exe/fetch.php?id=pubs&cache=cache&media=parco09.pdf
*/

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
//#include <helper_functions.h>

//This function computes the final values for a single tile in the output matrix (C)
__global__ void compute_tile(int *d_A, int *d_B, int *d_C_gpu, int n, int m, int p) {
	// Block index
    int bx = blockIdx.x;
	int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	//Chunk of 32x32 values loaded from A and B
	__shared__ float A[32][32];
	__shared__ float B[32][32];
	
	//Offset into A and B for each thread so it knows which piece of data to load into the shared arrays
	int indexA = by * (m*blockDim.y) + ty * m + tx;
	int indexB = bx*blockDim.x + ty*p + tx;
	
	//Each thread keeps track of its own sum in this variable
	int sum = 0;
		
	//Run through multiple tiles in A and B to compute our values for our tile in C
	for (int i = 0; i < p; i+=blockDim.x) {
		//Have each thread load a value into A and B
		A[ty][tx] = d_A[indexA];
		B[ty][tx] = d_B[indexB];
		//Synch threads so all threads wait until all data is loaded before we start calculating
		__syncthreads();
		//Have each thread run through the section of A and B we are at, doing 32 multiplications and summing them
		for (int j = 0; j < blockDim.x; j++) {
			sum += A[ty][j] * B[j][tx];
		}
		//Synch threads again so we know each thread is ready to move on to the next part of A and Block
		__syncthreads();
		indexA += blockDim.x;
		indexB += p*blockDim.x;
	}

	
	//Each thread should now have a complete value for its part in C, so figure out where it should go and store it
	int indexC = bx * blockDim.x + by * (p * blockDim.y) + p * ty + tx;
	d_C_gpu[indexC] = sum;
}

int main (int argc, char *argv[]) {
  int *A, *B, *C_cpu, *C_gpu; // matrices
  int n, m, p; // dimensions of matrices
  srand(time(NULL)); //random numbers each time
  clock_t cpu_start_time, cpu_end_time;
  cudaEvent_t gpu_start_time, gpu_end_time;
  double cpu_total_time;
  float gpu_total_time;
  
  //Make sure user puts in right parameters
  if (argc !=4) {
	printf("Usage: <./executable.exe n m p>, where n, m, and p are the dimensions of A (nxm) and B (mxp)");
	exit(1);
  }
  
  n = atoi(argv[1]);
  m = atoi(argv[2]);
  p = atoi(argv[3]);
  
  // Read number of rows (nr), number of columns (nc) and
  // number of elements and allocate memory for row_ptr, indices, data, b and c.
  unsigned int mem_size_A = (n*m)*sizeof(int);
  A = (int *) malloc (mem_size_A);
  unsigned int mem_size_B = (m*p)*sizeof(int);
  B = (int *) malloc (mem_size_B);
  unsigned int mem_size_C = (n*p)*sizeof(int);
  C_cpu = (int *) malloc (mem_size_C);
  C_gpu = (int *) malloc (mem_size_C);

  // Fill A with randomly generated data
  for (int i=0; i<n*m; i++) {
	int someInt = 1 + rand() % 10;
	A[i] = someInt;
  }
  
  // Fill B with randomly generated data
  for (int i=0; i<m*p; i++) {
	int someInt = 1 + rand() % 10;
	B[i] = someInt;
  }
  
  // Fill C with 0s
  for (int i=0; i<n*p; i++) {
	C_cpu[i] = 0;
	C_gpu[i] = 0;
  }
  
  // Allocate device memory
  int *d_A, *d_B, *d_C_gpu;
  
  cudaError_t error;

  error = cudaMalloc((void **) &d_A, mem_size_A);
  error = cudaMalloc((void **) &d_B, mem_size_B);
  error = cudaMalloc((void **) &d_C_gpu, mem_size_C);
  
  // copy host memory to device
  error = cudaMemcpy(d_A, A, mem_size_A, cudaMemcpyHostToDevice);
  error = cudaMemcpy(d_B, B, mem_size_B, cudaMemcpyHostToDevice);
  error = cudaMemcpy(d_C_gpu, C_gpu, mem_size_C, cudaMemcpyHostToDevice);
  
  // Setup execution parameters for parallel
  dim3 tile_size(32, 32);
  dim3 num_blocks(n/tile_size.x + (n % tile_size.x != 0), p/tile_size.y + (p % tile_size.y != 0));
  
  //Execute parallel code
  error = cudaEventCreate(&gpu_start_time);
  error = cudaEventCreate(&gpu_end_time);
  
  error = cudaEventRecord(gpu_start_time, NULL);
  //compute_tile<<<num_blocks, tile_size>>>(d_A, d_B, d_C_gpu, n, m, p);
  error = cudaEventRecord(gpu_end_time, NULL);
  error = cudaEventSynchronize(gpu_end_time);
  error = cudaEventElapsedTime(&gpu_total_time, gpu_start_time, gpu_end_time);
  
  cpu_start_time = clock();
  // MAIN COMPUTATION, SEQUENTIAL VERSION
  for (int row=0; row < n; row++) {
	for (int col = 0; col < p; col++) {
		int sum = 0;
		for (int i = 0; i < m; i++) {
			sum += A[row*m + i] * B[i*p + col];
		}
		C_cpu[row*p + col] = sum;
	}
  } 
  cpu_end_time = clock();
  
  cpu_total_time = ((double) (cpu_end_time - cpu_start_time)) / CLOCKS_PER_SEC;
  
  //Copy result from device to host
  error = cudaMemcpy(C_gpu, d_C_gpu, mem_size_C, cudaMemcpyDeviceToHost);
  
  cudaDeviceSynchronize();
  
  //Print values in our 2 c vectors to output.txt
  //Code taken from http://www.tutorialspoint.com/cprogramming/c_file_io.htm
  /*FILE *fpout;
  fpout = fopen("output.txt", "w+");
  fprintf(fpout, "index\tCPU\tGPU\tDifference\n");
  for (int i = 0; i < n; i++) {
	  int difference = A_cpu[i] - A_gpu[i];
	  fprintf(fpout, "%i\t%i\t%i\t%i\n", i, A_cpu[i], A_gpu[i], difference);
  }
  fclose(fpout);*/
  
  //Print out matrix A
  /*printf("Matrix A:\n");
  for (int row = 0; row < n; row++) {
	  for (int col = 0; col < m; col++) {
		  printf("%i ", A[row*n + col]);
	  }
	  printf("\n");
  }*/
  
  //Print out matrix B
  /*printf("Matrix B:\n");
  for (int row = 0; row < m; row++) {
	  for (int col = 0; col < p; col++) {
		  printf("%i ", B[row*n + col]);
	  }
	  printf("\n");
  }*/
  
  //Print out matrix C (CPU)
  /*printf("Matrix C (CPU):\n");
  for (int row = 0; row < n; row++) {
	  for (int col = 0; col < p; col++) {
		  printf("%i ", C_cpu[row*n + col]);
	  }
	  printf("\n");
  }*/
  
  //Print out matrix C (GPU)
  /*printf("Matrix C (GPU):\n");
  for (int row = 0; row < n; row++) {
	  for (int col = 0; col < p; col++) {
		  printf("%i ", C_gpu[row*n + col]);
	  }
	  printf("\n");
  }*/
  
  //Find discrepencies
  /*for (int row = 0; row < n; row++) {
	for (int col = 0; col < p; col++) {
	  if (C_gpu[row*n + col] != C_cpu[row*n + col]) {
		  printf("Error: C_gpu[%i] = %i, C_cpu[%i] = %i\n", row*n+col, C_gpu[row*n + col], row*n+col, C_cpu[row*n + col]);
	  }
	}
  }*/
  
  //Print performance time
  printf("CPU time was %f seconds\n", cpu_total_time);
  //printf("GPU time was %f milliseconds\n", gpu_total_time);
  
  // Clean up memory
  free(A);
  free(B);
  free(C_cpu);
  free(C_gpu);
  
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C_gpu);
  
  cudaDeviceReset();
  
  return 0;
}
