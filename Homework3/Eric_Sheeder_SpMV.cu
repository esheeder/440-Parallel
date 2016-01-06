#include <stdio.h>
#include <cstdlib>
#include <time.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
	
//This function does the multiply part of the dot-product we need to do and stores the intermediate values in products
__global__ void matrix_mult(float *products, float *b, float *data, int *indices, int n) {
	// Block index
    int bx = blockIdx.x;

    // Thread index
    int tx = threadIdx.x;
	
	//Thread ID
	int id = bx * blockDim.x + tx;
	
	//Make sure we don't go out of bounds
	if (id < n) {
	  //Store values in products matrix
      products[id] = data[id] * b[indices[id]];
	}	 
}
//This function does the sum part of the dot-product we need to do and stores the final values in c
__global__ void matrix_sum(float *c, int *row_ptr, float *products, int nr) {
	// Block index
    int bx = blockIdx.x;

    // Thread index
    int tx = threadIdx.x;
	
	//Thread ID
	int id = bx * blockDim.x + tx;
	
	//Row number is just block id since we have 1 block per row
	int row = bx;
	
	//Lane number
	int lane = id & 31;
	
	//Chunk of 32 values that will be loaded from products
	__shared__ float values[32];
	
	//Check to make sure we are in bounds
	if (row < nr) {
		//Check our row_ptr to get the values that correspond to our row
		int begin = row_ptr[row];
		int end = row_ptr[row + 1];
		//Initialize values all to 0
		values[lane] = 0;
		//Select all the values that belong to our row and store them in "values"
		if (begin + lane < end) {
			values[lane] = products[begin + lane];
		}
	}
	
	//Coaslesce all values into the last lane (31)
	if (lane >= 1) {
		values[tx] += values[tx - 1];
	}
	if (lane >= 2) {
		values[tx] += values[tx - 2];
	}
	if (lane >= 4) {
		values[tx] += values[tx - 4];
	}
	if (lane >= 8) {
		values[tx] += values[tx - 8];
	}
	if (lane >= 16) {
		values[tx] += values[tx - 16];
	}	
	//Store the dot product sums into the final array, c
	if (lane == 31) {
		c[row] += values[tx];
	}
	__syncthreads();
}

main (int argc, char **argv) {
  int *A_cpu, *A_gpu, *data;
  int n; // size of array
  srand(time(NULL)); //random numbers each time
  
  //Make sure user puts in right parameters
  if (argc !=2) {
  fprintf(fpout, "Usage: executable.exe {arraySize}, where arraySize is a positive integer")
	  exit(1);
  }
  
  n = atoi(agrv[1]);

  // Read number of rows (nr), number of columns (nc) and
  // number of elements and allocate memory for row_ptr, indices, data, b and c.
  unsigned int mem_size_matrices = (n)*sizeof(int);
  values = (int *) malloc (mem_size_row_ptr);
  A_cpu = (int *) malloc (mem_size_row_ptr);
  A_gpu = (int *) malloc (mem_size_row_ptr);

  // Read data in coordinate format and initialize sparse matrix
  for (i=0; i<n; i++) {
	int someInt = 1 + rand() % 1000;
	data[i] = someInt;
	A_cpu[i] = 0;
	A_gpu[i] = 0;
  }

  // MAIN COMPUTATION, SEQUENTIAL VERSION
  for (i=0; i<n; i++) {
	int sum = 0;
    for (j = 0; j<i; j++) {
      sum += data[j];
    }
	A_cpu[i] = sum;
  }
  
  // Allocate device memory
  int *d_A_gpu, *d_data;
  
  cudaError_t error;

  error = cudaMalloc((void **) &d_A_gpu, mem_size_matrices);
  error = cudaMalloc((void **) &d_data, mem_size_matrices);

  // copy host memory to device
  error = cudaMemcpy(d_A_gpu, A_gpu, mem_size_matrices, cudaMemcpyHostToDevice);
  error = cudaMemcpy(d_data, data, mem_size_matrices, cudaMemcpyHostToDevice);
  
  // Setup execution parameters
  //int block_size = 32;
  //Call mult with enough blocks to cover all values
  //int num_blocks = n/block_size + (n % block_size != 0);
  //matrix_mult<<<num_blocks, block_size>>>(d_products, d_b, d_data, d_indices, n);
  //Call sum with enough blocks to cover all rows
  //matrix_sum<<<nr, block_size>>>(d_c, d_row_ptr, d_products, nr);
  
  // Copy result from device to host
  //error = cudaMemcpy(c, d_c, mem_size_c, cudaMemcpyDeviceToHost);
  
  cudaDeviceSynchronize();
  
  //Print values in our 2 c vectors to output.txt
  //Code taken from http://www.tutorialspoint.com/cprogramming/c_file_io.htm
  //FILE *fpout;

  /*fpout = fopen("output.txt", "w+");
  fprintf(fpout, "index\tCPU\t\tGPU\t\tDifference\n");
  for (int i = 0; i < nr; i++) {
	  fprintf(fpout, "%i\t%f\t%f", i, c2[i], c[i]);
	  float difference = c2[i] - c[i];
	  if (difference < 0.00001) {
		  fprintf(fpout, "\t%.1f", difference);
	  } else {
		fprintf(fpout, "\t%f", difference);
	  }
	  fprintf(fpout, "\n");
  }
  fclose(fpout);*/
  
  for (int i = 0; i < n; i++) {
	  fprintf(fpout, "CPU[%i] = %i\n", i, A_cpu[i]);
  }
  
  // Clean up memory
  free(data);
  free(A_cpu);
  free(A_gpu);

  cudaFree(d_data);
  cudaFree(d_A_gpu);
  
  cudaDeviceReset();
}
