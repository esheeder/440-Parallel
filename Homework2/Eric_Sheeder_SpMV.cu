#include <stdio.h>
#include <cstdlib>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
//#include <helper_functions.h>
	
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
  FILE *fp;
  char line[1024]; 
  int *row_ptr, *indices;
  float *data, *b, *c, *c2;
  int i,j;
  int n; // number of nonzero elements in data
  int nr; // number of rows in matrix
  int nc; // number of columns in matrix

  // Open input file and read to end of comments
  if (argc !=2) exit(1); 

  if ((fp = fopen(argv[1], "r")) == NULL) {
    exit(1);
  }

  fgets(line, 128, fp);
  while (line[0] == '%') {
    fgets(line, 128, fp); 
  }

  // Read number of rows (nr), number of columns (nc) and
  // number of elements and allocate memory for row_ptr, indices, data, b and c.
  sscanf(line,"%d %d %d\n", &nr, &nc, &n);
  unsigned int mem_size_row_ptr = (nr+1)*sizeof(int);
  row_ptr = (int *) malloc (mem_size_row_ptr);
  unsigned int mem_size_indices = n*sizeof(int);
  indices = (int *) malloc(mem_size_indices);
  unsigned int mem_size_data = n*sizeof(float);
  data = (float *) malloc(mem_size_data);
  unsigned int mem_size_b = nc*sizeof(float);
  b = (float *) malloc(mem_size_b);
  unsigned int mem_size_c = nr*sizeof(float);
  c = (float *) malloc(mem_size_c);
  c2 = (float *) malloc(mem_size_c);

  // Read data in coordinate format and initialize sparse matrix
  int lastr=0;
  for (i=0; i<n; i++) {
    int r;
    fscanf(fp,"%d %d %f\n", &r, &(indices[i]), &(data[i]));  
    indices[i]--;  // start numbering at 0
    if (r!=lastr) { 
      row_ptr[r-1] = i; 
      lastr = r; 
    }
  }
  row_ptr[nr] = n;

  // initialize c to 0 and b with random data  
  for (i=0; i<nr; i++) {
    c[i] = 0.0;
	c2[i] = 0.0;
  }

  for (i=0; i<nc; i++) {
    b[i] = (float) rand()/1111111111;
  }

  // MAIN COMPUTATION, SEQUENTIAL VERSION
  for (i=0; i<nr; i++) {                                                      
    for (j = row_ptr[i]; j<row_ptr[i+1]; j++) {
      c2[i] = c2[i] + data[j] * b[indices[j]];
	  //printf("j: %i \n", j);
	  //printf("indices: %i \n", indices[j]);
    }
  }
  
  // Allocate device memory
  float *d_data, *d_products, *d_b, *d_c;
  int *d_row_ptr, *d_indices;
  
  cudaError_t error;

  error = cudaMalloc((void **) &d_row_ptr, mem_size_row_ptr);
  error = cudaMalloc((void **) &d_indices, mem_size_indices);
  error = cudaMalloc((void **) &d_data, mem_size_data);
  error = cudaMalloc((void **) &d_products, mem_size_data);
  error = cudaMalloc((void **) &d_b, mem_size_b);
  error = cudaMalloc((void **) &d_c, mem_size_c);

  // copy host memory to device
  error = cudaMemcpy(d_row_ptr, row_ptr, mem_size_row_ptr, cudaMemcpyHostToDevice);
  error = cudaMemcpy(d_indices, indices, mem_size_indices, cudaMemcpyHostToDevice);
  error = cudaMemcpy(d_data, data, mem_size_data, cudaMemcpyHostToDevice);
  error = cudaMemcpy(d_b, b, mem_size_b, cudaMemcpyHostToDevice);
  error = cudaMemcpy(d_c, c, mem_size_c, cudaMemcpyHostToDevice);
  
  // Setup execution parameters
  int block_size = 32;
  //Call mult with enough blocks to cover all values
  int num_blocks = n/block_size + (n % block_size != 0);
  matrix_mult<<<num_blocks, block_size>>>(d_products, d_b, d_data, d_indices, n);
  //Call sum with enough blocks to cover all rows
  matrix_sum<<<nr, block_size>>>(d_c, d_row_ptr, d_products, nr);
  
  // Copy result from device to host
  error = cudaMemcpy(c, d_c, mem_size_c, cudaMemcpyDeviceToHost);
  
  cudaDeviceSynchronize();
  
  //Print values in our 2 c vectors to output.txt
  //Code taken from http://www.tutorialspoint.com/cprogramming/c_file_io.htm
  FILE *fpout;

  fpout = fopen("output.txt", "w+");
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
  fclose(fpout);
  
  // Clean up memory
  free(row_ptr);
  free(indices);
  free(data);
  free(c2);
  free(c);
  free(b);
  cudaFree(d_data);
  cudaFree(d_products);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaFree(d_row_ptr);
  cudaFree(d_indices);
  
  cudaDeviceReset();
}
