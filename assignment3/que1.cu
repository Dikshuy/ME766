#include<bits/stdc++.h>

using namespace std;

#define BLOCK_SIZE 16

__global__ void matrix_multiplication(int *dev_a, int *dev_b, int *dev_c, int n){
    __shared__ int tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int tile_b[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y*BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x*BLOCK_SIZE + threadIdx.x;
    int temp = 0;
    int idx;

    for (int i=0; i<gridDim.x; ++i){
        idx = row*n + i*BLOCK_SIZE + threadIdx.x;
        if (idx >= n*n){
            tile_a[threadIdx.y][threadIdx.x] = 0;
        }
        else{
            tile_a[threadIdx.y][threadIdx.x] = dev_a[idx];
        }

        idx = (i*BLOCK_SIZE + threadIdx.y)*n + col;
        if (idx >= n*n){
            tile_b[threadIdx.y][threadIdx.x] = 0;
        }
        else{
            tile_b[threadIdx.y][threadIdx.x] = dev_b[idx];
        }
        __syncthreads();

        for (int j=0; j<BLOCK_SIZE; ++j){
            temp += tile_a[threadIdx.y][j]*tile_b[j][threadIdx.x];
        }
        __syncthreads();
    }
    if (row<n && col<n){
        dev_c[row*n+col] = temp;
    }
}

int main(int argc, char const *argv[]){
    int n;
    srand(1);
    int *a, *b, *c;
    n=10000;
    cudaMallocHost((void **) &a, sizeof(int)*n*n);
    cudaMallocHost((void **) &b, sizeof(int)*n*n);
    cudaMallocHost((void **) &c, sizeof(int)*n*n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            a[i * n + j] = rand() % n;
            b[i * n + j] = rand() % n;
        }
    }

    float time_taken;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start,0);

    int *dev_a, *dev_b, *dev_c;
    cudaMalloc((void **) &dev_a, sizeof(int)*n*n);
    cudaMalloc((void **) &dev_b, sizeof(int)*n*n);
    cudaMalloc((void **) &dev_c, sizeof(int)*n*n);

    cudaMemcpy(dev_a, a, sizeof(int)*n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, sizeof(int)*n*n, cudaMemcpyHostToDevice);

    unsigned int grid_rows = (n+BLOCK_SIZE-1)/BLOCK_SIZE;
    unsigned int grid_cols = (n+BLOCK_SIZE-1)/BLOCK_SIZE;

    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    matrix_multiplication<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, n);

    cudaMemcpy(c, dev_c, sizeof(int)*n*n, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_taken, start, stop);

    printf("Time elapsed in matrix multiplication on GPU: %f ms.\n",time_taken);

    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    return 0;
}