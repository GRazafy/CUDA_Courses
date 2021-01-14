//basic cpp includes
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <iostream>
//Last one to accumulate the results via accumulate function
#include <numeric>
//cuda includes
#include "cuda_runtime.h"
#include <cuda.h>
#include <helper_cuda.h>
#include <cassert>



__global__ static void reduction_kernel(const float *input, float *output, int tabSize)
{
    extern __shared__ float resSum[];

    unsigned int th = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    int index;

    /*******  Handle of exception values   **********/

    if (i<tabSize){
    	resSum[th] = input[i];
    }else {
    	resSum[th] = 0;
    }
    if (i+blockDim.x*gridDim.x < tabSize){
    	resSum[th+blockDim.x] = input[i+blockDim.x*gridDim.x];
    }else {
    	resSum[th+blockDim.x] = 0;
    }
   /***************************************************/
    __syncthreads();
    for (int stride = 1; stride < blockDim.x*2; stride *= 2)
    {
        //Sync at the beginning and not the end!
        __syncthreads();
        index = 2*stride*th;
        if (index < blockDim.x*2)
        {
            resSum[index] += resSum[index + stride];
        }
    }
    __syncthreads();
    if (th == 0){ //Committing to vram to retrieve the results of each block on the host
        output[blockIdx.x] = resSum[0];
    }
}
void wrapper(int lenTab ,float *input, int NUM_THREADS, int NUM_BLOCKS)
{
    float *dinput = NULL;
    float *doutput = NULL;
    float *output = NULL;
    //Allocating memory for the output data
    output = (float *)malloc(NUM_BLOCKS * sizeof(float));
    checkCudaErrors(cudaMalloc((void **)&dinput, sizeof(float) * lenTab));
    checkCudaErrors(cudaMalloc((void **)&doutput, sizeof(float) * NUM_BLOCKS));
    checkCudaErrors(cudaMemcpy(dinput, input, sizeof(float) * lenTab , cudaMemcpyHostToDevice));
     //Allocating CUDA events that we'll use for timing
    cudaEvent_t start;
    checkCudaErrors(cudaEventCreate(&start));
    cudaEvent_t stop;
    checkCudaErrors(cudaEventCreate(&stop));
    //Record start event
    checkCudaErrors(cudaEventRecord(start, NULL));

    reduction_kernel<<<NUM_BLOCKS, NUM_THREADS, sizeof(float) * 2 * NUM_THREADS >>>(dinput, doutput, lenTab);
    cudaDeviceSynchronize();

    //Record stop event
    checkCudaErrors(cudaEventRecord(stop, NULL));
    //Wait for the stp event to complete
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaMemcpy(output, doutput, sizeof(float) * NUM_BLOCKS, cudaMemcpyDeviceToHost));
    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
    float totalRes = std::accumulate(output, output + NUM_BLOCKS, int(0));
    printf("The result of the multithreaded function is: %f \n", totalRes);
    printf("Elapsed Time for reduction function to complete is : %f msec \n", msecTotal);

    checkCudaErrors(cudaFree(dinput));
    checkCudaErrors(cudaFree(doutput));

     //Allocating CUDA events that we'll use for timing
    cudaEvent_t start_singlethread;
    checkCudaErrors(cudaEventCreate(&start_singlethread));
    cudaEvent_t stop_singlethread;
    checkCudaErrors(cudaEventCreate(&stop_singlethread));
    //Record start event
    checkCudaErrors(cudaEventRecord(start_singlethread, NULL));

    
    //computing the result for a single thread and comparing to the multithreaded result
    int resultSingleThread = 0;
    for (int i = 0 ; i < lenTab; i++)
    {
          resultSingleThread += input[i];
    }
   
    //Record stop event
    checkCudaErrors(cudaEventRecord(stop_singlethread, NULL));
    //Wait for the stp event to complete
    checkCudaErrors(cudaEventSynchronize(stop_singlethread));
    float msecTotal_singlethread = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal_singlethread, start_singlethread, stop_singlethread));


    printf("The result of the thread function is: %d \n", resultSingleThread);
    printf("Elapsed Time for reduction function to complete is : %f msec \n", msecTotal_singlethread);
    free(output);
}

int main(int argc, char **argv)
{
    int lenTab;
    if (checkCmdLineFlag(argc, (const char **)argv, "tabSize")) {
        lenTab = getCmdLineArgumentInt(argc, (const char **)argv, "tabSize");
    }
    
    int NUM_THREADS =  1024;
    //Multiple of the sm number (30)
    int NUM_BLOCKS = min((lenTab/NUM_THREADS), 65520);
    //atleast one block
    if (NUM_BLOCKS == 0){ NUM_BLOCKS = 1;}
    int dev = findCudaDevice(argc, (const char **)argv);
    float *input = NULL;
    //Allocating memory for the input data
    input = (float *)malloc(lenTab * sizeof(float));

    //Generating the values of the array
    for (int i = 0; i < lenTab; i++)
    {
       input[i] = rand() % 10;
    }
    wrapper(lenTab,input, NUM_THREADS, NUM_BLOCKS);
    free(input);
    return EXIT_SUCCESS;
}
