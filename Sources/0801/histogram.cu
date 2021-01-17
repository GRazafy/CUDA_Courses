#include <stdio.h>
#include <stdint.h>
#include <assert.h>

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#define MAX_BINS 4096



/*****************************/
/*    printData is used for
/*    printing the generated data   
/*****************************/
void printData(unsigned int *data, unsigned int dataSize)
{
    printf("Data generated : [");
    for (int a = 0; a < dataSize; a++)
    {
        printf("%d", data[a]);
        if (a == dataSize - 1)
        {
            printf("]\n");
        }
        if (a != dataSize - 1)
        {
            printf("-");
        }
    }
}


/*****************************/
/*    histogram is the main kernel
/*    used to calculate the histogram generated   
/*****************************/
__global__ static void histogram(unsigned int *input, unsigned int *histo, unsigned int dataSize, unsigned int binSize)
{
    int th = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ int sharedHist[];
    for (int i = threadIdx.x; i < binSize; i += blockDim.x)
    {
        sharedHist[i] = 0;
    }
    __syncthreads();
    
    for (int counterFill = th; counterFill < dataSize; counterFill += blockDim.x * gridDim.x)
    {
        atomicAdd(&sharedHist[input[counterFill]], 1);
    }
    __syncthreads();
    

    for (int j = threadIdx.x; j < binSize; j += blockDim.x)
    {
        atomicAdd(&histo[j], sharedHist[j]);
    }

}

/*****************************/
/*    compareHistograms is used for
/*    comparing the performance of the 2 methods single and multi thread    
/*****************************/
bool compareHistograms(unsigned int *firstTab, unsigned int *secondTab, int tabSize)
{
    for(int i = 0; i<tabSize; i++)
    {
        if (firstTab[p] != secondTab[p])
        {
            return false;
        }
    }
    return true;
}

/*****************************/
/*    printResult is used for
/*    printing the results   
/*****************************/
void printResult(unsigned int *res, int threadNb, unsigned int Size )
{
    printf("Result for %d threads: [", threadNb);
    for (int i = 0; i < Size; i++)
    {
        printf("%d", res[i]);
        if (i != Size - 1){
            printf("|");
        }

        if (i == Size - 1){
            print("]\n");
        }
    }
}

/*****************************/
/*    cleanHisto when finish (all the columns to 0)    
/*****************************/
__global__ static void cleanHisto(unsigned int *histo, unsigned int binSize)
{
    for (int i = threadIdx.x; i < binSize; i += blockDim.x)
    {
        histo[i] = 0;
    }
    __syncthreads();

}

void wrapper(unsigned int dataSize, unsigned int binSize, int showData, int threadNb, int blockCount)
{

    unsigned int *histo = NULL;
    unsigned int *histo_single = NULL;
    unsigned int *d_histo = NULL;
    unsigned int *data = NULL;
    unsigned int *d_data = NULL;
    cudaEvent_t start;
    cudaEvent_t start_single;
    cudaEvent_t stop;
    cudaEvent_t stop_single;

    data = (unsigned int *)malloc(dataSize * sizeof(unsigned int));
    histo = (unsigned int *)malloc(binSize * sizeof(unsigned int));
    histo_single = (unsigned int *)malloc(binSize * sizeof(unsigned int));
    //generating some data
    srand(time(NULL));
    for (int i = 0; i < dataSize; i++){
        data[i] = rand() % binSize;
    }
    printf("Done\n");

    //showing the data if the user wants to see it
    if (showData == 1)
    {
        printData(data, dataSize);
    }

    checkCudaErrors(cudaMalloc((void **)&d_histo, sizeof(unsigned int) * binSize));
    checkCudaErrors(cudaMalloc((void **)&d_data, sizeof(unsigned int) * dataSize));

    checkCudaErrors(cudaMemcpy(d_data, data, sizeof(unsigned int) * dataSize, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start, NULL));

    //lauching the kernel with multiple thread
    histogram<<<blockCount, threadNb,sizeof(unsigned int) * binSize>>>(d_data, d_histo, dataSize, binSize);
    cudaDeviceSynchronize();

    printf("End of the kernel, fetching the results :\n");
    checkCudaErrors(cudaMemcpy(histo, d_histo, sizeof(unsigned int) * binSize, cudaMemcpyDeviceToHost));

    //creatiion of events for time measuring
    checkCudaErrors(cudaEventRecord(stop, NULL));
    checkCudaErrors(cudaEventSynchronize(stop));

    checkCudaErrors(cudaEventCreate(&start_single));
    checkCudaErrors(cudaEventCreate(&stop_single));
    checkCudaErrors(cudaEventRecord(start_single, NULL));

    //cleaning the histogram
    cleanHisto<<<1, threadNb>>>(d_histo, binSize);
    cudaDeviceSynchronize();

    //lauching the kernel for a single thread for comparaison
    histogram<<<1, 1,sizeof(unsigned int) * binSize>>>(d_data, d_histo, dataSize, binSize);
    cudaDeviceSynchronize();

    checkCudaErrors(cudaMemcpy(histo_single, d_histo, sizeof(unsigned int) * binSize, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaEventRecord(stop_single, NULL));
    checkCudaErrors(cudaEventSynchronize(stop_single));
    float msecTotal = 0.0f;
    float msecTotal_single = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal_single, start_single, stop_single));
    double gigaFlops = (dataSize * 1.0e-9f) / (msecTotal / 1000.0f);
    double gigaFlops_single = (dataSize * 1.0e-9f) / (msecTotal_single / 1000.0f);

    // Print the histograms if the parameter
    if (showData == 1)
    {
        printResult(histo, threadNb, binSize);
        printResult(histo_single, 1, binSize);
    }

    // compareHistograms the results of the two histograms
    if (compareHistograms(histo, histo_single, binSize))
    {
        printf("histograms matched");
    }
    else
    {
        printf("Something went wrong the histograms doesn't matched !!");
    }
    // Print performances
    printf("%d threads :\nCuda processing time = %.3fms, \n Perf = %.3f Gflops\n",threadNb, msecTotal, gigaFlops);
    printf("1 thread :\nCuda processing time = %.3fms, \n Perf = %.3f Gflops\n", msecTotal_single, gigaFlops_single);
    checkCudaErrors(cudaFree(d_data));
    checkCudaErrors(cudaFree(d_histo));
    free(histo);
    free(histo_single);
    free(data);

}

int main(int argc, char **argv)
{
    int print = 0;
    unsigned int binSize = MAX_BINS;
    unsigned long long bins_max_size = 256;

    char *dataSize = NULL;
    cudaDeviceProp cudaprop;

    // retrieve device
    int dev = findCudaDevice(argc, (const char **)argv);
    cudaGetDeviceProperties(&cudaprop, dev);
    //Retrieving parameters
    if (checkCmdLineFlag(argc, (const char **)argv, "size"))
    {
        getCmdLineArgumentString(argc, (const char **)argv, "size", &dataSize);
        ds = atoll(dataSize);
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "displayData"))
    {
        print = 1;
    }

    printf("Data Size is: %d \n", ds);
    //Max is 2^32 as asked
    if (ds >= 4294967296 || ds == 0) {
        printf("Error: Data size > 4,294,967,296");
        exit(EXIT_FAILURE);
    }
    //Defining the number of threads to follow the need (ds) with max value 256 (multiple of 32) and < 1024
    int nbThread = min((int)ds, 256);
    printf("nb thread: %d \n", nbThread);
    //Defining the number of blocks to follow the need (if ds = 500 only 2 blocks) with max value a multiple of 30
    int nbBlock =  min(((int)ds/256),18000);
    //if the data size is below 256 we still have to have atleast 1 block
    if (nbBlock == 0) nbBlock = 1;
    printf("nbblock: %d \n", nbBlock);
    wrapper(ds, binSize, print, nbThread, nbBlock);
    return EXIT_SUCCESS;
}