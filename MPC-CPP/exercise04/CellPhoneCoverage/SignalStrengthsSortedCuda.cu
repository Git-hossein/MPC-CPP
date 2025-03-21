#include "SignalStrengthsSortedCuda.h"

#include "CellPhoneCoverage.h"
#include "CudaArray.h"
#include "Helpers.h"


#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>
#include <thread>

using namespace std;

// "Smart" CUDA implementation which computes signal strengths
//
// First, all transmitters are sorted into buckets
// Then, all receivers are sorted into buckets
// Then, receivers only compute signal strength against transmitters in nearby buckets
//
// This multi-step algorithm makes the signal strength computation scale much
//  better to high number of transmitters/receivers

struct Bucket
{
    int startIndex; // Start of bucket within array
    int numElements; // Number of elements in bucket
};

///////////////////////////////////////////////////////////////////////////////////////////////
//
// No-operation sorting kernel
//
// This takes in an unordered set, and builds a dummy bucket representation around it
// It does not perform any actual sorting!
//
// This kernel must be launched with a 1,1 configuration (1 grid block, 1 thread).

static __global__ void noSortKernel(const Position* inputPositions, int numInputPositions,
                                    Position* outputPositions, Bucket* outputBuckets)
{
    int numBuckets = BucketsPerAxis * BucketsPerAxis;

    // Copy contents of input positions into output positions

    for (int i = 0; i < numInputPositions; ++i)
        outputPositions[i] = inputPositions[i];

    // Set up the set of buckets to cover the output positions evenly

    for (int i = 0; i < numBuckets; i++)
    {
        Bucket& bucket = outputBuckets[i];

        bucket.startIndex = numInputPositions * i / numBuckets;
        bucket.numElements = (numInputPositions * (i + 1) / numBuckets) - bucket.startIndex;
    }
}

// !!! missing !!!
// Kernels needed for sortPositionsIntoBuckets(...)

__global__ void bucketHistogram(Position* inputPositions, int* output, int n)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x ;
    
    if (i < 256) {
        output[i] = 0;}
    __syncthreads();

    if (i < n){
        int xValue = min((int) (inputPositions[i].x * 16.f), 15);
        int yValue = min((int) (inputPositions[i].y * 16.f), 15);
        int bucketNr = xValue * BucketsPerAxis + yValue;
        atomicAdd(&output[bucketNr], 1) ;
    }
}


__global__ void prefixSumHistogram(int* histogram, int* rHistogram)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x ;
    const int grid = BucketsPerAxis*BucketsPerAxis;
    __shared__ int temp[BucketsPerAxis * BucketsPerAxis];
    temp[0] = 0;
    temp[i+1] = histogram[i];
    __syncthreads();

    // reduce
    for (int stride = 1; stride < grid ; stride *= 2) {
        int index1 = (stride-1) + (threadIdx.x * 2 *stride) + 1 ; // +1 to shift to right
        int index2 = index1 + stride;

        if (index1 < grid) {
            temp[index2] += temp[index1];
        }
        __syncthreads();
    }

    // spread
    for (int stride = grid/2; stride > 0 ; stride /= 2) {
        int index1 = 2*threadIdx.x*stride ;
        int index2 = index1 + stride;

        if (index1 < grid) {
            temp[index2] += temp[index1];
        }
        __syncthreads();
    }

    rHistogram[i] = temp[i];
}


__global__ void loopPrefixSum(int* histogram, Bucket* Buckets)
{
    Buckets[0].startIndex = 0;
    for (int i = 1; i < BucketsPerAxis * BucketsPerAxis; i++) {
        Buckets[i].startIndex = Buckets[i-1].startIndex + histogram[i-1];
    }

}

__global__ void assignToBuckets(Position* inputPositions, Bucket* Buckets, Position* outputPositions, int n)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x ;
    if (i < 256) {
        Buckets[i].numElements = 0;}
    __syncthreads();
    // __shared__ float temp[16];
    
    if (i < n) {
        int xValue = min((int) (inputPositions[i].x * 16.f), 15);
        int yValue = min((int) (inputPositions[i].y * 16.f), 15);
        int bucketNr = xValue * BucketsPerAxis + yValue;
        int offset = atomicAdd(&Buckets[bucketNr].numElements, 1) ;
        outputPositions[offset + Buckets[bucketNr].startIndex] = inputPositions[i];
    }

}

///////////////////////////////////////////////////////////////////////////////////////////////
//
// Sort a set of positions into a set of buckets
//
// Given a set of input positions, these will be re-ordered such that
//  each range of elements in the output array belong to the same bucket.
// The list of buckets that is output describes where each such range begins
//  and ends in the re-ordered position array.

static void sortPositionsIntoBuckets(CudaArray<Position>& cudaInputPositions,
                                     CudaArray<Position>& cudaOutputPositions,
                                     CudaArray<Bucket>& cudaOutputPositionBuckets)
{
    // Bucket sorting with "Counting Sort" is a multi-phase process:
    //
    // 1. Determine how many of the input elements should end up in each bucket (build a histogram)
    //
    // 2. Given the histogram, compute where in the output array that each bucket begins, and how
    // large it is
    //    (perform prefix summation over the histogram)
    //
    // 3. Given the start of each bucket within the output array, scatter elements from the input
    //    array into the output array
    //
    // Your new sort implementation should be able to handle at least 10 million entries, and
    //  run in reasonable time (the reference implementations does the job in 200 milliseconds).

    //=================  Your code here =====================================
    // !!! missing !!!

    // Instead of sorting, we will now run a dummy kernel that just duplicates the
    //  output positions, and constructs a set of dummy buckets. This is just so that
    //  the test program will not crash when you try to run it.
    //
    // This kernel is run single-threaded because it is throw-away code where performance
    //  does not matter; after all, the purpose of the lab is to replace it with a
    //  proper sort algorithm instead!

    //========== Remove this code when you begin to implement your own sorting algorithm ==========

    int gridsize = (cudaInputPositions.size()) /(BucketsPerAxis * BucketsPerAxis) + 1;
    const int numThreads = BucketsPerAxis * BucketsPerAxis;

    int* histogram;
    cudaMalloc(&histogram, numThreads*sizeof(int));
    bucketHistogram<<<gridsize, numThreads>>>(cudaInputPositions.cudaArray(), histogram, cudaInputPositions.size());
    // prefixSumHistogram<<<1, numThreads>>>(histogram, rHistogram);
    loopPrefixSum<<<1,1>>>(histogram, cudaOutputPositionBuckets.cudaArray());
    assignToBuckets<<<gridsize, numThreads>>>(cudaInputPositions.cudaArray(), cudaOutputPositionBuckets.cudaArray(), cudaOutputPositions.cudaArray(), cudaInputPositions.size());
    cudaFree(histogram);
    
}

///////////////////////////////////////////////////////////////////////////////////////////////
//
// Go through all transmitters in one bucket, find highest signal strength
// Return highest strength (or the old value, if that was higher)

static __device__ float scanBucket(const Position* transmitters, int numTransmitters,
                                   const Position& receiver, float bestSignalStrength)
{
    for (int transmitterIndex = 0; transmitterIndex < numTransmitters; ++transmitterIndex)
    {
        const Position& transmitter = transmitters[transmitterIndex];

        float strength = signalStrength(transmitter, receiver);

        if (bestSignalStrength < strength)
            bestSignalStrength = strength;
    }

    return bestSignalStrength;
}

///////////////////////////////////////////////////////////////////////////////////////////////
//
// Calculate signal strength for all receivers

static __global__ void calculateSignalStrengthsSortedKernel(const Position* transmitters,
                                                            const Bucket* transmitterBuckets,
                                                            const Position* receivers,
                                                            const Bucket* receiverBuckets,
                                                            float* signalStrengths)
{
    // Determine which bucket the current grid block is processing

    int receiverBucketIndexX = blockIdx.x;
    int receiverBucketIndexY = blockIdx.y;

    int receiverBucketIndex = receiverBucketIndexY * BucketsPerAxis + receiverBucketIndexX;

    const Bucket& receiverBucket = receiverBuckets[receiverBucketIndex];

    int receiverStartIndex = receiverBucket.startIndex;
    int numReceivers = receiverBucket.numElements;

    // Distribute available receivers over the set of available threads

    for (int receiverIndex = threadIdx.x; receiverIndex < numReceivers; receiverIndex += blockDim.x)
    {
        // Locate current receiver within the current bucket

        const Position& receiver = receivers[receiverStartIndex + receiverIndex];
        float& finalStrength = signalStrengths[receiverStartIndex + receiverIndex];

        float bestSignalStrength = 0.f;

        // Scan all buckets in the 3x3 region enclosing the receiver's bucket index

        for (int transmitterBucketIndexY = receiverBucketIndexY - 1;
             transmitterBucketIndexY < receiverBucketIndexY + 2; ++transmitterBucketIndexY)
            for (int transmitterBucketIndexX = receiverBucketIndexX - 1;
                 transmitterBucketIndexX < receiverBucketIndexX + 2; ++transmitterBucketIndexX)
            {
                // Only process bucket if its index is within [0, BucketsPerAxis - 1] along each
                // axis

                if (transmitterBucketIndexX >= 0 && transmitterBucketIndexX < BucketsPerAxis
                    && transmitterBucketIndexY >= 0 && transmitterBucketIndexY < BucketsPerAxis)
                {
                    // Scan bucket for a potential new "highest signal strength"

                    int transmitterBucketIndex =
                        transmitterBucketIndexY * BucketsPerAxis + transmitterBucketIndexX;
                    int transmitterStartIndex =
                        transmitterBuckets[transmitterBucketIndex].startIndex;
                    int numTransmitters = transmitterBuckets[transmitterBucketIndex].numElements;
                    bestSignalStrength = scanBucket(&transmitters[transmitterStartIndex],
                                                    numTransmitters, receiver, bestSignalStrength);
                }
            }

        // Store out the highest signal strength found for the receiver

        finalStrength = bestSignalStrength;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////

void calculateSignalStrengthsSortedCuda(const PositionList& cpuTransmitters,
                                        const PositionList& cpuReceivers,
                                        SignalStrengthList& cpuSignalStrengths)
{
    int numBuckets = BucketsPerAxis * BucketsPerAxis;

    // Copy input positions to device memory

    CudaArray<Position> cudaTempTransmitters(cpuTransmitters.size());
    cudaTempTransmitters.copyToCuda(&(*cpuTransmitters.begin()));

    CudaArray<Position> cudaTempReceivers(cpuReceivers.size());
    cudaTempReceivers.copyToCuda(&(*cpuReceivers.begin()));

    // Allocate device memory for sorted arrays

    CudaArray<Position> cudaTransmitters(cpuTransmitters.size());
    CudaArray<Bucket> cudaTransmitterBuckets(numBuckets);

    CudaArray<Position> cudaReceivers(cpuReceivers.size());
    CudaArray<Bucket> cudaReceiverBuckets(numBuckets);

    // Sort transmitters and receivers into buckets

    sortPositionsIntoBuckets(cudaTempTransmitters, cudaTransmitters, cudaTransmitterBuckets);
    sortPositionsIntoBuckets(cudaTempReceivers, cudaReceivers, cudaReceiverBuckets);

    // Perform signal strength computation
    CudaArray<float> cudaSignalStrengths(cpuReceivers.size());

    int numThreads = 256;
    dim3 grid = dim3(BucketsPerAxis, BucketsPerAxis);

    calculateSignalStrengthsSortedKernel<<<grid, numThreads>>>(
        cudaTransmitters.cudaArray(), cudaTransmitterBuckets.cudaArray(), cudaReceivers.cudaArray(),
        cudaReceiverBuckets.cudaArray(), cudaSignalStrengths.cudaArray());

    // Copy results back to host memory
    cpuSignalStrengths.resize(cudaSignalStrengths.size());
    cudaSignalStrengths.copyFromCuda(&(*cpuSignalStrengths.begin()));
}
