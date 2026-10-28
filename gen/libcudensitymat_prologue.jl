# cuDensityMat takes CUDA runtime stream handles, which are interchangeable with
# the driver-API `CUstream` that CUDACore uses.
const cudaStream_t = CUstream
