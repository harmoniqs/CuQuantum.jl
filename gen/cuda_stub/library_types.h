/* Minimal stand-in for the CUDA toolkit's library_types.h, sufficient to parse
 * cudensitymat.h with Clang.jl. None of these declarations are emitted: the
 * corresponding Julia types come from CUDACore (see the prologue). Set
 * CUDA_INCLUDE_DIR to wrap against a real CUDA toolkit instead. */
#pragma once
typedef enum cudaDataType_t { CUDA_R_32F = 0, CUDA_R_64F = 1, CUDA_C_32F = 4, CUDA_C_64F = 5 } cudaDataType_t;
typedef enum cudaDataType_t cudaDataType;
typedef enum libraryPropertyType_t { MAJOR_VERSION, MINOR_VERSION, PATCH_LEVEL } libraryPropertyType_t;
typedef enum libraryPropertyType_t libraryPropertyType;
