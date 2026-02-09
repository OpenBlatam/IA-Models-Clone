/*
 * Inference Kernel - Low-level optimized inference operations
 * Uses CUDA/OpenCL for GPU acceleration
 */

#ifndef INFERENCE_KERNEL_H
#define INFERENCE_KERNEL_H

#include <vector>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// CUDA kernel for matrix multiplication (if CUDA available)
#ifdef CUDA_AVAILABLE
void cuda_matmul(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
);
#endif

// Optimized attention computation
void compute_attention(
    const float* query,
    const float* key,
    const float* value,
    float* output,
    int batch_size,
    int seq_len,
    int head_dim,
    int num_heads
);

// Memory management
void* allocate_gpu_memory(size_t size);
void free_gpu_memory(void* ptr);
void copy_to_gpu(void* dst, const void* src, size_t size);
void copy_from_gpu(void* dst, const void* src, size_t size);

#ifdef __cplusplus
}
#endif

#endif // INFERENCE_KERNEL_H












