#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16
#define GGML_CUDA_EXPERT_CACHE_MAX_WEIGHTS 4
#define GGML_CUDA_EXPERT_CACHE_MAGIC 0x4558504341434845ULL
#define GGML_CUDA_EXPERT_CACHE_VERSION 2

struct ggml_backend_cuda_expert_cache_stats {
    uint64_t resolve_calls;
    uint64_t cache_hits;
    uint64_t cache_misses;
    uint64_t evictions;
    uint64_t h2d_bytes;
    uint64_t fill_ticks;
};

struct ggml_backend_cuda_expert_cache_state {
    uint64_t use_clock;
    struct ggml_backend_cuda_expert_cache_stats stats;
    uint64_t fill_start_ticks;
    uint32_t n_active;
    uint32_t n_resident;
    uint32_t n_fills;
    uint32_t n_evictions;
    uint32_t copy_blocks_done;
};

struct ggml_backend_cuda_expert_cache_weight {
    const void * host_data;
    void * device_data;
    uint64_t host_offset;
    uint64_t expert_size;
};

struct ggml_backend_cuda_expert_cache_desc {
    uint64_t magic;
    uint32_t version;
    uint32_t n_expert;
    uint32_t n_cache;
    uint32_t n_weights;
    uint64_t expert_to_cache_offset;
    uint64_t cache_to_expert_offset;
    uint64_t last_used_offset;
    uint64_t route_indices_offset;
    uint64_t expert_bounds_offset;
    uint64_t expert_order_offset;
    uint64_t fill_expert_offset;
    uint64_t fill_slot_offset;
    uint64_t state_size;
    uint64_t wall_clock_hz;
    struct ggml_backend_cuda_expert_cache_weight weights[GGML_CUDA_EXPERT_CACHE_MAX_WEIGHTS];
};

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

#ifdef  __cplusplus
}
#endif
