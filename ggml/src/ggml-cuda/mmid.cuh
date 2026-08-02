#pragma once

#include <cstdint>

struct ggml_cuda_expert_source_view {
    const int32_t * selectors;
    int64_t selector_stride;
    uint32_t selector_width;
    const char * host_data;
    int64_t host_stride;
    uint32_t n_expert;
};

struct ggml_cuda_expert_plan {
    int32_t * ids_src;
    int32_t * ids_dst;
    int32_t * expert_bounds;

    int32_t * expert_order;
    int32_t * fill_expert;
    int32_t * fill_slot;
    int32_t * cache_ids;
    const uint8_t * policy;

    int32_t * expert_to_cache;
    int32_t * cache_to_expert;
    uint64_t * last_used;
    uint64_t * use_clock;

    uint32_t * n_active;
    uint32_t * n_resident;
    uint32_t * n_miss;
    uint32_t * n_fill;
    uint32_t * n_evictions;
    uint32_t * n_read_only;
    uint32_t * n_streamed;
    uint32_t * n_host_experts;
};

void ggml_cuda_launch_expert_plan(
        const int32_t * ids,
        const ggml_cuda_expert_plan & plan,
        int n_experts,
        int n_tokens,
        int n_expert_used,
        int nchannels_y,
        int si1,
        int sis1,
        bool write_inverse,
        int n_cache,
        cudaStream_t stream);

void ggml_cuda_launch_expert_cache_plan(
        const int32_t * ids,
        const ggml_cuda_expert_plan & plan,
        int n_experts,
        int n_tokens,
        int n_expert_used,
        int n_cache,
        cudaStream_t stream);
