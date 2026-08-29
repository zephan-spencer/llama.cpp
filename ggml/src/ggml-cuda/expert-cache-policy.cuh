#pragma once

#include "ggml-backend-moe-cache.h"
#include "expert-cache-route.cuh"

struct ggml_cuda_expert_cache_policy_state {
    int32_t * fill_expert;
    int32_t * fill_slot;
    int32_t * expert_to_cache;
    int32_t * cache_to_expert;

    uint32_t * n_fill;
    uint32_t * n_active;

    void * data;
};

struct ggml_cuda_expert_cache_policy_i {
    size_t (*state_size)(uint32_t n_expert, uint32_t n_cache);
    void (*initialize)(void * state, uint32_t n_expert, uint32_t n_cache);
    void (*resolve)(const ggml_cuda_expert_selector_plan &      selectors,
                    const ggml_cuda_expert_cache_policy_state & state,
                    int                                         n_experts,
                    int                                         n_cache,
                    cudaStream_t                                stream);
};

const ggml_cuda_expert_cache_policy_i * ggml_cuda_expert_cache_policy(
    enum ggml_backend_moe_cache_policy policy);
