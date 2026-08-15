#pragma once

#include "mmid.cuh"

struct ggml_cuda_expert_cache_policy_state {
    int32_t *  fill_expert;
    int32_t *  fill_slot;
    int32_t *  expert_to_cache;
    int32_t *  cache_to_expert;
    uint64_t * last_used;
    uint64_t * protected_epoch;
    uint64_t * pending_priority_epoch;
    uint64_t * use_clock;

    const int32_t * epoch;

    uint32_t * n_fill;
    uint32_t * resolve_active;
    uint32_t * policy_flags;
};

void ggml_cuda_apply_expert_cache_policy(const ggml_cuda_expert_selector_plan &     selectors,
                                         const ggml_cuda_expert_cache_policy_state & state,
                                         int                                         n_experts,
                                         int                                         n_cache,
                                         cudaStream_t                                stream);
