#pragma once

#include "ggml-backend-moe-cache.h"

#define GGML_CUDA_EXPERT_CACHE_MAGIC   0x4558504341434845ULL
#define GGML_CUDA_EXPERT_SOURCE_MAGIC  0x4558505352430001ULL
#define GGML_CUDA_EXPERT_PLAN_MAGIC    0x455850504C414E01ULL

struct ggml_backend_cuda_expert_cache_state {
    uint64_t                     use_clock;
    uint32_t                     n_fills;
    uint32_t                     resolve_active;
    uint32_t                     policy_flags;
};

struct ggml_backend_cuda_expert_cache_weight {
    const void * host_data;
    void *       device_data;
    uint64_t     host_offset;
    uint64_t     expert_size;
};
struct ggml_backend_moe_cache;

struct ggml_backend_cuda_expert_source {
    uint64_t                                      magic;
    uint32_t                                      n_expert;
    struct ggml_backend_moe_cache *               cache;
    const ggml_backend_cuda_expert_cache_weight * weight;
};

struct ggml_backend_cuda_expert_cache_desc {
    uint64_t                              magic;
    uint32_t                              n_expert;
    uint32_t                              n_cache;
    uint32_t                              n_weights;
    uint64_t                              expert_to_cache_offset;
    uint64_t                              cache_to_expert_offset;
    uint64_t                              last_used_offset;
    uint64_t                              protected_epoch_offset;
    uint64_t                              pending_priority_epoch_offset;
    uint64_t                              fill_expert_offset;
    uint64_t                              fill_slot_offset;
    uint64_t                              state_size;
    ggml_backend_cuda_expert_cache_weight weights[GGML_BACKEND_MOE_CACHE_MAX_WEIGHTS];
};

struct ggml_backend_cuda_expert_layout {
    uint64_t expert_to_cache_offset;
    uint64_t cache_to_expert_offset;
    uint64_t last_used_offset;
    uint64_t protected_epoch_offset;
    uint64_t pending_priority_epoch_offset;
    uint64_t fill_expert_offset;
    uint64_t fill_slot_offset;
    uint64_t state_size;
};

struct ggml_backend_cuda_expert_route_layout {
    uint64_t selectors_offset;
    uint64_t route_ids_offset;
    uint64_t route_bounds_offset;
    uint64_t route_plan_offset;
    uint64_t active_experts_offset;
    uint64_t route_tile_bounds_offset;
    uint64_t size;
};

struct ggml_backend_cuda_expert_plan_binding {
    uint64_t                        magic;
    struct ggml_backend_moe_cache * cache;
};

struct ggml_backend_moe_cache {
    ggml_backend_cuda_expert_cache_desc   desc;
    ggml_tensor *                         state;
    ggml_tensor *                         slots[GGML_BACKEND_MOE_CACHE_MAX_WEIGHTS];
    ggml_backend_cuda_expert_source       source_bindings[GGML_BACKEND_MOE_CACHE_MAX_WEIGHTS];
    ggml_backend_cuda_expert_plan_binding plan_binding;
};

static inline size_t ggml_cuda_expert_cache_align(size_t offset, size_t alignment) {
    return (offset + alignment - 1) & ~(alignment - 1);
}

static inline ggml_backend_cuda_expert_layout ggml_cuda_expert_cache_layout(uint32_t n_expert, uint32_t n_cache) {
    ggml_backend_cuda_expert_layout layout = {};
    size_t                          offset = sizeof(ggml_backend_cuda_expert_cache_state);

    layout.expert_to_cache_offset        = ggml_cuda_expert_cache_align(offset, alignof(int32_t));
    offset                               = layout.expert_to_cache_offset + n_expert * sizeof(int32_t);
    layout.cache_to_expert_offset        = ggml_cuda_expert_cache_align(offset, alignof(int32_t));
    offset                               = layout.cache_to_expert_offset + n_cache * sizeof(int32_t);
    layout.last_used_offset              = ggml_cuda_expert_cache_align(offset, alignof(uint64_t));
    offset                               = layout.last_used_offset + n_cache * sizeof(uint64_t);
    layout.protected_epoch_offset        = ggml_cuda_expert_cache_align(offset, alignof(uint64_t));
    offset                               = layout.protected_epoch_offset + n_cache * sizeof(uint64_t);
    layout.pending_priority_epoch_offset = ggml_cuda_expert_cache_align(offset, alignof(uint64_t));
    offset                               = layout.pending_priority_epoch_offset + n_expert * sizeof(uint64_t);
    layout.fill_expert_offset            = ggml_cuda_expert_cache_align(offset, alignof(int32_t));
    offset                               = layout.fill_expert_offset + n_cache * sizeof(int32_t);
    layout.fill_slot_offset              = ggml_cuda_expert_cache_align(offset, alignof(int32_t));
    offset                               = layout.fill_slot_offset + n_cache * sizeof(int32_t);
    layout.state_size                    = ggml_cuda_expert_cache_align(offset, GGML_MEM_ALIGN);

    return layout;
}

static inline ggml_backend_cuda_expert_route_layout ggml_cuda_expert_cache_route_layout(uint64_t n_routes,
                                                                                        uint32_t n_expert) {
    ggml_backend_cuda_expert_route_layout layout = {};
    layout.selectors_offset                      = 0;
    layout.route_ids_offset                      = n_routes * sizeof(int32_t);
    layout.route_bounds_offset                   = layout.route_ids_offset + n_routes * sizeof(int32_t);
    layout.route_plan_offset                     = layout.route_bounds_offset + (n_expert + 1) * sizeof(int32_t);
    layout.active_experts_offset                 = layout.route_plan_offset + 3 * n_expert * sizeof(int32_t);
    layout.route_tile_bounds_offset              = layout.active_experts_offset + n_expert * sizeof(int32_t);
    layout.size                                  = layout.route_tile_bounds_offset + (n_expert + 1) * sizeof(int32_t);
    return layout;
}

extern "C" const ggml_backend_moe_cache_i * ggml_backend_cuda_moe_cache_get_interface();
