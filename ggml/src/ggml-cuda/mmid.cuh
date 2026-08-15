#pragma once

#include <cstdint>

struct ggml_cuda_expert_route_plan {
    int32_t first_route;
    int32_t priority;
    int32_t source;
};

struct ggml_cuda_expert_source_view {
    const int32_t * selectors;
    const int32_t * route_ids;
    const int32_t * route_bounds;
    int32_t *       route_tile_bounds;
    int64_t         selector_stride;
    uint32_t        selector_width;
    const char *    host_data;
    int64_t         host_stride;
    uint32_t        n_expert;
};

struct ggml_cuda_expert_plan {
    int32_t * ids_src;
    int32_t * ids_dst;
    int32_t * expert_bounds;
};

struct ggml_cuda_expert_selector_plan {
    int32_t *                     route_ids;
    int32_t *                     route_bounds;
    ggml_cuda_expert_route_plan * route_plan;
    int32_t *                     route_tile_bounds;
    int32_t * expert_order;
    int32_t * selectors;
};

void ggml_cuda_launch_expert_plan(const int32_t *               ids,
                                  const ggml_cuda_expert_plan & plan,
                                  int                           n_experts,
                                  int                           n_tokens,
                                  int                           n_expert_used,
                                  int                           nchannels_y,
                                  int                           si1,
                                  int                           sis1,
                                  bool                          write_inverse,
                                  cudaStream_t                  stream);

void ggml_cuda_launch_expert_cache_selectors(const int32_t *                        ids,
                                             const ggml_cuda_expert_selector_plan & selectors,
                                             const int32_t *                        expert_to_cache,
                                             const int32_t *                        token_priority,
                                             uint32_t *                             resolve_active,
                                             uint32_t *                             policy_flags,
                                             int                                    n_experts,
                                             int                                    n_tokens,
                                             int                                    n_expert_used,
                                             cudaStream_t                           stream);

// Build the projection-specific source index from a route order saved by the
// cache planner.  This is deliberately a single linear pass; route grouping
// itself must not be repeated for each projection.
void ggml_cuda_launch_expert_plan_input_index(const int32_t * route_ids,
                                              int32_t *       ids_src,
                                              int             n_routes,
                                              int             n_expert_used,
                                              int             nchannels_y,
                                              int             sis1,
                                              bool            write_inverse,
                                              cudaStream_t    stream);

void ggml_cuda_launch_expert_route_tiles(const int32_t * route_bounds,
                                         int32_t *       route_tile_bounds,
                                         int             n_experts,
                                         int             routes_per_tile,
                                         cudaStream_t    stream);
