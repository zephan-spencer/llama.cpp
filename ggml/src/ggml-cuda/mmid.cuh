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

    // Optional route data produced by the cache planner.  Unlike ids_dst and
    // expert_bounds above, these buffers are owned by the cache-plan output
    // and are reused by every projection in the layer.
    int32_t *                     route_ids;
    int32_t *                     route_bounds;
    ggml_cuda_expert_route_plan * route_plan;
    int32_t *                     route_tile_bounds;

    int32_t * expert_order;
    int32_t * fill_expert;
    int32_t * fill_slot;
    int32_t * cache_ids;

    int32_t *  expert_to_cache;
    int32_t *  cache_to_expert;
    uint64_t * last_used;
    uint64_t * protected_epoch;
    uint64_t * pending_priority_epoch;
    uint64_t * use_clock;

    const int32_t * token_priority;
    const int32_t * epoch;

    uint32_t * n_fill;

    uint32_t *                     resolve_active;
    uint32_t *                     policy_flags;
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
                                  int                           n_cache,
                                  cudaStream_t                  stream);

void ggml_cuda_launch_expert_cache_plan(const int32_t *               ids,
                                        const ggml_cuda_expert_plan & plan,
                                        int                           n_experts,
                                        int                           n_tokens,
                                        int                           n_expert_used,
                                        int                           n_cache,
                                        cudaStream_t                  stream);

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
