#include "common.cuh"
#include "mmid.cuh"

static __global__ void expert_plan_routes_all(const int32_t * __restrict__ ids,
                                              int32_t * __restrict__ ids_src1,
                                              int32_t * __restrict__ ids_dst,
                                              int32_t * __restrict__ expert_bounds,
                                              ggml_cuda_expert_route_plan * __restrict__ route_plan,
                                              const int32_t * __restrict__ expert_to_cache,
                                              int32_t * __restrict__ cache_ids,
                                              const int32_t * __restrict__ token_priority,
                                              int32_t * __restrict__ active_experts,
                                              uint32_t * __restrict__ resolve_active,
                                              uint32_t * __restrict__ policy_flags,
                                              const int  n_tokens,
                                              const int  n_expert_used,
                                              const int  nchannels_y,
                                              const int  si1,
                                              const int  sis1,
                                              const bool write_inverse) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int     lane      = threadIdx.x;
    const int     n_routes  = n_tokens * n_expert_used;
    const int     expert    = blockIdx.x;

    int nex_prev = 0;
    for (int route = lane; route < n_routes; route += warp_size) {
        const int it  = route / n_expert_used;
        const int iex = route % n_expert_used;
        nex_prev += ids[it * si1 + iex] < expert;
    }
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);

    int it_compact         = 0;
    int priority_requested = 0;
    for (int route_base = 0; route_base < n_routes; route_base += warp_size) {
        const int  route     = route_base + lane;
        const bool match     = route < n_routes && ids[(route / n_expert_used) * si1 + route % n_expert_used] == expert;
        const int  prefix    = warp_prefix_inclusive_sum<int, warp_size>(match ? 1 : 0);
        const int  n_matches = warp_reduce_sum<warp_size>(match ? 1 : 0);

        if (route_plan != nullptr && match && token_priority[route / n_expert_used] != 0) {
            priority_requested = 1;
        }

        if (match) {
            const int compact = nex_prev + it_compact + prefix - 1;
            ids_dst[compact]  = route;
            if (route_plan != nullptr) {
                const int32_t slot = expert_to_cache[expert];
                cache_ids[route]   = slot >= 0 ? slot : -expert - 1;
            }
            if (ids_src1 != nullptr) {
                if (write_inverse) {
                    ids_src1[route] = compact;
                } else {
                    const int it      = route / n_expert_used;
                    const int iex     = route % n_expert_used;
                    ids_src1[compact] = it * sis1 + iex % nchannels_y;
                }
            }
        }

        it_compact += n_matches;
    }

    if (route_plan != nullptr && it_compact > 0) {
        __syncwarp();
    }
    if (route_plan != nullptr && it_compact > 0) {
        priority_requested = warp_reduce_any<warp_size>(priority_requested);
    }

    if (lane == 0) {
        expert_bounds[expert] = nex_prev;
        if (route_plan != nullptr) {
            const int32_t slot = expert_to_cache[expert];
            route_plan[expert] = {
                it_compact > 0 ? ids_dst[nex_prev] : n_routes,
                priority_requested,
                slot >= 0 ? slot : -expert - 1,
            };
            if (it_compact > 0) {
                active_experts[atomicAdd(resolve_active, 1u)] = expert;
                if (slot < 0) {
                    atomicOr(policy_flags, 1u);
                }
            }
        }
        if (expert == gridDim.x - 1) {
            expert_bounds[gridDim.x] = nex_prev + it_compact;
        }
    }
}

static void launch_expert_plan_routes(const int32_t * __restrict__ ids,
                                      int32_t * __restrict__ ids_src1,
                                      int32_t * __restrict__ ids_dst,
                                      int32_t * __restrict__ expert_bounds,
                                      ggml_cuda_expert_route_plan * __restrict__ route_plan,
                                      const int32_t * __restrict__ expert_to_cache,
                                      int32_t * __restrict__ cache_ids,
                                      const int32_t * __restrict__ token_priority,
                                      int32_t * __restrict__ active_experts,
                                      uint32_t * __restrict__ resolve_active,
                                      uint32_t * __restrict__ policy_flags,
                                      const int    n_experts,
                                      const int    n_tokens,
                                      const int    n_expert_used_var,
                                      const int    nchannels_y,
                                      const int    si1,
                                      const int    sis1,
                                      const bool   write_inverse,
                                      cudaStream_t stream) {
    const int  warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);
    expert_plan_routes_all<<<num_blocks, block_size, 0, stream>>>(
        ids, ids_src1, ids_dst, expert_bounds, route_plan, expert_to_cache, cache_ids, token_priority, active_experts,
        resolve_active, policy_flags, n_tokens, n_expert_used_var, nchannels_y, si1, sis1, write_inverse);
}

static __global__ void expert_plan_input_index(const int32_t * __restrict__ route_ids,
                                               int32_t * __restrict__ ids_src,
                                               const int  n_routes,
                                               const int  n_expert_used,
                                               const int  nchannels_y,
                                               const int  sis1,
                                               const bool write_inverse) {
    for (int route_index = blockIdx.x * blockDim.x + threadIdx.x; route_index < n_routes;
         route_index += blockDim.x * gridDim.x) {
        const int route = route_ids[route_index];
        if (write_inverse) {
            // Quantize-scatter consumes the inverse map (original route ->
            // compact row), whereas the regular quantizer consumes the
            // compact-row -> source-row map.
            ids_src[route] = route_index;
        } else {
            const int token       = route / n_expert_used;
            const int expert_slot = route % n_expert_used;
            ids_src[route_index]  = token * sis1 + expert_slot % nchannels_y;
        }
    }
}

void ggml_cuda_launch_expert_plan_input_index(const int32_t * route_ids,
                                              int32_t *       ids_src,
                                              const int       n_routes,
                                              const int       n_expert_used,
                                              const int       nchannels_y,
                                              const int       sis1,
                                              const bool      write_inverse,
                                              cudaStream_t    stream) {
    GGML_ASSERT(route_ids != nullptr);
    GGML_ASSERT(ids_src != nullptr);
    GGML_ASSERT(n_routes >= 0);
    GGML_ASSERT(n_expert_used > 0);
    GGML_ASSERT(nchannels_y > 0);
    GGML_ASSERT(sis1 > 0);

    if (n_routes == 0) {
        return;
    }

    constexpr int block_size = 256;
    const int     n_blocks   = (n_routes + block_size - 1) / block_size;
    expert_plan_input_index<<<n_blocks, block_size, 0, stream>>>(route_ids, ids_src, n_routes, n_expert_used,
                                                                 nchannels_y, sis1, write_inverse);
}

static __global__ void expert_plan_route_tiles(const int32_t * route_bounds,
                                               int32_t *       route_tile_bounds,
                                               int             n_experts,
                                               int             routes_per_tile) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    int32_t total        = 0;
    route_tile_bounds[0] = 0;
    for (int expert = 0; expert < n_experts; ++expert) {
        const int32_t routes = route_bounds[expert + 1] - route_bounds[expert];
        total += (routes + routes_per_tile - 1) / routes_per_tile;
        route_tile_bounds[expert + 1] = total;
    }
}

void ggml_cuda_launch_expert_route_tiles(const int32_t * route_bounds,
                                         int32_t *       route_tile_bounds,
                                         int             n_experts,
                                         int             routes_per_tile,
                                         cudaStream_t    stream) {
    GGML_ASSERT(route_bounds != nullptr);
    GGML_ASSERT(route_tile_bounds != nullptr);
    GGML_ASSERT(n_experts > 0);
    GGML_ASSERT(routes_per_tile > 0);

    expert_plan_route_tiles<<<1, 1, 0, stream>>>(route_bounds, route_tile_bounds, n_experts, routes_per_tile);
}

void ggml_cuda_launch_expert_plan(const int32_t *               ids,
                                  const ggml_cuda_expert_plan & plan,
                                  const int                     n_experts,
                                  const int                     n_tokens,
                                  const int                     n_expert_used,
                                  const int                     nchannels_y,
                                  const int                     si1,
                                  const int                     sis1,
                                  const bool                    write_inverse,
                                  cudaStream_t                  stream) {
    GGML_ASSERT(ids != nullptr);
    GGML_ASSERT(plan.ids_dst != nullptr);
    GGML_ASSERT(plan.expert_bounds != nullptr);

    launch_expert_plan_routes(ids, plan.ids_src, plan.ids_dst, plan.expert_bounds, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1,
                              write_inverse, stream);
}

void ggml_cuda_launch_expert_cache_selectors(const int32_t *                        ids,
                                             const ggml_cuda_expert_selector_plan & selectors,
                                             const int32_t *                        expert_to_cache,
                                             const int32_t *                        token_priority,
                                             uint32_t *                             resolve_active,
                                             uint32_t *                             policy_flags,
                                             const int                              n_experts,
                                             const int                              n_tokens,
                                             const int                              n_expert_used,
                                             cudaStream_t                           stream) {
    GGML_ASSERT(ids != nullptr);
    GGML_ASSERT(selectors.expert_order != nullptr);
    GGML_ASSERT(selectors.selectors != nullptr);
    GGML_ASSERT(selectors.route_ids != nullptr);
    GGML_ASSERT(selectors.route_bounds != nullptr);
    GGML_ASSERT(selectors.route_plan != nullptr);
    GGML_ASSERT(expert_to_cache != nullptr);
    GGML_ASSERT(resolve_active != nullptr);
    GGML_ASSERT(policy_flags != nullptr);

    launch_expert_plan_routes(ids, nullptr, selectors.route_ids, selectors.route_bounds, selectors.route_plan,
                              expert_to_cache, selectors.selectors, token_priority, selectors.expert_order,
                              resolve_active, policy_flags, n_experts, n_tokens, n_expert_used, /*nchannels_y=*/1,
                              /*si1=*/n_expert_used, /*sis1=*/n_expert_used,
                              /*write_inverse=*/false, stream);
}
