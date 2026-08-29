#include "common.cuh"
#include "expert-cache-route.cuh"

static __global__ void expert_cache_plan_routes(
        const int32_t * __restrict__ ids,
        int32_t * __restrict__ route_ids,
        int32_t * __restrict__ route_bounds,
        ggml_cuda_expert_route_plan * __restrict__ route_plan,
        const int32_t * __restrict__ expert_to_cache,
        int32_t * __restrict__ selectors,
        int32_t * __restrict__ active_experts,
        uint32_t * __restrict__ n_active,
        int n_tokens,
        int n_expert_used) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int     lane      = threadIdx.x;
    const int     n_routes  = n_tokens * n_expert_used;
    const int     expert    = blockIdx.x;

    int n_previous = 0;
    for (int route = lane; route < n_routes; route += warp_size) {
        n_previous += ids[route] < expert;
    }
    n_previous = warp_reduce_sum<warp_size>(n_previous);

    int n_matches_total = 0;
    for (int route_base = 0; route_base < n_routes; route_base += warp_size) {
        const int  route     = route_base + lane;
        const bool match     = route < n_routes && ids[route] == expert;
        const int  prefix    = warp_prefix_inclusive_sum<int, warp_size>(match ? 1 : 0);
        const int  n_matches = warp_reduce_sum<warp_size>(match ? 1 : 0);

        if (match) {
            const int compact = n_previous + n_matches_total + prefix - 1;
            route_ids[compact] = route;
            const int32_t slot = expert_to_cache[expert];
            selectors[route]   = slot >= 0 ? slot : -expert - 1;
        }
        n_matches_total += n_matches;
    }

    if (lane == 0) {
        route_bounds[expert] = n_previous;
        const int32_t slot   = expert_to_cache[expert];
        route_plan[expert]   = {
            n_matches_total > 0 ? route_ids[n_previous] : n_routes,
            slot >= 0 ? slot : -expert - 1,
        };
        if (n_matches_total > 0) {
            active_experts[atomicAdd(n_active, 1u)] = expert;
        }
        if (expert == gridDim.x - 1) {
            route_bounds[gridDim.x] = n_previous + n_matches_total;
        }
    }
}

void ggml_cuda_launch_expert_cache_selectors(const int32_t *                        ids,
                                             const ggml_cuda_expert_selector_plan & plan,
                                             const int32_t *                        expert_to_cache,
                                             uint32_t *                             n_active,
                                             int                                    n_experts,
                                             int                                    n_tokens,
                                             int                                    n_expert_used,
                                             cudaStream_t                           stream) {
    GGML_ASSERT(ids != nullptr);
    GGML_ASSERT(plan.expert_order != nullptr);
    GGML_ASSERT(plan.selectors != nullptr);
    GGML_ASSERT(plan.route_ids != nullptr);
    GGML_ASSERT(plan.route_bounds != nullptr);
    GGML_ASSERT(plan.route_plan != nullptr);
    GGML_ASSERT(expert_to_cache != nullptr);
    GGML_ASSERT(n_active != nullptr);

    const int  warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);
    expert_cache_plan_routes<<<num_blocks, block_size, 0, stream>>>(
        ids, plan.route_ids, plan.route_bounds, plan.route_plan, expert_to_cache, plan.selectors,
        plan.expert_order, n_active, n_tokens, n_expert_used);
}

static __global__ void expert_cache_input_index(
        const int32_t * __restrict__ route_ids,
        int32_t * __restrict__ ids_src,
        int n_routes,
        int n_expert_used,
        int nchannels_y,
        int sis1,
        bool write_inverse) {
    for (int route_index = blockIdx.x * blockDim.x + threadIdx.x; route_index < n_routes;
         route_index += blockDim.x * gridDim.x) {
        const int route = route_ids[route_index];
        if (write_inverse) {
            ids_src[route] = route_index;
        } else {
            const int token       = route / n_expert_used;
            const int expert_slot = route % n_expert_used;
            ids_src[route_index]  = token * sis1 + expert_slot % nchannels_y;
        }
    }
}

void ggml_cuda_launch_expert_cache_input_index(const int32_t * route_ids,
                                               int32_t *       ids_src,
                                               int             n_routes,
                                               int             n_expert_used,
                                               int             nchannels_y,
                                               int             sis1,
                                               bool            write_inverse,
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
    expert_cache_input_index<<<n_blocks, block_size, 0, stream>>>(
        route_ids, ids_src, n_routes, n_expert_used, nchannels_y, sis1, write_inverse);
}

static __global__ void expert_cache_route_tiles(
        const int32_t * route_bounds,
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

void ggml_cuda_launch_expert_cache_route_tiles(const int32_t * route_bounds,
                                               int32_t *       route_tile_bounds,
                                               int             n_experts,
                                               int             routes_per_tile,
                                               cudaStream_t    stream) {
    GGML_ASSERT(route_bounds != nullptr);
    GGML_ASSERT(route_tile_bounds != nullptr);
    GGML_ASSERT(n_experts > 0);
    GGML_ASSERT(routes_per_tile > 0);
    expert_cache_route_tiles<<<1, 1, 0, stream>>>(route_bounds, route_tile_bounds, n_experts, routes_per_tile);
}
