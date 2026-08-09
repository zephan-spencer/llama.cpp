#include "common.cuh"
#include "mmid.cuh"

static __global__ void expert_plan_routes_all(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst,
        int32_t * __restrict__ expert_bounds, const int n_tokens, const int n_expert_used,
        const int nchannels_y, const int si1, const int sis1, const bool write_inverse) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int lane = threadIdx.x;
    const int n_routes = n_tokens*n_expert_used;
    const int expert = blockIdx.x;

    int nex_prev = 0;
    for (int route = lane; route < n_routes; route += warp_size) {
        const int it = route / n_expert_used;
        const int iex = route % n_expert_used;
        nex_prev += ids[it*si1 + iex] < expert;
    }
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);

    int it_compact = 0;
    for (int route_base = 0; route_base < n_routes; route_base += warp_size) {
        const int route = route_base + lane;
        const bool match = route < n_routes && ids[(route / n_expert_used)*si1 + route % n_expert_used] == expert;
        const int prefix = warp_prefix_inclusive_sum<int, warp_size>(match ? 1 : 0);
        const int n_matches = warp_reduce_sum<warp_size>(match ? 1 : 0);

        if (match) {
            const int compact = nex_prev + it_compact + prefix - 1;
            ids_dst[compact] = route;
            if (ids_src1 != nullptr) {
                if (write_inverse) {
                    ids_src1[route] = compact;
                } else {
                    const int it = route / n_expert_used;
                    const int iex = route % n_expert_used;
                    ids_src1[compact] = it*sis1 + iex % nchannels_y;
                }
            }
        }

        it_compact += n_matches;
    }

    if (lane == 0) {
        expert_bounds[expert] = nex_prev;
        if (expert == gridDim.x - 1) {
            expert_bounds[gridDim.x] = nex_prev + it_compact;
        }
    }
}

static void launch_expert_plan_routes(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);
    expert_plan_routes_all<<<num_blocks, block_size, 0, stream>>>
        (ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used_var, nchannels_y, si1, sis1, write_inverse);
}

static __global__ void expert_plan_input_index(
        const int32_t * __restrict__ route_ids,
        int32_t * __restrict__ ids_src,
        const int n_routes,
        const int n_expert_used,
        const int nchannels_y,
        const int sis1,
        const bool write_inverse) {
    for (int route_index = blockIdx.x * blockDim.x + threadIdx.x;
            route_index < n_routes;
            route_index += blockDim.x * gridDim.x) {
        const int route = route_ids[route_index];
        if (write_inverse) {
            // Quantize-scatter consumes the inverse map (original route ->
            // compact row), whereas the regular quantizer consumes the
            // compact-row -> source-row map.
            ids_src[route] = route_index;
        } else {
            const int token = route / n_expert_used;
            const int expert_slot = route % n_expert_used;
            ids_src[route_index] = token * sis1 + expert_slot % nchannels_y;
        }
    }
}

void ggml_cuda_launch_expert_plan_input_index(
        const int32_t * route_ids,
        int32_t * ids_src,
        const int n_routes,
        const int n_expert_used,
        const int nchannels_y,
        const int sis1,
        const bool write_inverse,
        cudaStream_t stream) {
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
    const int n_blocks = (n_routes + block_size - 1) / block_size;
    expert_plan_input_index<<<n_blocks, block_size, 0, stream>>>(
        route_ids, ids_src, n_routes, n_expert_used, nchannels_y, sis1, write_inverse);
}

static __global__ void expert_plan_route_tiles(
        const int32_t * route_bounds,
        int32_t * route_tile_bounds,
        int n_experts,
        int routes_per_tile) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    int32_t total = 0;
    route_tile_bounds[0] = 0;
    for (int expert = 0; expert < n_experts; ++expert) {
        const int32_t routes = route_bounds[expert + 1] - route_bounds[expert];
        total += (routes + routes_per_tile - 1) / routes_per_tile;
        route_tile_bounds[expert + 1] = total;
    }
}

void ggml_cuda_launch_expert_route_tiles(
        const int32_t * route_bounds,
        int32_t * route_tile_bounds,
        int n_experts,
        int routes_per_tile,
        cudaStream_t stream) {
    GGML_ASSERT(route_bounds != nullptr);
    GGML_ASSERT(route_tile_bounds != nullptr);
    GGML_ASSERT(n_experts > 0);
    GGML_ASSERT(routes_per_tile > 0);

    expert_plan_route_tiles<<<1, 1, 0, stream>>>(route_bounds, route_tile_bounds, n_experts, routes_per_tile);
}

static __global__ void expert_plan_cache(
        const int32_t * ids,
        ggml_cuda_expert_plan plan,
        int64_t n_ids,
        int n_experts,
        int n_cache,
        int n_expert_used) {
    enum admission_pass {
        ADMIT_ACTIVE_PRIORITY,
        ADMIT_PENDING_PRIORITY,
        ADMIT_ACTIVE_PROMPT,
        ADMIT_PASS_COUNT,
    };

    extern __shared__ int32_t scratch[];

    int32_t * requested = scratch;
    int32_t * priority_requested = requested + n_experts;
    int32_t * unique = priority_requested + n_experts;

    for (int expert = threadIdx.x; expert < n_experts; expert += blockDim.x) {
        requested[expert] = 0;
        priority_requested[expert] = 0;
    }
    __syncthreads();

    if (threadIdx.x != 0) {
        return;
    }

    const uint64_t epoch = plan.epoch ? static_cast<uint64_t>(*plan.epoch) : 0;
    if (epoch != 0) {
        for (int expert = 0; expert < n_experts; ++expert) {
            if (plan.pending_priority_epoch[expert] != epoch) {
                plan.pending_priority_epoch[expert] = 0;
            }
        }
    }

    int32_t n_active = 0;
    for (int64_t i = 0; i < n_ids; ++i) {
        const int32_t expert = ids[i];
        assert(expert >= 0 && expert < n_experts);
        if (!requested[expert]) {
            requested[expert] = 1;
            unique[n_active++] = expert;
        }
        if (plan.token_priority != nullptr && plan.token_priority[i/n_expert_used] != 0) {
            priority_requested[expert] = 1;
        }
    }
    int32_t n_resident = 0;
    int32_t n_miss = 0;
    int32_t n_fill = 0;
    int32_t n_evictions = 0;
    for (int pass = ADMIT_ACTIVE_PRIORITY; pass < ADMIT_PASS_COUNT; ++pass) {
        const bool pending_pass = pass == ADMIT_PENDING_PRIORITY;
        const bool prompt_pass = pass == ADMIT_ACTIVE_PROMPT;
        const int32_t n_candidates = pending_pass ? n_experts : n_active;
        for (int32_t i = 0; i < n_candidates; ++i) {
            const int32_t expert = pending_pass ? i : unique[i];
            if ((pass == ADMIT_ACTIVE_PRIORITY && priority_requested[expert] == 0) ||
                    (pending_pass && (epoch == 0 || plan.pending_priority_epoch[expert] != epoch || priority_requested[expert] != 0)) ||
                    (prompt_pass && priority_requested[expert] != 0)) {
                continue;
            }

            const int32_t resident_slot = plan.expert_to_cache[expert];
            if (resident_slot >= 0) {
                if (!pending_pass) {
                    if (plan.expert_order != nullptr) {
                        plan.expert_order[n_resident + n_fill] = expert;
                    }
                    n_resident++;
                }
                if (!prompt_pass && epoch != 0) {
                    plan.protected_epoch[resident_slot] = epoch;
                    plan.pending_priority_epoch[expert] = 0;
                }
                continue;
            }
            if (!pending_pass) {
                n_miss++;
            }

            int32_t victim = -1;
            for (int32_t slot = 0; slot < n_cache; ++slot) {
                if (plan.cache_to_expert[slot] < 0) {
                    victim = slot;
                    break;
                }
            }

            if (victim < 0) {
                uint64_t oldest = UINT64_MAX;
                for (int32_t slot = 0; slot < n_cache; ++slot) {
                    const int32_t resident = plan.cache_to_expert[slot];
                    const bool protected_now = epoch != 0 && plan.protected_epoch[slot] == epoch;
                    const bool active_now = requested[resident] != 0;
                    if (!protected_now && !active_now && plan.last_used[slot] < oldest) {
                        oldest = plan.last_used[slot];
                        victim = slot;
                    }
                }
            }
            if (victim < 0) {
                if (!prompt_pass && epoch != 0) {
                    plan.pending_priority_epoch[expert] = epoch;
                }
                continue;
            }

            const int32_t evicted = plan.cache_to_expert[victim];
            if (evicted >= 0) {
                plan.expert_to_cache[evicted] = -1;
                n_evictions++;
            }

            plan.cache_to_expert[victim] = expert;
            plan.protected_epoch[victim] = !prompt_pass && epoch != 0 ? epoch : 0;
            if (!prompt_pass) {
                plan.pending_priority_epoch[expert] = 0;
            }
            if (plan.expert_order != nullptr) {
                plan.expert_order[n_resident + n_fill] = expert;
            }
            plan.fill_expert[n_fill] = expert;
            plan.fill_slot[n_fill] = victim;
            n_fill++;
        }
    }

    for (int32_t i = 0; i < n_fill; ++i) {
        const int32_t expert = plan.fill_expert[i];
        const int32_t slot = plan.fill_slot[i];
        plan.expert_to_cache[expert] = slot;
        plan.cache_to_expert[slot] = expert;
    }

    const uint64_t use_clock = n_active == 0 ? *plan.use_clock : ++*plan.use_clock;
    for (int32_t i = 0; i < n_active; ++i) {
        const int32_t slot = plan.expert_to_cache[unique[i]];
        if (slot >= 0) {
            plan.last_used[slot] = use_clock;
        }
    }
    for (int32_t i = 0; i < n_fill; ++i) {
        plan.last_used[plan.fill_slot[i]] = use_clock;
    }

    int32_t n_streamed = 0;
    for (int32_t expert = 0; expert < n_experts; ++expert) {
        requested[expert] = 0;
    }
    int32_t n_host_experts = 0;
    for (int64_t i = 0; i < n_ids; ++i) {
        const int32_t expert = ids[i];
        const int32_t slot = plan.expert_to_cache[expert];
        plan.cache_ids[i] = slot >= 0 ? slot : -expert - 1;
        n_streamed += slot < 0;
        if (slot < 0 && !requested[expert]) {
            requested[expert] = 1;
            n_host_experts++;
        }
    }

    *plan.n_active = n_active;
    *plan.n_resident = n_resident;
    *plan.n_miss = n_miss;
    *plan.n_fill = n_fill;
    *plan.n_evictions = n_evictions;
    *plan.n_streamed = n_streamed;
    *plan.n_host_experts = n_host_experts;
}

void ggml_cuda_launch_expert_plan(
        const int32_t * ids,
        const ggml_cuda_expert_plan & plan,
        const int n_experts,
        const int n_tokens,
        const int n_expert_used,
        const int nchannels_y,
        const int si1,
        const int sis1,
        const bool write_inverse,
        const int n_cache,
        cudaStream_t stream) {
    GGML_ASSERT(ids != nullptr);
    GGML_ASSERT(plan.ids_dst != nullptr);
    GGML_ASSERT(plan.expert_bounds != nullptr);

    launch_expert_plan_routes(
        ids, plan.ids_src, plan.ids_dst, plan.expert_bounds,
        n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);

    if (plan.expert_order != nullptr) {
        ggml_cuda_launch_expert_cache_plan(
            ids, plan, n_experts, n_tokens, n_expert_used, n_cache, stream);
    }
}

void ggml_cuda_launch_expert_cache_plan(
        const int32_t * ids,
        const ggml_cuda_expert_plan & plan,
        const int n_experts,
        const int n_tokens,
        const int n_expert_used,
        const int n_cache,
        cudaStream_t stream) {
    GGML_ASSERT(ids != nullptr);

    GGML_ASSERT(n_cache > 0 && n_cache <= n_experts);
    GGML_ASSERT(plan.fill_expert != nullptr);
    GGML_ASSERT(plan.fill_slot != nullptr);
    GGML_ASSERT(plan.cache_ids != nullptr);
    GGML_ASSERT(plan.expert_to_cache != nullptr);
    GGML_ASSERT(plan.cache_to_expert != nullptr);
    GGML_ASSERT(plan.last_used != nullptr);
    GGML_ASSERT(plan.protected_epoch != nullptr);
    GGML_ASSERT(plan.pending_priority_epoch != nullptr);
    GGML_ASSERT(plan.use_clock != nullptr);
    GGML_ASSERT(plan.n_active != nullptr);
    GGML_ASSERT(plan.n_resident != nullptr);
    GGML_ASSERT(plan.n_miss != nullptr);
    GGML_ASSERT(plan.n_fill != nullptr);
    GGML_ASSERT(plan.n_evictions != nullptr);
    GGML_ASSERT(plan.n_streamed != nullptr);
    GGML_ASSERT(plan.n_host_experts != nullptr);

    if (plan.route_ids != nullptr || plan.route_bounds != nullptr) {
        GGML_ASSERT(plan.route_ids != nullptr);
        GGML_ASSERT(plan.route_bounds != nullptr);

        // The cache plan owns the route grouping in the cached path.  Keep it
        // on the same stream and before the cache-selection kernel so the
        // grouped routes and boundaries are ready for all projections.
        launch_expert_plan_routes(
            ids, nullptr, plan.route_ids, plan.route_bounds,
            n_experts, n_tokens, n_expert_used,
            /*nchannels_y=*/1, /*si1=*/n_expert_used, /*sis1=*/n_expert_used,
            /*write_inverse=*/false, stream);
    }

    const size_t shared_size = 3 * n_experts * sizeof(int32_t);
    expert_plan_cache<<<1, 256, shared_size, stream>>>(
        ids,
        plan,
        (int64_t) n_tokens * n_expert_used,
        n_experts,
        n_cache,
        n_expert_used);
}
