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

static __global__ void expert_plan_cache(
        const int32_t * ids,
        ggml_cuda_expert_plan plan,
        int64_t n_ids,
        int n_experts,
        int n_cache,
        int n_expert_used) {
    extern __shared__ int32_t scratch[];

    int32_t * requested = scratch;
    int32_t * unique = requested + n_experts;

    for (int expert = threadIdx.x; expert < n_experts; expert += blockDim.x) {
        requested[expert] = 0;
    }
    __syncthreads();

    if (threadIdx.x != 0) {
        return;
    }

    const bool compatibility = plan.policy == nullptr || plan.policy[0] == 2;
    int32_t n_active = 0;
    int32_t n_read_only = 0;
    for (int64_t i = 0; i < n_ids; ++i) {
        const int32_t expert = ids[i];
        assert(expert >= 0 && expert < n_experts);
        const bool update = compatibility || plan.policy == nullptr || plan.policy[i / n_expert_used] == 1;
        n_read_only += !update;
        if (!update) {
            continue;
        }
        if (!requested[expert]) {
            requested[expert] = 1;
            unique[n_active++] = expert;
        }
    }
    if (compatibility && n_active > n_cache) {
        for (int32_t expert = 0; expert < n_experts; ++expert) {
            requested[expert] = 0;
        }
        n_read_only = n_ids;
        n_active = 0;
    }

    int32_t n_resident = 0;
    for (int32_t i = 0; i < n_active; ++i) {
        const int32_t expert = unique[i];
        if (plan.expert_to_cache[expert] >= 0) {
            if (plan.expert_order != nullptr) {
                plan.expert_order[n_resident] = expert;
            }
            n_resident++;
        }
    }

    int32_t n_miss = 0;
    int32_t n_fill = 0;
    int32_t n_evictions = 0;
    for (int32_t i = 0; i < n_active; ++i) {
        const int32_t expert = unique[i];
        if (plan.expert_to_cache[expert] >= 0) {
            continue;
        }
        n_miss++;

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
                if (!requested[resident] && plan.last_used[slot] < oldest) {
                    oldest = plan.last_used[slot];
                    victim = slot;
                }
            }
        }
        if (victim < 0) {
            continue;
        }

        const int32_t evicted = plan.cache_to_expert[victim];
        if (evicted >= 0) {
            plan.expert_to_cache[evicted] = -1;
            n_evictions++;
        }

        plan.cache_to_expert[victim] = expert;
        if (plan.expert_order != nullptr) {
            plan.expert_order[n_resident + n_fill] = expert;
        }
        plan.fill_expert[n_fill] = expert;
        plan.fill_slot[n_fill] = victim;
        n_fill++;
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
    *plan.n_read_only = n_read_only;
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
    GGML_ASSERT(plan.use_clock != nullptr);
    GGML_ASSERT(plan.n_active != nullptr);
    GGML_ASSERT(plan.n_resident != nullptr);
    GGML_ASSERT(plan.n_miss != nullptr);
    GGML_ASSERT(plan.n_fill != nullptr);
    GGML_ASSERT(plan.n_evictions != nullptr);
    GGML_ASSERT(plan.n_read_only != nullptr);
    GGML_ASSERT(plan.n_streamed != nullptr);
    GGML_ASSERT(plan.n_host_experts != nullptr);

    const size_t shared_size = 2 * n_experts * sizeof(int32_t);
    expert_plan_cache<<<1, 256, shared_size, stream>>>(
        ids,
        plan,
        (int64_t) n_tokens * n_expert_used,
        n_experts,
        n_cache,
        n_expert_used);
}
