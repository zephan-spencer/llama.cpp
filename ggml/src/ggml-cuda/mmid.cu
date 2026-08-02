#include "common.cuh"
#include "mmid.cuh"

// To reduce shared memory use, store "it" and "iex_used" with 22/10 bits each.
struct expert_plan_route {
    uint32_t data;

    __device__ expert_plan_route(const uint32_t it, const uint32_t iex_used) {
        data = (it & 0x003FFFFF) | (iex_used << 22);
    }

    __device__ uint32_t it() const {
        return data & 0x003FFFFF;
    }

    __device__ uint32_t iex_used() const {
        return data >> 22;
    }
};
static_assert(sizeof(expert_plan_route) == 4, "unexpected size for expert_plan_route");

// Converts routed expert IDs to a compact expert-major plan.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The upper and lower bounds for the ith expert in the compact src1 tensor are stored in expert_bounds[i:i+1].
template <int n_expert_used_template>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void expert_plan_routes(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;

    extern __shared__ char data_expert_plan[];
    expert_plan_route * store = (expert_plan_route *) data_expert_plan;

    int nex_prev   = 0; // Number of columns for experts with a lower index.
    int it_compact = 0; // Running index for the compact slice of this expert.

    if constexpr (n_expert_used_template == 0) {
        // Generic implementation:
        for (int it = 0; it < n_tokens; ++it) {
            int iex_used = -1; // The index at which the expert is used, if any.
            for (int iex = threadIdx.x; iex < n_expert_used; iex += warp_size) {
                const int expert_used = ids[it*si1 + iex];
                nex_prev += expert_used < expert;
                if (expert_used == expert) {
                    iex_used = iex;
                }
            }

            if (iex_used != -1) {
                store[it_compact] = expert_plan_route(it, iex_used);
            }

            if (warp_reduce_any<warp_size>(iex_used != -1)) {
                it_compact++;
            }
        }
    } else {
        // Implementation optimized for specific numbers of experts used:
        static_assert(n_expert_used == 6 || warp_size % n_expert_used == 0, "bad n_expert_used");
        const int neu_padded = n_expert_used == 6 ? 8 : n_expert_used; // Padded to next higher power of 2.
        for (int it0 = 0; it0 < n_tokens; it0 += warp_size/neu_padded) {
            const int it = it0 + threadIdx.x / neu_padded;

            const int iex = threadIdx.x % neu_padded; // The index at which the expert is used, if any.
            const int expert_used = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
                ids[it*si1 + iex] : INT_MAX;
            const int iex_used = expert_used == expert ? iex : -1;
            nex_prev += expert_used < expert;

            // Whether the threads at this token position have used the expert:
            const int it_compact_add_self = warp_reduce_any<neu_padded>(iex_used != -1);

            // Do a scan over threads at lower token positions in warp to get the correct index for writing data:
            int it_compact_add_lower = 0;
#pragma unroll
            for (int offset = neu_padded; offset < warp_size; offset += neu_padded) {
                const int tmp = __shfl_up_sync(0xFFFFFFFF, it_compact_add_self, offset, warp_size);
                if (threadIdx.x >= static_cast<unsigned int>(offset)) {
                    it_compact_add_lower += tmp;
                }
            }

            if (iex_used != -1) {
                store[it_compact + it_compact_add_lower] = expert_plan_route(it, iex_used);
            }

            // The thread with the highest index in the warp always has the sum over the whole warp, use it to increment all threads:
            it_compact += __shfl_sync(0xFFFFFFFF, it_compact_add_lower + it_compact_add_self, warp_size - 1, warp_size);
        }
    }
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);

    for (int itc = threadIdx.x; itc < it_compact; itc += warp_size) {
        const expert_plan_route store_it = store[itc];
        const int it       = store_it.it();
        const int iex_used = store_it.iex_used();
        ids_dst[nex_prev + itc] = it*n_expert_used + iex_used;
        // ids_src1 holds the forward map, or the inverse map (token slot -> compact row) for quant dedup
        if (ids_src1 == nullptr) {
            continue;
        } else if (write_inverse) {
            ids_src1[it*n_expert_used + iex_used] = nex_prev + itc;
        } else {
            ids_src1[nex_prev + itc] = it*sis1 + iex_used % nchannels_y;
        }
    }

    if (threadIdx.x != 0) {
        return;
    }

    expert_bounds[expert] = nex_prev;

    if (expert < static_cast<int>(gridDim.x) - 1) {
        return;
    }

    expert_bounds[gridDim.x] = nex_prev + it_compact;
}

template <int n_expert_used_template>
static void launch_expert_plan_routes(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    GGML_ASSERT(n_tokens          < (1 << 22) && "too few bits in expert_plan_route");
    GGML_ASSERT(n_expert_used_var < (1 << 10) && "too few bits in expert_plan_route");

    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
    CUDA_SET_SHARED_MEMORY_LIMIT(expert_plan_routes<n_expert_used_template>, smpbo);

    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);
    const size_t nbytes_shared = n_tokens*sizeof(expert_plan_route);
    GGML_ASSERT(nbytes_shared <= smpbo);
    expert_plan_routes<n_expert_used_template><<<num_blocks, block_size, nbytes_shared, stream>>>
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

    const bool compatibility = plan.policy != nullptr && plan.policy[0] == 2;
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
    for (int64_t i = 0; i < n_ids; ++i) {
        const int32_t slot = plan.expert_to_cache[ids[i]];
        plan.cache_ids[i] = slot;
        n_streamed += slot < 0;
    }

    *plan.n_active = n_active;
    *plan.n_resident = n_resident;
    *plan.n_miss = n_miss;
    *plan.n_fill = n_fill;
    *plan.n_evictions = n_evictions;
    *plan.n_read_only = n_read_only;
    *plan.n_streamed = n_streamed;
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

    switch (n_expert_used) {
        case  2:
            launch_expert_plan_routes< 2>(ids, plan.ids_src, plan.ids_dst, plan.expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  4:
            launch_expert_plan_routes< 4>(ids, plan.ids_src, plan.ids_dst, plan.expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  6:
            launch_expert_plan_routes< 6>(ids, plan.ids_src, plan.ids_dst, plan.expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  8:
            launch_expert_plan_routes< 8>(ids, plan.ids_src, plan.ids_dst, plan.expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 16:
            launch_expert_plan_routes<16>(ids, plan.ids_src, plan.ids_dst, plan.expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 32:
            launch_expert_plan_routes<32>(ids, plan.ids_src, plan.ids_dst, plan.expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        default:
            launch_expert_plan_routes< 0>(ids, plan.ids_src, plan.ids_dst, plan.expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
    }

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

    const size_t shared_size = 2 * n_experts * sizeof(int32_t);
    expert_plan_cache<<<1, 256, shared_size, stream>>>(
        ids,
        plan,
        (int64_t) n_tokens * n_expert_used,
        n_experts,
        n_cache,
        n_expert_used);
}
