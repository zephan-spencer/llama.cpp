#include "common.cuh"
#include "expert-cache-policy.cuh"

#include <algorithm>
#include <cstring>

struct ggml_cuda_expert_cache_lru_state {
    uint64_t use_clock;
};

static size_t expert_cache_lru_state_size(uint32_t, uint32_t n_cache) {
    return sizeof(ggml_cuda_expert_cache_lru_state) + n_cache * sizeof(uint64_t);
}

static void expert_cache_lru_initialize(void * state, uint32_t, uint32_t n_cache) {
    memset(state, 0, expert_cache_lru_state_size(0, n_cache));
}

static __global__ void expert_cache_lru(
        ggml_cuda_expert_selector_plan selectors,
        ggml_cuda_expert_cache_policy_state state,
        int n_experts,
        int n_cache) {
    extern __shared__ uint64_t scratch[];

    auto * lru = static_cast<ggml_cuda_expert_cache_lru_state *>(state.data);
    auto * last_used = reinterpret_cast<uint64_t *>(lru + 1);

    const int32_t n_active = *state.n_active;
    const int32_t n_routes = selectors.route_bounds[n_experts];

    if (lru->use_clock == UINT64_MAX) {
        for (int32_t slot = threadIdx.x; slot < n_cache; slot += blockDim.x) {
            scratch[slot] = last_used[slot];
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            uint64_t occupied = 0;
            for (int32_t slot = 0; slot < n_cache; ++slot) {
                if (state.cache_to_expert[slot] < 0) {
                    last_used[slot] = 0;
                    continue;
                }
                uint64_t rank = 1;
                for (int32_t other = 0; other < n_cache; ++other) {
                    if (state.cache_to_expert[other] >= 0 &&
                        (scratch[other] < scratch[slot] ||
                         (scratch[other] == scratch[slot] && other < slot))) {
                        ++rank;
                    }
                }
                last_used[slot] = rank;
                occupied = max(occupied, rank);
            }
            lru->use_clock = occupied;
        }
        __syncthreads();
    }

    for (int32_t i = threadIdx.x; i < n_active; i += blockDim.x) {
        const int32_t expert = selectors.expert_order[i];
        scratch[i] = (static_cast<uint64_t>(selectors.route_plan[expert].first_route) << 32) |
                     static_cast<uint32_t>(expert);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int32_t i = 1; i < n_active; ++i) {
            const uint64_t value = scratch[i];
            int32_t position = i;
            while (position > 0 && scratch[position - 1] > value) {
                scratch[position] = scratch[position - 1];
                --position;
            }
            scratch[position] = value;
        }

        int32_t n_fill = 0;
        for (int32_t i = 0; i < n_active; ++i) {
            const int32_t expert = static_cast<uint32_t>(scratch[i]);
            int32_t slot = state.expert_to_cache[expert];

            if (slot < 0) {
                uint64_t oldest = UINT64_MAX;
                for (int32_t candidate = 0; candidate < n_cache; ++candidate) {
                    const int32_t resident = state.cache_to_expert[candidate];
                    if (resident < 0) {
                        slot = candidate;
                        break;
                    }
                    const bool pinned = selectors.route_plan[resident].first_route != n_routes;
                    if (!pinned && last_used[candidate] < oldest) {
                        oldest = last_used[candidate];
                        slot = candidate;
                    }
                }

                if (slot < 0) {
                    continue;
                }

                const int32_t evicted = state.cache_to_expert[slot];
                if (evicted >= 0) {
                    state.expert_to_cache[evicted] = -1;
                }
                state.expert_to_cache[expert] = slot;
                state.cache_to_expert[slot] = expert;
                state.fill_expert[n_fill] = expert;
                state.fill_slot[n_fill] = slot;
                ++n_fill;

                selectors.route_plan[expert].source = slot;
                for (int32_t route = selectors.route_bounds[expert]; route < selectors.route_bounds[expert + 1]; ++route) {
                    selectors.selectors[selectors.route_ids[route]] = slot;
                }
            }

            last_used[slot] = ++lru->use_clock;
        }

        *state.n_fill = n_fill;
        *state.n_active = 0;
    }
}

static void expert_cache_lru_resolve(
        const ggml_cuda_expert_selector_plan & selectors,
        const ggml_cuda_expert_cache_policy_state & state,
        int n_experts,
        int n_cache,
        cudaStream_t stream) {
    GGML_ASSERT(n_cache > 0 && n_cache < n_experts);
    GGML_ASSERT(selectors.expert_order != nullptr);
    GGML_ASSERT(selectors.selectors != nullptr);
    GGML_ASSERT(selectors.route_ids != nullptr);
    GGML_ASSERT(selectors.route_bounds != nullptr);
    GGML_ASSERT(selectors.route_plan != nullptr);
    GGML_ASSERT(state.fill_expert != nullptr);
    GGML_ASSERT(state.fill_slot != nullptr);
    GGML_ASSERT(state.expert_to_cache != nullptr);
    GGML_ASSERT(state.cache_to_expert != nullptr);
    GGML_ASSERT(state.n_fill != nullptr);
    GGML_ASSERT(state.n_active != nullptr);
    GGML_ASSERT(state.data != nullptr);

    const size_t shared_size = std::max(n_experts, n_cache) * sizeof(uint64_t);
    expert_cache_lru<<<1, 256, shared_size, stream>>>(selectors, state, n_experts, n_cache);
}

static const ggml_cuda_expert_cache_policy_i lru_policy = {
    expert_cache_lru_state_size,
    expert_cache_lru_initialize,
    expert_cache_lru_resolve,
};

const ggml_cuda_expert_cache_policy_i * ggml_cuda_expert_cache_policy(
        enum ggml_backend_moe_cache_policy policy) {
    switch (policy) {
        case GGML_BACKEND_MOE_CACHE_POLICY_LRU:
            return &lru_policy;
    }
    return nullptr;
}
