#include "common.cuh"
#include "expert-cache-policy.cuh"

static __global__ void expert_cache_policy(ggml_cuda_expert_selector_plan      selectors,
                                           ggml_cuda_expert_cache_policy_state state,
                                           int                                 n_experts,
                                           int                                 n_cache) {
    enum admission_pass {
        ADMIT_ACTIVE_PRIORITY,
        ADMIT_PENDING_PRIORITY,
        ADMIT_ACTIVE_PROMPT,
        ADMIT_PASS_COUNT,
    };
    constexpr uint32_t POLICY_MISS    = 1u;
    constexpr uint32_t POLICY_PENDING = 2u;

    extern __shared__ uint64_t active_routes[];

    const int32_t n_routes = selectors.route_bounds[n_experts];
    if ((*state.policy_flags & (POLICY_MISS | POLICY_PENDING)) == 0) {
        if (threadIdx.x == 0) {
            const int32_t  n_active  = *state.resolve_active;
            const uint64_t epoch     = state.epoch ? static_cast<uint64_t>(*state.epoch) : 0;
            const uint64_t use_clock = n_active == 0 ? *state.use_clock : ++*state.use_clock;
            for (int32_t i = 0; i < n_active; ++i) {
                const int32_t expert = selectors.expert_order[i];
                const int32_t slot   = state.expert_to_cache[expert];
                state.last_used[slot] = use_clock;
                if (epoch != 0 && selectors.route_plan[expert].priority != 0) {
                    state.protected_epoch[slot] = epoch;
                }
            }

            *state.n_fill         = 0;
            *state.resolve_active = 0;
        }
        return;
    }

    const uint32_t policy_flags = *state.policy_flags;
    const int32_t  n_active     = *state.resolve_active;

    for (int32_t i = threadIdx.x; i < n_active; i += blockDim.x) {
        const int32_t expert = selectors.expert_order[i];
        active_routes[i] =
            (static_cast<uint64_t>(selectors.route_plan[expert].first_route) << 32) | static_cast<uint32_t>(expert);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        const uint64_t epoch       = state.epoch ? static_cast<uint64_t>(*state.epoch) : 0;
        bool           has_pending = false;
        if ((policy_flags & POLICY_PENDING) != 0) {
            for (int expert = 0; expert < n_experts; ++expert) {
                if (epoch == 0 || state.pending_priority_epoch[expert] != epoch) {
                    state.pending_priority_epoch[expert] = 0;
                } else {
                    has_pending = true;
                }
            }
        }

        const bool needs_policy = (policy_flags & POLICY_MISS) != 0 || has_pending;
        if (needs_policy) {
            for (int32_t i = 1; i < n_active; ++i) {
                const uint64_t active_route = active_routes[i];
                int32_t        position     = i;
                while (position > 0 && active_routes[position - 1] > active_route) {
                    active_routes[position] = active_routes[position - 1];
                    position--;
                }
                active_routes[position] = active_route;
            }
        }
        int32_t n_resident      = 0;
        int32_t n_fill          = 0;
        bool    may_have_pending = has_pending;
        if (!needs_policy) {
            n_resident = n_active;
            if (epoch != 0) {
                for (int32_t i = 0; i < n_active; ++i) {
                    const int32_t expert = static_cast<uint32_t>(active_routes[i]);
                    if (selectors.route_plan[expert].priority != 0) {
                        const int32_t slot                         = state.expert_to_cache[expert];
                        state.protected_epoch[slot]                = epoch;
                        state.pending_priority_epoch[expert]       = 0;
                    }
                }
            }
        }
        if (needs_policy) {
            for (int pass = ADMIT_ACTIVE_PRIORITY; pass < ADMIT_PASS_COUNT; ++pass) {
                const bool    pending_pass = pass == ADMIT_PENDING_PRIORITY;
                const bool    prompt_pass  = pass == ADMIT_ACTIVE_PROMPT;
                const int32_t n_candidates = pending_pass ? n_experts : n_active;
                for (int32_t i = 0; i < n_candidates; ++i) {
                    const int32_t expert = pending_pass ? i : static_cast<uint32_t>(active_routes[i]);
                    if ((pass == ADMIT_ACTIVE_PRIORITY && selectors.route_plan[expert].priority == 0) ||
                        (pending_pass && (epoch == 0 || state.pending_priority_epoch[expert] != epoch ||
                                          selectors.route_plan[expert].priority != 0)) ||
                        (prompt_pass && selectors.route_plan[expert].priority != 0)) {
                        continue;
                    }

                    const int32_t resident_slot = state.expert_to_cache[expert];
                    if (resident_slot >= 0) {
                        if (!pending_pass) {
                            if (selectors.expert_order != nullptr) {
                                selectors.expert_order[n_resident + n_fill] = expert;
                            }
                            n_resident++;
                        }
                        if (!prompt_pass && epoch != 0) {
                            state.protected_epoch[resident_slot]          = epoch;
                            state.pending_priority_epoch[expert]          = 0;
                        }
                        continue;
                    }
                    int32_t  victim = -1;
                    uint64_t oldest = UINT64_MAX;
                    for (int32_t slot = 0; slot < n_cache; ++slot) {
                        if (state.cache_to_expert[slot] < 0) {
                            victim = slot;
                            break;
                        }
                        if (!prompt_pass) {
                            const int32_t resident      = state.cache_to_expert[slot];
                            const bool    protected_now = epoch != 0 && state.protected_epoch[slot] == epoch;
                            const bool active_now = selectors.route_plan[resident].first_route != n_routes;
                            if (!protected_now && !active_now && state.last_used[slot] < oldest) {
                                oldest = state.last_used[slot];
                                victim = slot;
                            }
                        }
                    }
                    if (victim < 0) {
                        if (!prompt_pass && epoch != 0) {
                            state.pending_priority_epoch[expert] = epoch;
                            may_have_pending                     = true;
                        }
                        continue;
                    }

                    const int32_t evicted = state.cache_to_expert[victim];
                    if (evicted >= 0) {
                        state.expert_to_cache[evicted] = -1;
                    }

                    state.cache_to_expert[victim] = expert;
                    state.protected_epoch[victim] = !prompt_pass && epoch != 0 ? epoch : 0;
                    if (!prompt_pass) {
                        state.pending_priority_epoch[expert] = 0;
                    }
                    if (selectors.expert_order != nullptr) {
                        selectors.expert_order[n_resident + n_fill] = expert;
                    }
                    state.fill_expert[n_fill] = expert;
                    state.fill_slot[n_fill]   = victim;
                    n_fill++;
                }
            }
        }

        for (int32_t i = 0; i < n_fill; ++i) {
            const int32_t expert               = state.fill_expert[i];
            const int32_t slot                 = state.fill_slot[i];
            state.expert_to_cache[expert]       = slot;
            state.cache_to_expert[slot]         = expert;
        }

        const uint64_t use_clock = n_active == 0 ? *state.use_clock : ++*state.use_clock;
        for (int32_t i = 0; i < n_active; ++i) {
            const int32_t expert = static_cast<uint32_t>(active_routes[i]);
            const int32_t slot   = state.expert_to_cache[expert];
            if (slot >= 0) {
                state.last_used[slot] = use_clock;
            }
        }
        for (int32_t i = 0; i < n_fill; ++i) {
            state.last_used[state.fill_slot[i]] = use_clock;
        }

        *state.n_fill = n_fill;

        bool pending_remains = false;
        if (epoch != 0 && may_have_pending) {
            for (int32_t expert = 0; expert < n_experts; ++expert) {
                pending_remains |= state.pending_priority_epoch[expert] == epoch;
            }
        }
        *state.policy_flags  = pending_remains ? POLICY_PENDING : 0;
        *state.resolve_active = 0;
    }

    __syncthreads();
    for (int32_t i = threadIdx.x; i < n_active; i += blockDim.x) {
        const int32_t expert = static_cast<uint32_t>(active_routes[i]);
        if (selectors.route_plan[expert].first_route != n_routes && selectors.route_plan[expert].source < 0) {
            const int32_t slot = state.expert_to_cache[expert];
            if (slot >= 0) {
                selectors.route_plan[expert].source = slot;
                for (int32_t compact = selectors.route_bounds[expert]; compact < selectors.route_bounds[expert + 1];
                     ++compact) {
                    selectors.selectors[selectors.route_ids[compact]] = slot;
                }
            }
        }
    }
}

void ggml_cuda_apply_expert_cache_policy(const ggml_cuda_expert_selector_plan &      selectors,
                                         const ggml_cuda_expert_cache_policy_state & state,
                                         const int                                   n_experts,
                                         const int                                   n_cache,
                                         cudaStream_t                                stream) {
    GGML_ASSERT(n_cache > 0 && n_cache <= n_experts);
    GGML_ASSERT(selectors.expert_order != nullptr);
    GGML_ASSERT(selectors.selectors != nullptr);
    GGML_ASSERT(selectors.route_ids != nullptr);
    GGML_ASSERT(selectors.route_bounds != nullptr);
    GGML_ASSERT(selectors.route_plan != nullptr);
    GGML_ASSERT(state.fill_expert != nullptr);
    GGML_ASSERT(state.fill_slot != nullptr);
    GGML_ASSERT(state.expert_to_cache != nullptr);
    GGML_ASSERT(state.cache_to_expert != nullptr);
    GGML_ASSERT(state.last_used != nullptr);
    GGML_ASSERT(state.protected_epoch != nullptr);
    GGML_ASSERT(state.pending_priority_epoch != nullptr);
    GGML_ASSERT(state.use_clock != nullptr);
    GGML_ASSERT(state.n_fill != nullptr);
    GGML_ASSERT(state.resolve_active != nullptr);
    GGML_ASSERT(state.policy_flags != nullptr);

    const size_t shared_size = n_experts * sizeof(uint64_t);
    expert_cache_policy<<<1, 256, shared_size, stream>>>(selectors, state, n_experts, n_cache);
}
