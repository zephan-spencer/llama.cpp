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

static __global__ void expert_plan_cache(ggml_cuda_expert_plan plan, int n_experts, int n_cache) {
    enum admission_pass {
        ADMIT_ACTIVE_PRIORITY,
        ADMIT_PENDING_PRIORITY,
        ADMIT_ACTIVE_PROMPT,
        ADMIT_PASS_COUNT,
    };
    constexpr uint32_t POLICY_MISS    = 1u;
    constexpr uint32_t POLICY_PENDING = 2u;

    extern __shared__ uint64_t active_routes[];

    const int32_t n_routes = plan.route_bounds[n_experts];
    if ((*plan.policy_flags & (POLICY_MISS | POLICY_PENDING)) == 0) {
        if (threadIdx.x == 0) {
            const int32_t  n_active = *plan.resolve_active;
            const uint64_t epoch    = plan.epoch ? static_cast<uint64_t>(*plan.epoch) : 0;
            const uint64_t use_clock = n_active == 0 ? *plan.use_clock : ++*plan.use_clock;
            for (int32_t i = 0; i < n_active; ++i) {
                const int32_t expert = plan.expert_order[i];
                const int32_t slot   = plan.expert_to_cache[expert];
                plan.last_used[slot] = use_clock;
                if (epoch != 0 && plan.route_plan[expert].priority != 0) {
                    plan.protected_epoch[slot] = epoch;
                }
            }

            *plan.n_fill         = 0;
            *plan.resolve_active = 0;
        }
        return;
    }

    const uint32_t policy_flags = *plan.policy_flags;
    const int32_t  n_active     = *plan.resolve_active;

    // Route grouping owns active-expert discovery; the serial policy consumes that saved list.
    for (int32_t i = threadIdx.x; i < n_active; i += blockDim.x) {
        const int32_t expert = plan.expert_order[i];
        active_routes[i] =
            (static_cast<uint64_t>(plan.route_plan[expert].first_route) << 32) | static_cast<uint32_t>(expert);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        const uint64_t epoch       = plan.epoch ? static_cast<uint64_t>(*plan.epoch) : 0;
        bool           has_pending = false;
        if ((policy_flags & POLICY_PENDING) != 0) {
            for (int expert = 0; expert < n_experts; ++expert) {
                if (epoch == 0 || plan.pending_priority_epoch[expert] != epoch) {
                    plan.pending_priority_epoch[expert] = 0;
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
        int32_t n_resident  = 0;
        int32_t n_fill      = 0;
        bool    may_have_pending = has_pending;
        if (!needs_policy) {
            n_resident = n_active;
            if (epoch != 0) {
                for (int32_t i = 0; i < n_active; ++i) {
                    const int32_t expert = static_cast<uint32_t>(active_routes[i]);
                    if (plan.route_plan[expert].priority != 0) {
                        const int32_t slot                  = plan.expert_to_cache[expert];
                        plan.protected_epoch[slot]          = epoch;
                        plan.pending_priority_epoch[expert] = 0;
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
                    if ((pass == ADMIT_ACTIVE_PRIORITY && plan.route_plan[expert].priority == 0) ||
                        (pending_pass && (epoch == 0 || plan.pending_priority_epoch[expert] != epoch ||
                                          plan.route_plan[expert].priority != 0)) ||
                        (prompt_pass && plan.route_plan[expert].priority != 0)) {
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
                    int32_t victim = -1;
                    uint64_t oldest = UINT64_MAX;
                    for (int32_t slot = 0; slot < n_cache; ++slot) {
                        if (plan.cache_to_expert[slot] < 0) {
                            victim = slot;
                            break;
                        }
                        if (!prompt_pass) {
                            const int32_t resident      = plan.cache_to_expert[slot];
                            const bool    protected_now = epoch != 0 && plan.protected_epoch[slot] == epoch;
                            const bool    active_now    = plan.route_plan[resident].first_route != n_routes;
                            if (!protected_now && !active_now && plan.last_used[slot] < oldest) {
                                oldest = plan.last_used[slot];
                                victim = slot;
                            }
                        }
                    }
                    if (victim < 0) {
                        if (!prompt_pass && epoch != 0) {
                            plan.pending_priority_epoch[expert] = epoch;
                            may_have_pending                    = true;
                        }
                        continue;
                    }

                    const int32_t evicted = plan.cache_to_expert[victim];
                    if (evicted >= 0) {
                        plan.expert_to_cache[evicted] = -1;
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
                    plan.fill_slot[n_fill]   = victim;
                    n_fill++;
                }
            }
        }

        for (int32_t i = 0; i < n_fill; ++i) {
            const int32_t expert         = plan.fill_expert[i];
            const int32_t slot           = plan.fill_slot[i];
            plan.expert_to_cache[expert] = slot;
            plan.cache_to_expert[slot]   = expert;
        }

        const uint64_t use_clock = n_active == 0 ? *plan.use_clock : ++*plan.use_clock;
        for (int32_t i = 0; i < n_active; ++i) {
            const int32_t expert = static_cast<uint32_t>(active_routes[i]);
            const int32_t slot   = plan.expert_to_cache[expert];
            if (slot >= 0) {
                plan.last_used[slot] = use_clock;
            }
        }
        for (int32_t i = 0; i < n_fill; ++i) {
            plan.last_used[plan.fill_slot[i]] = use_clock;
        }

        *plan.n_fill         = n_fill;

        bool pending_remains = false;
        if (epoch != 0 && may_have_pending) {
            for (int32_t expert = 0; expert < n_experts; ++expert) {
                pending_remains |= plan.pending_priority_epoch[expert] == epoch;
            }
        }
        *plan.policy_flags  = pending_remains ? POLICY_PENDING : 0;
        *plan.resolve_active = 0;
    }

    __syncthreads();
    for (int32_t i = threadIdx.x; i < n_active; i += blockDim.x) {
        const int32_t expert = static_cast<uint32_t>(active_routes[i]);
        if (plan.route_plan[expert].first_route != n_routes && plan.route_plan[expert].source < 0) {
            const int32_t slot = plan.expert_to_cache[expert];
            if (slot >= 0) {
                plan.route_plan[expert].source = slot;
                for (int32_t compact = plan.route_bounds[expert]; compact < plan.route_bounds[expert + 1]; ++compact) {
                    plan.cache_ids[plan.route_ids[compact]] = slot;
                }
            }
        }
    }
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
                                  const int                     n_cache,
                                  cudaStream_t                  stream) {
    GGML_ASSERT(ids != nullptr);
    GGML_ASSERT(plan.ids_dst != nullptr);
    GGML_ASSERT(plan.expert_bounds != nullptr);

    launch_expert_plan_routes(ids, plan.ids_src, plan.ids_dst, plan.expert_bounds, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1,
                              write_inverse, stream);

    if (plan.expert_order != nullptr) {
        ggml_cuda_launch_expert_cache_plan(ids, plan, n_experts, n_tokens, n_expert_used, n_cache, stream);
    }
}

void ggml_cuda_launch_expert_cache_plan(const int32_t *               ids,
                                        const ggml_cuda_expert_plan & plan,
                                        const int                     n_experts,
                                        const int                     n_tokens,
                                        const int                     n_expert_used,
                                        const int                     n_cache,
                                        cudaStream_t                  stream) {
    GGML_ASSERT(ids != nullptr);

    GGML_ASSERT(n_cache > 0 && n_cache <= n_experts);
    GGML_ASSERT(plan.fill_expert != nullptr);
    GGML_ASSERT(plan.fill_slot != nullptr);
    GGML_ASSERT(plan.expert_order != nullptr);
    GGML_ASSERT(plan.cache_ids != nullptr);
    GGML_ASSERT(plan.expert_to_cache != nullptr);
    GGML_ASSERT(plan.cache_to_expert != nullptr);
    GGML_ASSERT(plan.last_used != nullptr);
    GGML_ASSERT(plan.protected_epoch != nullptr);
    GGML_ASSERT(plan.pending_priority_epoch != nullptr);
    GGML_ASSERT(plan.use_clock != nullptr);
    GGML_ASSERT(plan.n_fill != nullptr);
    GGML_ASSERT(plan.resolve_active != nullptr);
    GGML_ASSERT(plan.policy_flags != nullptr);

    GGML_ASSERT(plan.route_ids != nullptr);
    GGML_ASSERT(plan.route_bounds != nullptr);
    GGML_ASSERT(plan.route_plan != nullptr);

    // Route grouping is the only stage that interprets router output.  The
    // serial cache policy and selector binding consume this saved plan.
    launch_expert_plan_routes(ids, nullptr, plan.route_ids, plan.route_bounds, plan.route_plan, plan.expert_to_cache,
                              plan.cache_ids, plan.token_priority, plan.expert_order, plan.resolve_active,
                              plan.policy_flags, n_experts, n_tokens, n_expert_used, /*nchannels_y=*/1,
                              /*si1=*/n_expert_used, /*sis1=*/n_expert_used,
                              /*write_inverse=*/false, stream);

    const size_t shared_size = n_experts * sizeof(uint64_t);
    expert_plan_cache<<<1, 256, shared_size, stream>>>(plan, n_experts, n_cache);
}
