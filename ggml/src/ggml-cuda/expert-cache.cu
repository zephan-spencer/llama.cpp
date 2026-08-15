#include "expert-cache-private.cuh"
#include "expert-cache.cuh"
#include "ggml-cuda.h"
#include "ggml-impl.h"
#include "mmid.cuh"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(GGML_USE_HIP)

static ggml_backend_buffer_type_t ggml_backend_cuda_moe_cache_source_buffer_type(ggml_backend_dev_t device) {
    return device != nullptr && ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU ?
               ggml_backend_dev_host_buffer_type(device) :
               nullptr;
}

static size_t ggml_backend_cuda_moe_cache_state_size(ggml_backend_dev_t device,
                                                     uint32_t           n_expert,
                                                     uint32_t           n_cache,
                                                     uint32_t           n_weights) {
    if (ggml_backend_cuda_moe_cache_source_buffer_type(device) == nullptr || n_expert == 0 || n_cache == 0 ||
        n_cache >= n_expert || n_weights == 0 || n_weights > GGML_BACKEND_MOE_CACHE_MAX_WEIGHTS) {
        return 0;
    }
    return ggml_cuda_expert_cache_layout(n_expert, n_cache).state_size;
}

static ggml_backend_moe_cache_t ggml_backend_cuda_moe_cache_create(ggml_backend_t        backend,
                                                                   ggml_tensor *         state,
                                                                   ggml_tensor * const * source,
                                                                   ggml_tensor * const * slots,
                                                                   uint32_t              n_expert,
                                                                   uint32_t              n_cache,
                                                                   uint32_t              n_weights) {
    const ggml_backend_dev_t device     = backend != nullptr ? ggml_backend_get_device(backend) : nullptr;
    const size_t             state_size = ggml_backend_cuda_moe_cache_state_size(device, n_expert, n_cache, n_weights);
    if (backend == nullptr || state_size == 0 || state == nullptr || source == nullptr || slots == nullptr ||
        state->buffer == nullptr || ggml_nbytes(state) < state_size) {
        return nullptr;
    }

    if (device == nullptr || ggml_backend_buft_get_device(ggml_backend_buffer_get_type(state->buffer)) != device) {
        return nullptr;
    }

    auto * cache                       = new ggml_backend_moe_cache{};
    cache->state                       = state;
    auto & desc                        = cache->desc;
    desc.magic                         = GGML_CUDA_EXPERT_CACHE_MAGIC;
    desc.n_expert                      = n_expert;
    desc.n_cache                       = n_cache;
    desc.n_weights                     = n_weights;
    desc.state_size                    = state_size;
    const auto layout                  = ggml_cuda_expert_cache_layout(n_expert, n_cache);
    desc.expert_to_cache_offset        = layout.expert_to_cache_offset;
    desc.cache_to_expert_offset        = layout.cache_to_expert_offset;
    desc.last_used_offset              = layout.last_used_offset;
    desc.protected_epoch_offset        = layout.protected_epoch_offset;
    desc.pending_priority_epoch_offset = layout.pending_priority_epoch_offset;
    desc.fill_expert_offset            = layout.fill_expert_offset;
    desc.fill_slot_offset              = layout.fill_slot_offset;
    for (uint32_t i = 0; i < n_weights; ++i) {
        if (source[i] == nullptr || slots[i] == nullptr || source[i]->buffer == nullptr || source[i]->data == nullptr ||
            slots[i]->buffer == nullptr || !ggml_backend_buffer_is_host(source[i]->buffer) ||
            ggml_backend_buffer_get_type(source[i]->buffer) != ggml_backend_dev_host_buffer_type(device) ||
            ggml_backend_buft_get_device(ggml_backend_buffer_get_type(slots[i]->buffer)) != device ||
            source[i]->type != slots[i]->type || source[i]->ne[0] != slots[i]->ne[0] ||
            source[i]->ne[1] != slots[i]->ne[1] || source[i]->ne[2] != n_expert || source[i]->ne[3] != 1 ||
            slots[i]->ne[2] != n_cache || slots[i]->ne[3] != 1 || source[i]->nb[2] != slots[i]->nb[2]) {
            delete cache;
            return nullptr;
        }
    }
    for (uint32_t i = 0; i < n_weights; ++i) {
        const uint8_t * base = static_cast<const uint8_t *>(ggml_backend_buffer_get_base(source[i]->buffer));
        desc.weights[i]      = {
            base,
            nullptr,
            static_cast<uint64_t>(static_cast<const uint8_t *>(source[i]->data) - base),
            static_cast<uint64_t>(source[i]->nb[2]),
        };
        cache->slots[i]           = slots[i];
        cache->source_bindings[i] = {
            GGML_CUDA_EXPERT_SOURCE_MAGIC,
            n_expert,
            cache,
            &desc.weights[i],
        };
        slots[i]->extra = &cache->source_bindings[i];
    }
    cache->plan_binding = {
        GGML_CUDA_EXPERT_PLAN_MAGIC,
        cache,
    };
    std::vector<uint8_t> data(state_size, 0);
    auto *               expert_to_cache = reinterpret_cast<int32_t *>(data.data() + desc.expert_to_cache_offset);
    auto *               cache_to_expert = reinterpret_cast<int32_t *>(data.data() + desc.cache_to_expert_offset);
    std::fill(expert_to_cache, expert_to_cache + n_expert, -1);
    std::fill(cache_to_expert, cache_to_expert + n_cache, -1);
    ggml_backend_tensor_set(state, data.data(), 0, data.size());
    return cache;
}

static void ggml_backend_cuda_moe_cache_destroy(ggml_backend_moe_cache_t cache) {
    if (cache == nullptr) {
        return;
    }
    for (uint32_t i = 0; i < cache->desc.n_weights; ++i) {
        if (cache->slots[i] != nullptr && cache->slots[i]->extra == &cache->source_bindings[i]) {
            cache->slots[i]->extra = nullptr;
        }
    }
    delete cache;
}

static bool ggml_backend_cuda_moe_cache_get_plan_layout(
    ggml_backend_moe_cache_t               cache,
    const ggml_tensor *                    ids,
    ggml_backend_moe_cache_plan_layout *   result) {
    if (cache == nullptr || ids == nullptr || result == nullptr || ids->type != GGML_TYPE_I32) {
        return false;
    }

    const auto route_layout = ggml_cuda_expert_cache_route_layout(ggml_nelements(ids), cache->desc.n_expert);

    *result                 = {};
    result->execution_type  = GGML_TYPE_I32;
    result->execution_ne[0] = route_layout.size / sizeof(int32_t);
    result->execution_ne[1] = 1;
    result->execution_ne[2] = 1;
    result->execution_ne[3] = 1;
    result->selectors_type  = GGML_TYPE_I32;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        result->selectors_ne[i] = ids->ne[i];
        result->selectors_nb[i] = ids->nb[i];
    }
    result->selectors_offset = route_layout.selectors_offset;
    return true;
}

static ggml_backend_moe_cache_plan ggml_backend_cuda_moe_cache_build_plan(ggml_backend_moe_cache_t cache,
                                                                          ggml_context *           ctx,
                                                                          ggml_tensor *            ids,
                                                                          ggml_tensor *            token_priority,
                                                                          ggml_tensor *            epoch) {
    if (cache == nullptr || ctx == nullptr || ids == nullptr || token_priority == nullptr || epoch == nullptr ||
        ids->type != GGML_TYPE_I32 || token_priority->type != GGML_TYPE_I32 || epoch->type != GGML_TYPE_I32 ||
        ggml_nelements(token_priority) != ids->ne[1] || ggml_nelements(epoch) != 1) {
        return { nullptr, nullptr };
    }

    ggml_backend_moe_cache_plan_layout layout;
    if (!ggml_backend_cuda_moe_cache_get_plan_layout(cache, ids, &layout)) {
        return { nullptr, nullptr };
    }

    ggml_tensor * args[4 + GGML_BACKEND_MOE_CACHE_MAX_WEIGHTS] = {};
    args[0]                                                    = ids;
    args[1]                                                    = token_priority;
    args[2]                                                    = epoch;
    args[3]                                                    = cache->state;
    for (uint32_t i = 0; i < cache->desc.n_weights; ++i) {
        args[4 + i] = cache->slots[i];
    }
    auto * execution = ggml_custom_4d(ctx, layout.execution_type, layout.execution_ne[0], layout.execution_ne[1],
                                      layout.execution_ne[2], layout.execution_ne[3], args,
                                      4 + cache->desc.n_weights, nullptr, 1, &cache->desc);
    auto * selectors = ggml_view_4d(ctx, execution, layout.selectors_ne[0], layout.selectors_ne[1],
                                    layout.selectors_ne[2], layout.selectors_ne[3], layout.selectors_nb[1],
                                    layout.selectors_nb[2], layout.selectors_nb[3], layout.selectors_offset);
    selectors->extra = &cache->plan_binding;
    return { selectors, execution };
}

static const ggml_backend_moe_cache_i ggml_backend_cuda_moe_cache_interface = {
    ggml_backend_cuda_moe_cache_source_buffer_type,
    ggml_backend_cuda_moe_cache_state_size,
    ggml_backend_cuda_moe_cache_create,
    ggml_backend_cuda_moe_cache_destroy,
    ggml_backend_cuda_moe_cache_get_plan_layout,
    ggml_backend_cuda_moe_cache_build_plan,
};

extern "C" const ggml_backend_moe_cache_i * ggml_backend_cuda_moe_cache_get_interface() {
    return &ggml_backend_cuda_moe_cache_interface;
}

struct expert_cache_weight {
    const uint8_t * host;
    uint8_t *       cache;
    uint64_t        size;
};

struct expert_cache_params {
    expert_cache_weight weights[GGML_BACKEND_MOE_CACHE_MAX_WEIGHTS];
    uint32_t            n_expert;
    uint32_t            n_cache;
    uint32_t            n_weights;
    uint32_t            n_routes;
    uint64_t            expert_to_cache_offset;
    uint64_t            cache_to_expert_offset;
    uint64_t            last_used_offset;
    uint64_t            protected_epoch_offset;
    uint64_t            pending_priority_epoch_offset;
    uint64_t            fill_expert_offset;
    uint64_t            fill_slot_offset;
};

static ggml_backend_cuda_expert_cache_desc * expert_cache_desc(const ggml_tensor * dst) {
    if (dst->op != GGML_OP_CUSTOM) {
        return nullptr;
    }

    ggml_custom_op_params op_params;
    memcpy(&op_params, dst->op_params, sizeof(op_params));
    if (op_params.fun != nullptr || op_params.userdata == nullptr) {
        return nullptr;
    }

    auto * desc = static_cast<ggml_backend_cuda_expert_cache_desc *>(op_params.userdata);
    if (desc->magic != GGML_CUDA_EXPERT_CACHE_MAGIC) {
        return nullptr;
    }
    return desc;
}

static __global__ void expert_cache_copy_kernel(uint8_t * state_data, expert_cache_params params) {
    auto *         header      = reinterpret_cast<ggml_backend_cuda_expert_cache_state *>(state_data);
    const auto *   fill_expert = reinterpret_cast<const int32_t *>(state_data + params.fill_expert_offset);
    const auto *   fill_slot   = reinterpret_cast<const int32_t *>(state_data + params.fill_slot_offset);
    const uint32_t n_fills     = header->n_fills;

    if (n_fills == 0) {
        return;
    }

    uint64_t vectors_per_fill = 0;
    for (uint32_t weight = 0; weight < params.n_weights; ++weight) {
        vectors_per_fill += params.weights[weight].size / sizeof(uint4);
    }

    const uint64_t n_vectors = n_fills * vectors_per_fill;
    for (uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n_vectors;
         i += (uint64_t) gridDim.x * blockDim.x) {
        const uint32_t fill          = i / vectors_per_fill;
        uint64_t       weight_offset = i - fill * vectors_per_fill;
        uint32_t       weight        = 0;
        while (weight_offset >= params.weights[weight].size / sizeof(uint4)) {
            weight_offset -= params.weights[weight].size / sizeof(uint4);
            weight++;
        }

        const expert_cache_weight cached = params.weights[weight];
        const uint64_t            expert = fill_expert[fill];
        const uint64_t            slot   = fill_slot[fill];
        const auto *              src    = reinterpret_cast<const uint4 *>(cached.host + expert * cached.size);
        auto *                    dst    = reinterpret_cast<uint4 *>(cached.cache + slot * cached.size);
        dst[weight_offset]               = src[weight_offset];
    }
}

bool ggml_cuda_expert_cache_supported(const ggml_tensor * dst) {
    ggml_backend_cuda_expert_cache_desc * desc = expert_cache_desc(dst);
    if (desc == nullptr || desc->n_weights == 0 || desc->n_weights > GGML_BACKEND_MOE_CACHE_MAX_WEIGHTS) {
        return false;
    }
    if (dst->type != GGML_TYPE_I32 || dst->src[0] == nullptr || dst->src[0]->type != GGML_TYPE_I32 ||
        !ggml_is_contiguous(dst->src[0])) {
        return false;
    }
    const size_t n_routes     = ggml_nelements(dst->src[0]);
    const auto   route_layout = ggml_cuda_expert_cache_route_layout(n_routes, desc->n_expert);
    if (ggml_nbytes(dst) < route_layout.size) {
        return false;
    }
    if (dst->src[1] == nullptr || dst->src[1]->type != GGML_TYPE_I32 || !ggml_is_contiguous(dst->src[1]) ||
        ggml_nelements(dst->src[1]) != dst->src[0]->ne[1]) {
        return false;
    }
    if (dst->src[2] == nullptr || dst->src[2]->type != GGML_TYPE_I32 || !ggml_is_contiguous(dst->src[2]) ||
        ggml_nelements(dst->src[2]) != 1) {
        return false;
    }
    if (desc->n_expert == 0 || desc->n_cache == 0 || desc->n_cache > desc->n_expert) {
        return false;
    }
    const int state_index = 3;
    if (dst->src[state_index] == nullptr || ggml_nbytes(dst->src[state_index]) < desc->state_size) {
        return false;
    }
    for (uint32_t i = 0; i < desc->n_weights; ++i) {
        if (dst->src[state_index + 1 + i] == nullptr || desc->weights[i].host_data == nullptr ||
            desc->weights[i].expert_size == 0 || desc->weights[i].expert_size % sizeof(uint4) != 0) {
            return false;
        }
    }
    return true;
}

void ggml_cuda_expert_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_backend_cuda_expert_cache_desc * desc = expert_cache_desc(dst);
    GGML_ASSERT(desc != nullptr);

    const int priority_index = 1;
    const int epoch_index    = 2;
    const int state_index    = 3;

    expert_cache_params params           = {};
    params.n_expert                      = desc->n_expert;
    params.n_cache                       = desc->n_cache;
    params.n_weights                     = desc->n_weights;
    params.n_routes                      = ggml_nelements(dst->src[0]);
    params.expert_to_cache_offset        = desc->expert_to_cache_offset;
    params.cache_to_expert_offset        = desc->cache_to_expert_offset;
    params.last_used_offset              = desc->last_used_offset;
    params.protected_epoch_offset        = desc->protected_epoch_offset;
    params.pending_priority_epoch_offset = desc->pending_priority_epoch_offset;
    params.fill_expert_offset            = desc->fill_expert_offset;
    params.fill_slot_offset              = desc->fill_slot_offset;

    for (uint32_t i = 0; i < desc->n_weights; ++i) {
        if (desc->weights[i].device_data == nullptr) {
            void *           device_data = nullptr;
            const hipError_t error =
                hipHostGetDevicePointer(&device_data, const_cast<void *>(desc->weights[i].host_data), 0);
            if (error != hipSuccess) {
                GGML_ABORT("failed to map expert cache host weight: %s", hipGetErrorString(error));
            }
            desc->weights[i].device_data = device_data;
        }

        params.weights[i] = {
            static_cast<const uint8_t *>(desc->weights[i].device_data) + desc->weights[i].host_offset,
            static_cast<uint8_t *>(dst->src[state_index + 1 + i]->data),
            desc->weights[i].expert_size,
        };
    }

    uint8_t * state_data = static_cast<uint8_t *>(dst->src[state_index]->data);
    auto *    header     = reinterpret_cast<ggml_backend_cuda_expert_cache_state *>(state_data);

    ggml_cuda_expert_plan plan = {};
    plan.fill_expert           = reinterpret_cast<int32_t *>(state_data + params.fill_expert_offset);
    plan.fill_slot             = reinterpret_cast<int32_t *>(state_data + params.fill_slot_offset);
    const auto route_layout    = ggml_cuda_expert_cache_route_layout(params.n_routes, desc->n_expert);
    plan.cache_ids = reinterpret_cast<int32_t *>(static_cast<uint8_t *>(dst->data) + route_layout.selectors_offset);
    const size_t n_routes = params.n_routes;
    plan.route_ids = reinterpret_cast<int32_t *>(static_cast<uint8_t *>(dst->data) + route_layout.route_ids_offset);
    plan.route_bounds =
        reinterpret_cast<int32_t *>(static_cast<uint8_t *>(dst->data) + route_layout.route_bounds_offset);
    plan.route_plan = reinterpret_cast<ggml_cuda_expert_route_plan *>(static_cast<uint8_t *>(dst->data) +
                                                                      route_layout.route_plan_offset);
    plan.expert_order =
        reinterpret_cast<int32_t *>(static_cast<uint8_t *>(dst->data) + route_layout.active_experts_offset);
    plan.route_tile_bounds =
        reinterpret_cast<int32_t *>(static_cast<uint8_t *>(dst->data) + route_layout.route_tile_bounds_offset);
    plan.expert_to_cache        = reinterpret_cast<int32_t *>(state_data + params.expert_to_cache_offset);
    plan.cache_to_expert        = reinterpret_cast<int32_t *>(state_data + params.cache_to_expert_offset);
    plan.last_used              = reinterpret_cast<uint64_t *>(state_data + params.last_used_offset);
    plan.protected_epoch        = reinterpret_cast<uint64_t *>(state_data + params.protected_epoch_offset);
    plan.pending_priority_epoch = reinterpret_cast<uint64_t *>(state_data + params.pending_priority_epoch_offset);
    plan.token_priority         = static_cast<const int32_t *>(dst->src[priority_index]->data);
    plan.epoch                  = static_cast<const int32_t *>(dst->src[epoch_index]->data);
    plan.use_clock              = &header->use_clock;
    plan.n_fill                 = &header->n_fills;
    plan.resolve_active         = &header->resolve_active;
    plan.policy_flags           = &header->policy_flags;

    const int n_expert_used = dst->src[0]->ne[0];
    const int n_tokens      = ggml_nelements(dst->src[0]) / n_expert_used;
    ggml_cuda_launch_expert_cache_plan(static_cast<const int32_t *>(dst->src[0]->data), plan, desc->n_expert, n_tokens,
                                       n_expert_used, desc->n_cache, ctx.stream());
    CUDA_CHECK(cudaGetLastError());

    expert_cache_copy_kernel<<<32, 256, 0, ctx.stream()>>>(state_data, params);
    CUDA_CHECK(cudaGetLastError());
}

#else

bool ggml_cuda_expert_cache_supported(const ggml_tensor * dst) {
    GGML_UNUSED(dst);
    return false;
}

void ggml_cuda_expert_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("expert cache is only supported by HIP");
}

#endif
