#include "expert-cache-private.cuh"
#include "expert-cache.cuh"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "ggml-impl.h"
#include "expert-cache-policy.cuh"
#include "expert-cache-route.cuh"

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#if defined(GGML_USE_HIP)

static bool ggml_backend_cuda_moe_cache_supports_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
            return true;
        default:
            return false;
    }
}

static bool ggml_backend_cuda_moe_cache_supports_weight(
        ggml_backend_dev_t device, const ggml_tensor * tensor) {
    return device != nullptr && ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU &&
           tensor != nullptr && ggml_backend_cuda_moe_cache_supports_type(tensor->type) &&
           tensor->ne[2] > 0 && tensor->ne[3] == 1 &&
           tensor->nb[2] % sizeof(uint4) == 0;
}

static const char * ggml_backend_cuda_moe_cache_source_buffer_name(ggml_backend_buffer_type_t) {
    return GGML_CUDA_NAME "_MoE_Cache_Host";
}

static void ggml_backend_cuda_moe_cache_source_buffer_free(ggml_backend_buffer_t buffer) {
    CUDA_CHECK(cudaFreeHost(buffer->context));
}

static ggml_backend_buffer_t ggml_backend_cuda_moe_cache_source_buffer_alloc(
        ggml_backend_buffer_type_t buft, size_t size) {
    if (getenv("GGML_CUDA_NO_PINNED") != nullptr) {
        return nullptr;
    }

    void * ptr = nullptr;
    const hipError_t error = hipHostMalloc(&ptr, size, hipHostMallocMapped);
    if (error != hipSuccess) {
        (void) hipGetLastError();
        return nullptr;
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft                 = buft;
    buffer->iface.free_buffer    = ggml_backend_cuda_moe_cache_source_buffer_free;
    return buffer;
}

static ggml_backend_buffer_type_t ggml_backend_cuda_moe_cache_source_buffer_type(ggml_backend_dev_t device) {
    if (device == nullptr || ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        return nullptr;
    }

    static std::mutex mutex;
    static std::unordered_map<ggml_backend_dev_t, std::unique_ptr<ggml_backend_buffer_type>> types;
    std::lock_guard<std::mutex> lock(mutex);

    auto & result = types[device];
    if (!result) {
        result = std::make_unique<ggml_backend_buffer_type>();
        *result = {
            /* .iface   = */ {
                /* .get_name       = */ ggml_backend_cuda_moe_cache_source_buffer_name,
                /* .alloc_buffer   = */ ggml_backend_cuda_moe_cache_source_buffer_alloc,
                /* .get_alignment  = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
                /* .get_max_size   = */ nullptr,
                /* .get_alloc_size = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
                /* .is_host        = */ ggml_backend_cpu_buffer_type()->iface.is_host,
            },
            /* .device  = */ device,
            /* .context = */ nullptr,
        };
    }
    return result.get();
}

static size_t ggml_backend_cuda_moe_cache_state_size(ggml_backend_dev_t device,
                                                     enum ggml_backend_moe_cache_policy policy,
                                                     uint32_t           n_expert,
                                                     uint32_t           n_cache,
                                                     uint32_t           n_weights) {
    const ggml_cuda_expert_cache_policy_i * policy_api = ggml_cuda_expert_cache_policy(policy);
    if (ggml_backend_cuda_moe_cache_source_buffer_type(device) == nullptr || policy_api == nullptr ||
        n_expert == 0 || n_cache == 0 ||
        n_cache >= n_expert || n_weights == 0 || n_weights > GGML_BACKEND_MOE_CACHE_MAX_WEIGHTS) {
        return 0;
    }
    return ggml_cuda_expert_cache_layout(n_expert, n_cache, policy_api->state_size(n_expert, n_cache)).state_size;
}

static ggml_backend_moe_cache_t ggml_backend_cuda_moe_cache_create(ggml_backend_t        backend,
                                                                   ggml_tensor *         state,
                                                                   ggml_tensor * const * source,
                                                                   ggml_tensor * const * slots,
                                                                   enum ggml_backend_moe_cache_policy policy,
                                                                   uint32_t              n_expert,
                                                                   uint32_t              n_cache,
                                                                   uint32_t              n_weights) {
    const ggml_backend_dev_t device     = backend != nullptr ? ggml_backend_get_device(backend) : nullptr;
    const ggml_cuda_expert_cache_policy_i * policy_api = ggml_cuda_expert_cache_policy(policy);
    const size_t state_size = ggml_backend_cuda_moe_cache_state_size(
        device, policy, n_expert, n_cache, n_weights);
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
    desc.policy                        = policy;
    desc.state_size                    = state_size;
    const auto layout                  = ggml_cuda_expert_cache_layout(
        n_expert, n_cache, policy_api->state_size(n_expert, n_cache));
    desc.expert_to_cache_offset        = layout.expert_to_cache_offset;
    desc.cache_to_expert_offset        = layout.cache_to_expert_offset;
    desc.fill_expert_offset            = layout.fill_expert_offset;
    desc.fill_slot_offset              = layout.fill_slot_offset;
    desc.policy_state_offset           = layout.policy_state_offset;
    for (uint32_t i = 0; i < n_weights; ++i) {
        if (!ggml_backend_cuda_moe_cache_supports_weight(device, source[i]) || slots[i] == nullptr ||
            source[i]->buffer == nullptr || source[i]->data == nullptr ||
            slots[i]->buffer == nullptr || !ggml_backend_buffer_is_host(source[i]->buffer) ||
            ggml_backend_buffer_get_type(source[i]->buffer) !=
                ggml_backend_cuda_moe_cache_source_buffer_type(device) ||
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
    policy_api->initialize(data.data() + desc.policy_state_offset, n_expert, n_cache);
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
    result->type  = GGML_TYPE_I32;
    result->ne[0] = route_layout.size / sizeof(int32_t);
    result->ne[1] = 1;
    result->ne[2] = 1;
    result->ne[3] = 1;
    return true;
}

static ggml_tensor * ggml_backend_cuda_moe_cache_build_plan(ggml_backend_moe_cache_t cache,
                                                            ggml_context *           ctx,
                                                            ggml_tensor *            ids) {
    if (cache == nullptr || ctx == nullptr || ids == nullptr || ids->type != GGML_TYPE_I32) {
        return nullptr;
    }

    ggml_backend_moe_cache_plan_layout layout;
    if (!ggml_backend_cuda_moe_cache_get_plan_layout(cache, ids, &layout)) {
        return nullptr;
    }

    ggml_tensor * args[2 + GGML_BACKEND_MOE_CACHE_MAX_WEIGHTS] = {};
    args[0]                                                    = ids;
    args[1]                                                    = cache->state;
    for (uint32_t i = 0; i < cache->desc.n_weights; ++i) {
        args[2 + i] = cache->slots[i];
    }
    auto * plan = ggml_custom_4d(ctx, layout.type, layout.ne[0], layout.ne[1], layout.ne[2], layout.ne[3], args,
                                 2 + cache->desc.n_weights, nullptr, 1, &cache->desc);
    plan->extra = &cache->plan_binding;
    return plan;
}

static const ggml_backend_moe_cache_i ggml_backend_cuda_moe_cache_interface = {
    ggml_backend_cuda_moe_cache_supports_weight,
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
    uint64_t            fill_expert_offset;
    uint64_t            fill_slot_offset;
    uint64_t            policy_state_offset;
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
    if (desc->n_expert == 0 || desc->n_cache == 0 || desc->n_cache > desc->n_expert) {
        return false;
    }
    const int state_index = 1;
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

    const int state_index = 1;

    expert_cache_params params           = {};
    params.n_expert                      = desc->n_expert;
    params.n_cache                       = desc->n_cache;
    params.n_weights                     = desc->n_weights;
    params.n_routes                      = ggml_nelements(dst->src[0]);
    params.expert_to_cache_offset        = desc->expert_to_cache_offset;
    params.cache_to_expert_offset        = desc->cache_to_expert_offset;
    params.fill_expert_offset            = desc->fill_expert_offset;
    params.fill_slot_offset              = desc->fill_slot_offset;
    params.policy_state_offset           = desc->policy_state_offset;

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

    const auto route_layout = ggml_cuda_expert_cache_route_layout(params.n_routes, desc->n_expert);

    ggml_cuda_expert_selector_plan selectors = {};
    selectors.selectors =
        reinterpret_cast<int32_t *>(static_cast<uint8_t *>(dst->data) + route_layout.selectors_offset);
    selectors.route_ids =
        reinterpret_cast<int32_t *>(static_cast<uint8_t *>(dst->data) + route_layout.route_ids_offset);
    selectors.route_bounds =
        reinterpret_cast<int32_t *>(static_cast<uint8_t *>(dst->data) + route_layout.route_bounds_offset);
    selectors.route_plan = reinterpret_cast<ggml_cuda_expert_route_plan *>(static_cast<uint8_t *>(dst->data) +
                                                                           route_layout.route_plan_offset);
    selectors.expert_order =
        reinterpret_cast<int32_t *>(static_cast<uint8_t *>(dst->data) + route_layout.active_experts_offset);
    selectors.route_tile_bounds =
        reinterpret_cast<int32_t *>(static_cast<uint8_t *>(dst->data) + route_layout.route_tile_bounds_offset);

    ggml_cuda_expert_cache_policy_state policy = {};
    policy.fill_expert = reinterpret_cast<int32_t *>(state_data + params.fill_expert_offset);
    policy.fill_slot   = reinterpret_cast<int32_t *>(state_data + params.fill_slot_offset);
    policy.expert_to_cache        = reinterpret_cast<int32_t *>(state_data + params.expert_to_cache_offset);
    policy.cache_to_expert        = reinterpret_cast<int32_t *>(state_data + params.cache_to_expert_offset);
    policy.n_fill                 = &header->n_fills;
    policy.n_active               = &header->n_active;
    policy.data                   = state_data + params.policy_state_offset;

    const int n_expert_used = dst->src[0]->ne[0];
    const int n_tokens      = ggml_nelements(dst->src[0]) / n_expert_used;
    ggml_cuda_launch_expert_cache_selectors(
        static_cast<const int32_t *>(dst->src[0]->data), selectors, policy.expert_to_cache,
        policy.n_active, desc->n_expert, n_tokens, n_expert_used, ctx.stream());
    const ggml_cuda_expert_cache_policy_i * policy_api = ggml_cuda_expert_cache_policy(desc->policy);
    GGML_ASSERT(policy_api != nullptr);
    policy_api->resolve(selectors, policy, desc->n_expert, desc->n_cache, ctx.stream());
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
