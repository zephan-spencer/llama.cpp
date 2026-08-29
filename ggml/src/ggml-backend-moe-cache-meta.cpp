#include "ggml-backend-moe-cache.h"

#include "ggml-backend-impl.h"
#include "ggml-backend-moe-cache-impl.h"

#include <algorithm>
#include <cstring>
#include <vector>

struct ggml_backend_meta_moe_cache_child {
    const ggml_backend_moe_cache_i * api    = nullptr;
    ggml_backend_moe_cache_t         handle = nullptr;
};

struct ggml_backend_meta_moe_cache {
    ggml_tensor *                                  state = nullptr;
    uint32_t                                       n_weights;
    std::vector<ggml_tensor *>                     slots;
    std::vector<ggml_backend_meta_moe_cache_child> children;
    // The physical cache owns this opaque association.  Keep one per child
    // here so selector materialization does not depend on the execution
    // tensor's extra field or on view initialization order.
    std::vector<void *>                             selector_bindings;
    ggml_backend_meta_tensor_adapter               execution_adapter;
    ggml_backend_meta_tensor_adapter               selector_adapter;
};

static bool ggml_backend_meta_moe_cache_supports_weight(
        ggml_backend_dev_t device, const ggml_tensor * tensor) {
    if (device == nullptr || ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_META || tensor == nullptr) {
        return false;
    }
    for (size_t index = 0; index < ggml_backend_meta_n_devices(device); ++index) {
        ggml_backend_dev_t child = ggml_backend_meta_simple_device(device, index);
        const ggml_backend_moe_cache_i * api = ggml_backend_moe_cache_get_interface(child);
        if (api == nullptr || !api->supports_weight(child, tensor)) {
            return false;
        }
    }
    return ggml_backend_meta_n_devices(device) > 0;
}

static ggml_backend_buffer_type_t ggml_backend_meta_moe_cache_source_buffer_type(ggml_backend_dev_t device) {
    if (device == nullptr || ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_META) {
        return nullptr;
    }

    const size_t                     n_devices = ggml_backend_meta_n_devices(device);
    const ggml_backend_moe_cache_i * child_api = nullptr;
    std::vector<ggml_backend_buffer_type_t> child_bufts;
    child_bufts.reserve(n_devices);
    for (size_t index = 0; index < n_devices; ++index) {
        ggml_backend_dev_t               child = ggml_backend_meta_simple_device(device, index);
        const ggml_backend_moe_cache_i * api   = ggml_backend_moe_cache_get_interface(child);
        if (api == nullptr || (child_api != nullptr && api != child_api)) {
            return nullptr;
        }
        ggml_backend_buffer_type_t buft = api->get_source_buffer_type(child);
        if (buft == nullptr) {
            return nullptr;
        }
        child_api = api;
        child_bufts.push_back(buft);
    }

    return child_bufts.empty() ? nullptr :
                                 ggml_backend_meta_buffer_type(device, child_bufts.data(), child_bufts.size());
}

static size_t ggml_backend_meta_moe_cache_state_size(ggml_backend_dev_t device,
                                                     enum ggml_backend_moe_cache_policy policy,
                                                     uint32_t           n_expert,
                                                     uint32_t           n_cache,
                                                     uint32_t           n_weights) {
    if (ggml_backend_meta_moe_cache_source_buffer_type(device) == nullptr) {
        return 0;
    }

    size_t result = 0;
    for (size_t index = 0; index < ggml_backend_meta_n_devices(device); ++index) {
        ggml_backend_dev_t               child = ggml_backend_meta_simple_device(device, index);
        const ggml_backend_moe_cache_i * api   = ggml_backend_moe_cache_get_interface(child);
        result = std::max(result, api->get_state_size(child, policy, n_expert, n_cache, n_weights));
    }
    return result;
}

static ggml_backend_moe_cache_t ggml_backend_meta_moe_cache_create(ggml_backend_t        backend,
                                                                   ggml_tensor *         state,
                                                                   ggml_tensor * const * source,
                                                                   ggml_tensor * const * slots,
                                                                   enum ggml_backend_moe_cache_policy policy,
                                                                   uint32_t              n_expert,
                                                                   uint32_t              n_cache,
                                                                   uint32_t              n_weights) {
    if (!ggml_backend_is_meta(backend) || state == nullptr || source == nullptr || slots == nullptr ||
        state->buffer == nullptr || !ggml_backend_buffer_is_meta(state->buffer) || n_weights == 0 ||
        n_weights > GGML_BACKEND_MOE_CACHE_MAX_WEIGHTS) {
        return nullptr;
    }
    for (uint32_t weight = 0; weight < n_weights; ++weight) {
        if (source[weight] == nullptr || slots[weight] == nullptr || source[weight]->buffer == nullptr ||
            slots[weight]->buffer == nullptr || !ggml_backend_buffer_is_meta(source[weight]->buffer) ||
            !ggml_backend_buffer_is_meta(slots[weight]->buffer) ||
            !ggml_backend_buffer_is_host(source[weight]->buffer)) {
            return nullptr;
        }
    }

    auto * cache     = new ggml_backend_meta_moe_cache{};
    cache->state     = state;
    cache->n_weights = n_weights;
    cache->slots.assign(slots, slots + n_weights);

    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    if (n_backends != ggml_backend_meta_n_devices(ggml_backend_get_device(backend))) {
        delete cache;
        return nullptr;
    }

    const ggml_backend_moe_cache_i * child_api = nullptr;
    cache->children.reserve(n_backends);
    cache->selector_bindings.resize(n_backends);
    for (size_t index = 0; index < n_backends; ++index) {
        ggml_backend_t                   child_backend = ggml_backend_meta_simple_backend(backend, index);
        ggml_backend_dev_t               child_device  = ggml_backend_get_device(child_backend);
        const ggml_backend_moe_cache_i * api           = ggml_backend_moe_cache_get_interface(child_device);
        ggml_tensor *                    child_state   = ggml_backend_meta_buffer_simple_tensor(state, index);
        if (api == nullptr || (child_api != nullptr && api != child_api) || child_state == nullptr ||
            ggml_nbytes(child_state) < api->get_state_size(child_device, policy, n_expert, n_cache, n_weights)) {
            goto fail;
        }
        child_api = api;

        ggml_tensor * child_sources[GGML_BACKEND_MOE_CACHE_MAX_WEIGHTS] = {};
        ggml_tensor * child_slots[GGML_BACKEND_MOE_CACHE_MAX_WEIGHTS]   = {};
        for (uint32_t weight = 0; weight < n_weights; ++weight) {
            child_sources[weight] = ggml_backend_meta_buffer_simple_tensor(source[weight], index);
            child_slots[weight]   = ggml_backend_meta_buffer_simple_tensor(slots[weight], index);
        }
        ggml_backend_moe_cache_t handle =
            api->create(child_backend, child_state, child_sources, child_slots, policy,
                        n_expert, n_cache, n_weights);
        if (handle == nullptr) {
            goto fail;
        }
        cache->children.push_back({ api, handle });
    }

    return reinterpret_cast<ggml_backend_moe_cache_t>(cache);

fail:
    for (const auto & child : cache->children) {
        child.api->destroy(child.handle);
    }
    delete cache;
    return nullptr;
}

static void ggml_backend_meta_moe_cache_destroy(ggml_backend_moe_cache_t opaque) {
    auto * cache = reinterpret_cast<ggml_backend_meta_moe_cache *>(opaque);
    if (cache == nullptr) {
        return;
    }
    for (const auto & child : cache->children) {
        child.api->destroy(child.handle);
    }
    delete cache;
}

static bool ggml_backend_meta_moe_cache_layouts_equal(const ggml_backend_moe_cache_plan_layout & left,
                                                      const ggml_backend_moe_cache_plan_layout & right) {
    if (left.execution_type != right.execution_type || left.selectors_type != right.selectors_type ||
        left.selectors_offset != right.selectors_offset) {
        return false;
    }
    for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
        if (left.execution_ne[dimension] != right.execution_ne[dimension] ||
            left.selectors_ne[dimension] != right.selectors_ne[dimension] ||
            left.selectors_nb[dimension] != right.selectors_nb[dimension]) {
            return false;
        }
    }
    return true;
}

static bool ggml_backend_meta_moe_cache_get_plan_layout(ggml_backend_moe_cache_t             opaque,
                                                        const ggml_tensor *                  logical_ids,
                                                        ggml_backend_moe_cache_plan_layout * result) {
    auto * cache = reinterpret_cast<ggml_backend_meta_moe_cache *>(opaque);
    if (cache == nullptr || logical_ids == nullptr || result == nullptr || cache->children.empty()) {
        return false;
    }

    ggml_backend_moe_cache_plan_layout layout = {};
    if (!cache->children.front().api->get_plan_layout(cache->children.front().handle, logical_ids, &layout)) {
        return false;
    }
    for (size_t index = 1; index < cache->children.size(); ++index) {
        ggml_backend_moe_cache_plan_layout child_layout = {};
        if (!cache->children[index].api->get_plan_layout(cache->children[index].handle, logical_ids, &child_layout) ||
            !ggml_backend_meta_moe_cache_layouts_equal(layout, child_layout)) {
            return false;
        }
    }
    *result = layout;
    return true;
}

static enum ggml_status ggml_backend_meta_moe_cache_materialize_execution(ggml_context * ctx,
                                                                          ggml_tensor *  tensor,
                                                                          size_t         index,
                                                                          void *         userdata) {
    auto * cache = static_cast<ggml_backend_meta_moe_cache *>(userdata);
    if (cache == nullptr || index >= cache->children.size() || tensor->src[0] == nullptr) {
        return GGML_STATUS_FAILED;
    }

    const auto & child = cache->children[index];
    const ggml_backend_moe_cache_plan plan = child.api->build_plan(child.handle, ctx, tensor->src[0]);
    if (plan.execution == nullptr || plan.selectors == nullptr || plan.execution->type != tensor->type ||
        !ggml_are_same_shape(plan.execution, tensor)) {
        return GGML_STATUS_FAILED;
    }
    if (plan.selectors->extra == nullptr) {
        return GGML_STATUS_FAILED;
    }

    tensor->op = plan.execution->op;
    memcpy(tensor->op_params, plan.execution->op_params, sizeof(tensor->op_params));
    for (int source = 0; source < GGML_MAX_SRC; ++source) {
        tensor->src[source] = plan.execution->src[source];
    }
    cache->selector_bindings[index] = plan.selectors->extra;
    tensor->extra = nullptr;
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_meta_moe_cache_materialize_selector(ggml_context * ctx,
                                                                         ggml_tensor *  tensor,
                                                                         size_t         index,
                                                                         void *         userdata) {
    GGML_UNUSED(ctx);
    auto * cache = static_cast<ggml_backend_meta_moe_cache *>(userdata);
    if (cache == nullptr || index >= cache->selector_bindings.size() || tensor->view_src == nullptr ||
        cache->selector_bindings[index] == nullptr) {
        return GGML_STATUS_FAILED;
    }
    tensor->extra = cache->selector_bindings[index];
    return GGML_STATUS_SUCCESS;
}

static ggml_backend_moe_cache_plan ggml_backend_meta_moe_cache_build_plan(ggml_backend_moe_cache_t opaque,
                                                                          ggml_context *           ctx,
                                                                          ggml_tensor *            ids) {
    auto * cache = reinterpret_cast<ggml_backend_meta_moe_cache *>(opaque);
    if (cache == nullptr || ctx == nullptr || ids == nullptr) {
        return { nullptr, nullptr };
    }

    ggml_backend_moe_cache_plan_layout layout = {};
    if (!ggml_backend_meta_moe_cache_get_plan_layout(opaque, ids, &layout)) {
        return { nullptr, nullptr };
    }

    ggml_tensor * args[2 + GGML_BACKEND_MOE_CACHE_MAX_WEIGHTS] = {};
    args[0]                                                    = ids;
    args[1]                                                    = cache->state;
    for (uint32_t weight = 0; weight < cache->n_weights; ++weight) {
        args[2 + weight] = cache->slots[weight];
    }
    ggml_tensor * execution =
        ggml_custom_4d(ctx, layout.execution_type, layout.execution_ne[0], layout.execution_ne[1],
                       layout.execution_ne[2], layout.execution_ne[3], args, 2 + cache->n_weights, nullptr, 1, nullptr);
    ggml_tensor * selectors =
        ggml_view_4d(ctx, execution, layout.selectors_ne[0], layout.selectors_ne[1], layout.selectors_ne[2],
                     layout.selectors_ne[3], layout.selectors_nb[1], layout.selectors_nb[2], layout.selectors_nb[3],
                     layout.selectors_offset);

    const ggml_backend_meta_split_state mirrored = { GGML_BACKEND_SPLIT_AXIS_MIRRORED, { 0 }, { 1 }, 1 };
    ggml_backend_meta_tensor_set_adapter(execution, &cache->execution_adapter, mirrored,
                                         ggml_backend_meta_moe_cache_materialize_execution, cache);
    ggml_backend_meta_tensor_set_adapter(selectors, &cache->selector_adapter, mirrored,
                                         ggml_backend_meta_moe_cache_materialize_selector, cache);
    return { selectors, execution };
}

static const ggml_backend_moe_cache_i ggml_backend_meta_moe_cache_interface = {
    ggml_backend_meta_moe_cache_supports_weight,
    ggml_backend_meta_moe_cache_source_buffer_type,
    ggml_backend_meta_moe_cache_state_size,
    ggml_backend_meta_moe_cache_create,
    ggml_backend_meta_moe_cache_destroy,
    ggml_backend_meta_moe_cache_get_plan_layout,
    ggml_backend_meta_moe_cache_build_plan,
};

const ggml_backend_moe_cache_i * ggml_backend_meta_moe_cache_get_interface() {
    return &ggml_backend_meta_moe_cache_interface;
}
