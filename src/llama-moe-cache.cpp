#include "llama-moe-cache.h"

#include "ggml-alloc.h"
#include "ggml-cpp.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

bool llama_moe_cache_is_routed_weight(llm_tensor tensor, const char * suffix) {
    return suffix != nullptr && strcmp(suffix, "weight") == 0 &&
           (tensor == LLM_TENSOR_FFN_DOWN_EXPS || tensor == LLM_TENSOR_FFN_GATE_EXPS ||
            tensor == LLM_TENSOR_FFN_UP_EXPS || tensor == LLM_TENSOR_FFN_GATE_UP_EXPS);
}

bool llama_moe_cache_supports_weight(const ggml_tensor * tensor) {
    return tensor != nullptr && ggml_is_quantized(tensor->type);
}

void llama_moe_cache_validate_model(const llama_model & model, uint32_t n_cache) {
    if (model.hparams.n_expert == 0) {
        throw std::runtime_error("MoE expert cache: model has no routed experts");
    }

    if (n_cache < model.hparams.n_expert_used) {
        throw std::runtime_error("MoE expert cache: capacity must cover the experts used per token");
    }

    if (n_cache >= model.hparams.n_expert) {
        throw std::runtime_error(
            "MoE expert cache: capacity must remain below the model's total expert count; "
            "omit the option to use ordinary model placement");
    }

    if (model.n_devices() != 1) {
        throw std::runtime_error("MoE expert cache: exactly one GPU device is required");
    }

    if (model.n_gpu_layers() != model.hparams.n_layer_all + 1) {
        throw std::runtime_error("MoE expert cache: every model layer requires GPU placement");
    }
}

struct llama_moe_expert_cache::impl {
    struct weight {
        const ggml_tensor * source;
        ggml_tensor *       slots;
    };

    struct group {
        int                              il;
        int32_t                          n_expert;
        ggml_backend_t                   backend;
        ggml_backend_buffer_type_t       buft;
        const ggml_backend_moe_cache_i * api;
        ggml_backend_moe_cache_t         handle = nullptr;
        ggml_context_ptr                 context;
        ggml_backend_buffer_ptr          buffer;
        size_t                           buffer_size = 0;
        ggml_tensor *                    state       = nullptr;
        std::vector<weight>              weights;

        ~group() {
            if (handle != nullptr) {
                api->destroy(handle);
            }
        }
    };

    const llama_model &                              model;
    const std::vector<ggml_backend_t> &              backends;
    uint32_t                                         n_cache;
    std::vector<std::unique_ptr<group>>              groups;
    std::unordered_map<const ggml_tensor *, group *> by_anchor;
    std::unordered_set<ggml_backend_t>               pending;
    ggml_backend_dev_t                               device       = nullptr;
    bool                                             info_printed = false;
    uint32_t                                         batch_epoch  = 0;

    impl(const llama_model & model, const std::vector<ggml_backend_t> & backends, uint32_t n_cache) :
        model(model),
        backends(backends),
        n_cache(n_cache) {}

    ggml_backend_t backend_for_layer(int il) const {
        const ggml_backend_dev_t target = model.dev_layer(il);
        for (ggml_backend_t backend : backends) {
            if (ggml_backend_get_device(backend) == target) {
                return backend;
            }
        }
        return nullptr;
    }

    static std::vector<ggml_tensor *> sources(ggml_tensor * up,
                                              ggml_tensor * gate,
                                              ggml_tensor * down,
                                              ggml_tensor * gate_up) {
        std::vector<ggml_tensor *> result;
        for (ggml_tensor * tensor : { gate_up, up, gate, down }) {
            if (llama_moe_cache_supports_weight(tensor) &&
                std::find(result.begin(), result.end(), tensor) == result.end()) {
                result.push_back(tensor);
            }
        }
        return result;
    }

    static const ggml_backend_moe_cache_i * api_for(ggml_backend_dev_t device) {
        return ggml_backend_moe_cache_get_interface(device);
    }

    group * create(int il, const std::vector<ggml_tensor *> & source_tensors) {
        if (source_tensors.empty()) {
            throw std::runtime_error("MoE expert cache: no routed expert weights");
        }

        const int64_t n_expert = source_tensors.front()->ne[2];
        if (n_expert <= 0 || n_cache == 0 || n_cache >= static_cast<uint64_t>(n_expert)) {
            throw std::runtime_error("MoE expert cache: invalid capacity");
        }

        ggml_backend_t     backend      = backend_for_layer(il);
        ggml_backend_dev_t layer_device = backend ? ggml_backend_get_device(backend) : nullptr;
        const enum ggml_backend_dev_type device_type =
            layer_device != nullptr ? ggml_backend_dev_type(layer_device) : GGML_BACKEND_DEVICE_TYPE_CPU;
        if (backend == nullptr || layer_device == nullptr ||
            (device_type != GGML_BACKEND_DEVICE_TYPE_GPU && device_type != GGML_BACKEND_DEVICE_TYPE_META)) {
            throw std::runtime_error("MoE expert cache: every routed layer requires GPU placement");
        }

        const ggml_backend_moe_cache_i * api = api_for(layer_device);
        if (api == nullptr) {
            throw std::runtime_error("MoE expert cache: the selected backend lacks expert-cache support");
        }

        if (device != nullptr && device != layer_device) {
            throw std::runtime_error("MoE expert cache: every cached layer requires one backend device");
        }
        device = layer_device;

        if (source_tensors.size() > GGML_BACKEND_MOE_CACHE_MAX_WEIGHTS) {
            throw std::runtime_error("MoE expert cache: too many routed projections");
        }

        for (ggml_tensor * tensor : source_tensors) {
            if (tensor->ne[2] != n_expert || tensor->ne[3] != 1 || tensor->buffer == nullptr ||
                !ggml_backend_buffer_is_host(tensor->buffer)) {
                throw std::runtime_error(
                    "MoE expert cache: routed expert weights require complete host-resident tensors");
            }
        }

        auto result      = std::make_unique<group>();
        result->il       = il;
        result->n_expert = n_expert;
        result->backend  = backend;
        result->buft     = ggml_backend_get_default_buffer_type(backend);
        result->api      = api;

        const size_t state_size = api->get_state_size(layer_device, n_expert, n_cache, source_tensors.size());
        if (state_size == 0) {
            throw std::runtime_error("MoE expert cache: backend rejected cache configuration");
        }

        result->context.reset(ggml_init({
            (source_tensors.size() + 1) * ggml_tensor_overhead(),
            nullptr,
            true,
        }));

        for (ggml_tensor * tensor : source_tensors) {
            const int64_t ne[4] = { tensor->ne[0], tensor->ne[1], static_cast<int64_t>(n_cache), 1 };
            ggml_tensor * slots = ggml_new_tensor(result->context.get(), tensor->type, 4, ne);
            for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
                slots->nb[dimension] = tensor->nb[dimension];
            }
            if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
                ggml_set_name(slots, tensor->name);
            } else {
                ggml_format_name(slots, "%s#moe_cache", tensor->name);
            }
            result->weights.push_back({ tensor, slots });
        }

        result->state = ggml_new_tensor_1d(result->context.get(), GGML_TYPE_I8, state_size);
        ggml_format_name(result->state, "blk.%d.moe_cache_state", il);
        result->buffer_size = ggml_backend_alloc_ctx_tensors_from_buft_size(result->context.get(), result->buft);

        if (!model.hparams.no_alloc) {
            result->buffer.reset(ggml_backend_alloc_ctx_tensors(result->context.get(), backend));
            if (!result->buffer) {
                throw std::runtime_error("MoE expert cache: failed to allocate device cache");
            }

            ggml_backend_buffer_set_usage(result->buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            ggml_backend_buffer_clear(result->buffer.get(), 0);

            std::vector<ggml_tensor *> sources;
            std::vector<ggml_tensor *> slots;
            for (const weight & entry : result->weights) {
                sources.push_back(const_cast<ggml_tensor *>(entry.source));
                slots.push_back(entry.slots);
            }

            result->handle =
                api->create(backend, result->state, sources.data(), slots.data(), n_expert, n_cache, sources.size());
            if (result->handle == nullptr) {
                throw std::runtime_error("MoE expert cache: backend failed to create cache handle");
            }

            pending.insert(backend);
            result->buffer_size = ggml_backend_buffer_get_size(result->buffer.get());
        }

        group * value = result.get();
        by_anchor.emplace(source_tensors.front(), value);
        groups.push_back(std::move(result));
        return value;
    }

    group * find(int il, const std::vector<ggml_tensor *> & source_tensors) {
        const auto it = by_anchor.find(source_tensors.front());
        return it == by_anchor.end() ? create(il, source_tensors) : it->second;
    }

    static ggml_tensor * cached(group * group, ggml_tensor * source) {
        if (source == nullptr) {
            return nullptr;
        }

        for (const weight & entry : group->weights) {
            if (entry.source == source) {
                return entry.slots;
            }
        }

        return source;
    }
};

llama_moe_expert_cache::llama_moe_expert_cache(const llama_model &                 model,
                                               const std::vector<ggml_backend_t> & backends,
                                               uint32_t                            n_cache) :
    pimpl(std::make_unique<impl>(model, backends, n_cache)) {
    llama_moe_cache_validate_model(model, n_cache);
}

llama_moe_expert_cache::~llama_moe_expert_cache() = default;

llama_moe_cache_binding llama_moe_expert_cache::bind(ggml_context *       ctx,
                                                     ggml_backend_sched_t sched,
                                                     int                  il,
                                                     ggml_tensor *        ids,
                                                     ggml_tensor *        token_priority,
                                                     ggml_tensor *        epoch,
                                                     ggml_tensor *        up,
                                                     ggml_tensor *        gate,
                                                     ggml_tensor *        down,
                                                     ggml_tensor *        gate_up) {
    const std::vector<ggml_tensor *> source_tensors = impl::sources(up, gate, down, gate_up);
    if (source_tensors.empty()) {
        return { up, gate, down, gate_up, ids };
    }

    impl::group * group = pimpl->find(il, source_tensors);
    if (pimpl->model.hparams.no_alloc) {
        return { up, gate, down, gate_up, ids };
    }

    ggml_tensor *                     logical_ids = ggml_is_contiguous(ids) ? ids : ggml_cont(ctx, ids);
    const ggml_backend_moe_cache_plan plan =
        group->api->build_plan(group->handle, ctx, logical_ids, token_priority, epoch);
    if (plan.execution == nullptr || plan.selectors == nullptr ||
        !ggml_backend_dev_supports_op(ggml_backend_get_device(group->backend), plan.execution)) {
        throw std::runtime_error("MoE expert cache: backend cannot build route plan");
    }

    ggml_format_name(plan.execution, "blk.%d.moe_cache_plan", il);
    ggml_format_name(plan.selectors, "blk.%d.moe_cache_ids", il);
    ggml_backend_sched_set_tensor_backend(sched, plan.execution, group->backend);
    ggml_backend_sched_set_tensor_backend(sched, plan.selectors, group->backend);

    return {
        impl::cached(group, up), impl::cached(group, gate), impl::cached(group, down), impl::cached(group, gate_up),
        plan.selectors,
    };
}

void llama_moe_expert_cache::synchronize() {
    for (ggml_backend_t backend : pimpl->pending) {
        ggml_backend_synchronize(backend);
    }
    pimpl->pending.clear();
}

void llama_moe_expert_cache::begin_batch() {
    ++pimpl->batch_epoch;
    if (pimpl->batch_epoch == 0) {
        ++pimpl->batch_epoch;
    }
}

uint32_t llama_moe_expert_cache::epoch() const {
    return pimpl->batch_epoch;
}

void llama_moe_expert_cache::print_info() {
    if (pimpl->info_printed || pimpl->groups.empty()) {
        return;
    }
    pimpl->info_printed = true;

    size_t total = 0;
    size_t host  = 0;
    for (const auto & group : pimpl->groups) {
        total += group->buffer_size;
        for (const impl::weight & weight : group->weights) {
            host += ggml_nbytes(weight.source);
        }
    }

    LLAMA_LOG_INFO(
        "MoE expert cache: host expert weights = %.2f MiB, GPU cache = %.2f MiB, experts = %u / %u, layers = %zu\n",
        host / 1048576.0, total / 1048576.0, pimpl->n_cache, pimpl->model.hparams.n_expert, pimpl->groups.size());
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_expert_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> result;
    for (const auto & group : pimpl->groups) {
        result[group->buft] += group->buffer_size;
    }
    return result;
}
