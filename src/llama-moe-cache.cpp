#include "llama-moe-cache.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-alloc.h"
#include "ggml-cuda.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

struct llama_moe_expert_cache::impl {
    struct weight {
        const ggml_tensor * source;
        ggml_tensor * slots;
    };

    struct group {
        int il;
        int32_t n_expert;
        int32_t n_cache;

        ggml_backend_t backend;
        ggml_backend_buffer_type_t buft;

        ggml_context_ptr context;
        ggml_backend_buffer_ptr buffer;
        size_t buffer_size = 0;
        ggml_tensor * state = nullptr;

        std::vector<weight> weights;
        std::vector<ggml_backend_cuda_expert_source> sources;
        std::vector<int32_t> expert_to_cache;
        std::vector<int32_t> cache_to_expert;
        std::vector<uint64_t> last_used;
        uint64_t use_clock = 0;

        ggml_backend_cuda_expert_cache_desc device_desc = {};
        bool device_resolver = false;
    };

    const llama_model & model;
    std::vector<ggml_backend_t> backends;
    std::vector<std::unique_ptr<group>> groups;
    std::unordered_map<const ggml_tensor *, group *> groups_by_anchor;
    std::unordered_set<ggml_backend_t> pending_backends;

    ggml_backend_dev_t cache_device = nullptr;
    uint32_t n_cache_experts;
    bool info_printed = false;

    static size_t align_offset(size_t offset, size_t alignment) {
        return (offset + alignment - 1) & ~(alignment - 1);
    }

    impl(
            const llama_model & model,
            const std::vector<ggml_backend_t> & backends,
            uint32_t n_cache_experts) :
        model(model),
        backends(backends),
        n_cache_experts(n_cache_experts) {
    }

    ggml_backend_t backend_for_layer(int il) const {
        ggml_backend_dev_t device = model.dev_layer(il);
        for (ggml_backend_t backend : backends) {
            if (ggml_backend_get_device(backend) == device) {
                return backend;
            }
        }
        return nullptr;
    }

    static std::vector<ggml_tensor *> expert_sources(
            ggml_tensor * up,
            ggml_tensor * gate,
            ggml_tensor * down,
            ggml_tensor * gate_up) {
        std::vector<ggml_tensor *> sources;
        for (ggml_tensor * source : { gate_up, up, gate, down }) {
            if (source != nullptr && std::find(sources.begin(), sources.end(), source) == sources.end()) {
                sources.push_back(source);
            }
        }
        return sources;
    }

    group * create_group(
            int il,
            const std::vector<ggml_tensor *> & sources) {
        if (sources.empty()) {
            throw std::runtime_error("MoE expert cache: no expert weights");
        }

        const int64_t n_expert = sources.front()->ne[2];
        if (il < 0 || n_expert <= 0 || n_cache_experts > (uint64_t) n_expert) {
            throw std::runtime_error("MoE expert cache: invalid layer or expert count");
        }

        ggml_backend_t backend = backend_for_layer(il);
        if (backend == nullptr) {
            throw std::runtime_error("MoE expert cache: failed to find the layer backend");
        }

        ggml_backend_dev_t device = ggml_backend_get_device(backend);
        if (device == nullptr || ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            throw std::runtime_error("MoE expert cache: layer backend is not a GPU");
        }
        if (cache_device != nullptr && cache_device != device) {
            throw std::runtime_error("MoE expert cache: multiple cache devices are not supported");
        }
        cache_device = device;

        for (const ggml_tensor * source : sources) {
            if (source->ne[2] != n_expert || source->ne[3] != 1) {
                throw std::runtime_error("MoE expert cache: incompatible expert tensor dimensions");
            }
            if (source->buffer == nullptr || !ggml_backend_buffer_is_host(source->buffer)) {
                throw std::runtime_error("MoE expert cache: expert weights must be host resident");
            }
            if (!model.hparams.no_alloc && source->data == nullptr) {
                throw std::runtime_error("MoE expert cache: expert weight data is unavailable");
            }
        }

        const bool needs_planner_state = n_cache_experts < (uint64_t) n_expert;
        const ggml_init_params params = {
            /*.mem_size   =*/ (sources.size() + 1) * ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };

        auto result = std::make_unique<group>();
        result->il = il;
        result->n_expert = n_expert;
        result->n_cache = n_cache_experts;
        result->backend = backend;
        result->buft = ggml_backend_get_default_buffer_type(backend);
        result->context.reset(ggml_init(params));
        if (!result->context) {
            throw std::runtime_error("MoE expert cache: failed to create tensor context");
        }

        for (ggml_tensor * source : sources) {
            const int64_t ne[4] = { source->ne[0], source->ne[1], (int64_t) n_cache_experts, 1 };
            ggml_tensor * slots = ggml_new_tensor(result->context.get(), source->type, 4, ne);
            for (int i = 0; i < GGML_MAX_DIMS; ++i) {
                slots->nb[i] = source->nb[i];
            }
            ggml_format_name(slots, "%s#moe_cache", source->name);
            result->weights.push_back({ source, slots });
        }
        result->sources.resize(result->weights.size());
        for (size_t i = 0; i < result->sources.size(); ++i) {
            result->sources[i] = {
                GGML_CUDA_EXPERT_SOURCE_MAGIC,
                (uint32_t) result->n_expert,
                nullptr,
            };
            result->weights[i].slots->extra = &result->sources[i];
        }

        size_t state_offset = sizeof(ggml_backend_cuda_expert_cache_state);
        if (needs_planner_state) {
            result->device_desc.expert_to_cache_offset = align_offset(state_offset, alignof(int32_t));
            state_offset = result->device_desc.expert_to_cache_offset + n_expert * sizeof(int32_t);
            result->device_desc.cache_to_expert_offset = align_offset(state_offset, alignof(int32_t));
            state_offset = result->device_desc.cache_to_expert_offset + n_cache_experts * sizeof(int32_t);
            result->device_desc.last_used_offset = align_offset(state_offset, alignof(uint64_t));
            state_offset = result->device_desc.last_used_offset + n_cache_experts * sizeof(uint64_t);
            result->device_desc.fill_expert_offset = align_offset(state_offset, alignof(int32_t));
            state_offset = result->device_desc.fill_expert_offset + n_cache_experts * sizeof(int32_t);
            result->device_desc.fill_slot_offset = align_offset(state_offset, alignof(int32_t));
            state_offset = result->device_desc.fill_slot_offset + n_cache_experts * sizeof(int32_t);
        }
        result->device_desc.state_size = align_offset(state_offset, GGML_MEM_ALIGN);

        result->state = ggml_new_tensor_1d(
            result->context.get(), GGML_TYPE_I8, result->device_desc.state_size);
        ggml_format_name(result->state, "blk.%d.moe_cache_state", il);

        result->buffer_size = ggml_backend_alloc_ctx_tensors_from_buft_size(
            result->context.get(), result->buft);

        if (!model.hparams.no_alloc) {
            result->buffer.reset(ggml_backend_alloc_ctx_tensors(result->context.get(), backend));
            if (!result->buffer) {
                throw std::runtime_error("MoE expert cache: failed to allocate device cache");
            }
            ggml_backend_buffer_set_usage(result->buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            ggml_backend_buffer_clear(result->buffer.get(), 0);
            result->buffer_size = ggml_backend_buffer_get_size(result->buffer.get());

            {
                std::vector<uint8_t> state_data(result->device_desc.state_size, 0);
                if (needs_planner_state) {
                    auto * expert_to_cache = reinterpret_cast<int32_t *>(
                        state_data.data() + result->device_desc.expert_to_cache_offset);
                    auto * cache_to_expert = reinterpret_cast<int32_t *>(
                        state_data.data() + result->device_desc.cache_to_expert_offset);
                    std::fill(expert_to_cache, expert_to_cache + n_expert, -1);
                    std::fill(cache_to_expert, cache_to_expert + n_cache_experts, -1);
                }
                ggml_backend_tensor_set(
                    result->state, state_data.data(), 0, result->device_desc.state_size);
            }
            pending_backends.insert(backend);
        }

        result->expert_to_cache.assign(n_expert, -1);
        result->cache_to_expert.assign(n_cache_experts, -1);
        result->last_used.assign(n_cache_experts, 0);

        result->device_desc.magic = GGML_CUDA_EXPERT_CACHE_MAGIC;
        result->device_desc.version = GGML_CUDA_EXPERT_CACHE_VERSION;
        result->device_desc.n_expert = n_expert;
        result->device_desc.n_cache = n_cache_experts;
        result->device_desc.n_weights = result->weights.size();

        for (size_t i = 0; i < result->sources.size(); ++i) {
            result->sources[i].weight = &result->device_desc.weights[i];
        }

        if (!model.hparams.no_alloc && needs_planner_state) {
            ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(device);
            for (size_t i = 0; i < result->weights.size(); ++i) {
                const weight & cached = result->weights[i];
                if (ggml_backend_buffer_get_type(cached.source->buffer) != host_buft) {
                    result->device_desc.magic = 0;
                    break;
                }

                const uint8_t * host_base = static_cast<const uint8_t *>(
                    ggml_backend_buffer_get_base(cached.source->buffer));
                result->device_desc.weights[i].host_data = host_base;
                result->device_desc.weights[i].host_offset =
                    static_cast<const uint8_t *>(cached.source->data) - host_base;
                result->device_desc.weights[i].expert_size = cached.source->nb[2];
            }
        }

        if (!model.hparams.no_alloc && n_cache_experts == (uint64_t) n_expert) {
            for (int32_t expert = 0; expert < n_expert; ++expert) {
                result->expert_to_cache[expert] = expert;
                result->cache_to_expert[expert] = expert;
            }
            for (const weight & cached : result->weights) {
                ggml_backend_tensor_set_async(
                    backend,
                    cached.slots,
                    cached.source->data,
                    0,
                    ggml_nbytes(cached.slots));
            }
            pending_backends.insert(backend);
        }

        group * value = result.get();
        groups_by_anchor.emplace(sources.front(), value);
        groups.push_back(std::move(result));
        return value;
    }

    group * find_or_create_group(
            int il,
            const std::vector<ggml_tensor *> & sources) {
        if (sources.empty()) {
            throw std::runtime_error("MoE expert cache: no expert weights");
        }
        auto it = groups_by_anchor.find(sources.front());
        return it == groups_by_anchor.end() ? create_group(il, sources) : it->second;
    }

    static ggml_tensor * cached_weight(group * cache_group, ggml_tensor * source) {
        if (source == nullptr) {
            return nullptr;
        }
        for (const weight & cached : cache_group->weights) {
            if (cached.source == source) {
                return cached.slots;
            }
        }
        GGML_ABORT("MoE expert cache: weight is not part of the cache group");
    }

    void print_stats() const {
        if (model.hparams.no_alloc) {
            return;
        }

        uint64_t total_resolve_calls = 0;
        uint64_t total_update_touches = 0;
        uint64_t total_read_only_touches = 0;
        uint64_t total_resident_routes = 0;
        uint64_t total_streamed_routes = 0;
        uint64_t total_cache_hits = 0;
        uint64_t total_cache_misses = 0;
        uint64_t total_evictions = 0;
        uint64_t total_h2d_bytes = 0;
        uint64_t total_host_expert_bytes = 0;
        double total_fill_ms = 0.0;

        for (const auto & cache_group : groups) {
            if (!cache_group->device_resolver) {
                continue;
            }

            ggml_backend_cuda_expert_cache_state state = {};
            ggml_backend_tensor_get(cache_group->state, &state, 0, sizeof(state));
            total_resolve_calls += state.stats.resolve_calls;
            total_update_touches += state.stats.update_touches;
            total_read_only_touches += state.stats.read_only_touches;
            total_resident_routes += state.stats.resident_routes;
            total_streamed_routes += state.stats.streamed_routes;
            total_cache_hits += state.stats.cache_hits;
            total_cache_misses += state.stats.cache_misses;
            total_evictions += state.stats.evictions;
            total_h2d_bytes += state.stats.h2d_bytes;
            total_host_expert_bytes += state.stats.host_expert_bytes;
            const uint64_t wall_clock_hz = cache_group->device_desc.wall_clock_hz;
            if (wall_clock_hz > 0) {
                total_fill_ms += state.stats.fill_ticks * 1000.0 / wall_clock_hz;
            }
        }

        LLAMA_LOG_INFO(
            "MoE expert cache: resolves = %" PRIu64 ", updates = %" PRIu64
            ", read-only routes = %" PRIu64 ", resident routes = %" PRIu64
            ", streamed routes = %" PRIu64 ", hits = %" PRIu64 ", misses = %" PRIu64
            ", evictions = %" PRIu64 ", H2D = %.2f MiB, host expert bytes = %.2f MiB, fill = %.2f ms\n",
            total_resolve_calls,
            total_update_touches,
            total_read_only_touches,
            total_resident_routes,
            total_streamed_routes,
            total_cache_hits,
            total_cache_misses,
            total_evictions,
            total_h2d_bytes / 1024.0 / 1024.0,
            total_host_expert_bytes / 1024.0 / 1024.0,
            total_fill_ms);
    }

    void reset_stats() {
        const ggml_backend_cuda_expert_cache_stats zero = {};
        for (const auto & cache_group : groups) {
            if (cache_group->device_resolver) {
                ggml_backend_tensor_set_async(
                    cache_group->backend,
                    cache_group->state,
                    &zero,
                    offsetof(ggml_backend_cuda_expert_cache_state, stats),
                    sizeof(zero));
            }
        }
    }
};

llama_moe_expert_cache::llama_moe_expert_cache(
        const llama_model & model,
        const std::vector<ggml_backend_t> & backends,
        uint32_t n_cache_experts) :
    pimpl(std::make_unique<impl>(model, backends, n_cache_experts)) {
    if (model.hparams.n_expert == 0) {
        throw std::runtime_error("MoE expert cache: model has no routed experts");
    }
    if (n_cache_experts < model.hparams.n_expert_used) {
        throw std::runtime_error(
            "MoE expert cache: capacity must be at least the number of experts used per token (" +
            std::to_string(model.hparams.n_expert_used) + ")");
    }
    if (n_cache_experts >= model.hparams.n_expert) {
        throw std::runtime_error(
            "MoE expert cache: capacity must be smaller than the model expert count (" +
            std::to_string(model.hparams.n_expert) +
            "); omit --moe-cache-experts when all experts fit in VRAM");
    }
    if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        throw std::runtime_error("MoE expert cache: tensor parallelism is not supported");
    }
    LLAMA_LOG_INFO(
        "%s: capacity = %u / %u routed experts per host-backed MoE layer\n",
        __func__, n_cache_experts, model.hparams.n_expert);
}

llama_moe_expert_cache::~llama_moe_expert_cache() {
}

llama_moe_cache_binding llama_moe_expert_cache::bind(
        ggml_context * ctx,
        ggml_backend_sched_t sched,
        int il,
        ggml_tensor * ids,
        ggml_tensor * policy,
        ggml_tensor * up,
        ggml_tensor * gate,
        ggml_tensor * down,
        ggml_tensor * gate_up) {
    const std::vector<ggml_tensor *> sources = impl::expert_sources(up, gate, down, gate_up);

    ggml_backend_t backend = pimpl->backend_for_layer(il);
    if (backend == nullptr) {
        throw std::runtime_error("MoE expert cache: failed to find the layer backend");
    }
    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    if (device == nullptr || ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        throw std::runtime_error("MoE expert cache: routed expert layer is not assigned to a GPU");
    }
    bool all_on_device = !sources.empty();
    bool all_on_host = !sources.empty();
    for (const ggml_tensor * source : sources) {
        all_on_device = all_on_device &&
            source->buffer != nullptr &&
            !ggml_backend_buffer_is_host(source->buffer) &&
            ggml_backend_buft_get_device(ggml_backend_buffer_get_type(source->buffer)) == device;
        all_on_host = all_on_host &&
            source->buffer != nullptr &&
            ggml_backend_buffer_is_host(source->buffer);
    }
    if (all_on_device) {
        throw std::runtime_error("MoE expert cache: routed expert weights are not host resident");
    }
    if (!all_on_host) {
        throw std::runtime_error("MoE expert cache: routed expert weights have mixed placement");
    }

    impl::group * cache_group = pimpl->find_or_create_group(il, sources);

    if (pimpl->model.hparams.no_alloc) {
        return {
            /*.up      =*/ up,
            /*.gate    =*/ gate,
            /*.down    =*/ down,
            /*.gate_up =*/ gate_up,
            /*.ids     =*/ ids,
        };
    }

    if (pimpl->n_cache_experts == (uint64_t) cache_group->n_expert) {
        return {
            /*.up      =*/ impl::cached_weight(cache_group, up),
            /*.gate    =*/ impl::cached_weight(cache_group, gate),
            /*.down    =*/ impl::cached_weight(cache_group, down),
            /*.gate_up =*/ impl::cached_weight(cache_group, gate_up),
            /*.ids     =*/ ids,
        };
    }

    ggml_tensor * ids_cont = ggml_is_contiguous(ids) ? ids : ggml_cont(ctx, ids);

    if (cache_group->state != nullptr && cache_group->device_desc.magic == GGML_CUDA_EXPERT_CACHE_MAGIC) {
        const int state_index = policy != nullptr ? 2 : 1;
        ggml_tensor * args[3 + GGML_CUDA_EXPERT_CACHE_MAX_WEIGHTS] = {};
        args[0] = ids_cont;
        if (policy != nullptr) {
            args[1] = policy;
        }
        args[state_index] = cache_group->state;
        for (size_t i = 0; i < cache_group->weights.size(); ++i) {
            args[state_index + 1 + i] = cache_group->weights[i].slots;
        }

        ggml_tensor * cache_ids = ggml_custom_4d(
            ctx,
            GGML_TYPE_I32,
            ids_cont->ne[0],
            ids_cont->ne[1],
            ids_cont->ne[2],
            ids_cont->ne[3],
            args,
            state_index + 1 + cache_group->weights.size(),
            nullptr,
            1,
            &cache_group->device_desc);

        if (ggml_backend_dev_supports_op(device, cache_ids)) {
            ggml_format_name(cache_ids, "blk.%d.moe_cache_ids", il);
            ggml_backend_sched_set_tensor_backend(sched, cache_ids, backend);
            if (policy != nullptr) {
                ggml_backend_sched_set_tensor_backend(sched, policy, backend);
            }
            cache_group->device_resolver = true;
            return {
                /*.up      =*/ impl::cached_weight(cache_group, up),
                /*.gate    =*/ impl::cached_weight(cache_group, gate),
                /*.down    =*/ impl::cached_weight(cache_group, down),
                /*.gate_up =*/ impl::cached_weight(cache_group, gate_up),
                /*.ids     =*/ cache_ids,
            };
        }
    }

    return {
        /*.up      =*/ up,
        /*.gate    =*/ gate,
        /*.down    =*/ down,
        /*.gate_up =*/ gate_up,
        /*.ids     =*/ ids,
    };
}

void llama_moe_expert_cache::synchronize() {
    for (ggml_backend_t backend : pimpl->pending_backends) {
        ggml_backend_synchronize(backend);
    }
    pimpl->pending_backends.clear();
}

void llama_moe_expert_cache::print_info() {
    if (pimpl->info_printed) {
        return;
    }
    pimpl->info_printed = true;

    if (pimpl->groups.empty()) {
        throw std::runtime_error("MoE expert cache: model has no supported routed expert layers");
    }

    size_t total = 0;
    size_t host = 0;
    for (const auto & cache_group : pimpl->groups) {
        total += cache_group->buffer_size;
        for (const auto & cached : cache_group->weights) {
            host += ggml_nbytes(cached.source);
        }
    }
    LLAMA_LOG_INFO(
        "MoE expert cache: host expert weights = %.2f MiB, GPU cache = %.2f MiB, experts = %u / %u, layers = %zu\n",
        host / 1024.0 / 1024.0,
        total / 1024.0 / 1024.0,
        pimpl->n_cache_experts,
        pimpl->model.hparams.n_expert,
        pimpl->groups.size());
}

void llama_moe_expert_cache::print_stats() const {
    pimpl->print_stats();
}

void llama_moe_expert_cache::reset_stats() {
    pimpl->reset_stats();
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_expert_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> result;
    for (const auto & cache_group : pimpl->groups) {
        result[cache_group->buft] += cache_group->buffer_size;
    }
    return result;
}
