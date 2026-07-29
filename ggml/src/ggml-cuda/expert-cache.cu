#include "expert-cache.cuh"

#include "ggml-cuda.h"
#include "ggml-impl.h"

#include <cstdint>
#include <cstring>

#if defined(GGML_USE_HIP)

struct expert_cache_weight {
    const uint8_t * host;
    uint8_t * cache;
    uint64_t size;
};

struct expert_cache_params {
    expert_cache_weight weights[GGML_CUDA_EXPERT_CACHE_MAX_WEIGHTS];
    uint32_t n_expert;
    uint32_t n_cache;
    uint32_t n_weights;
    uint64_t expert_to_cache_offset;
    uint64_t cache_to_expert_offset;
    uint64_t last_used_offset;
    uint64_t fill_expert_offset;
    uint64_t fill_slot_offset;
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
    if (desc->magic != GGML_CUDA_EXPERT_CACHE_MAGIC || desc->version != GGML_CUDA_EXPERT_CACHE_VERSION) {
        return nullptr;
    }
    return desc;
}

static __global__ void expert_cache_resolve_kernel(
        const int32_t * ids,
        int32_t * cache_ids,
        uint8_t * state_data,
        int64_t n_ids,
        expert_cache_params params) {
    extern __shared__ int32_t scratch[];

    int32_t * requested = scratch;
    int32_t * unique = requested + params.n_expert;

    auto * header = reinterpret_cast<ggml_backend_cuda_expert_cache_state *>(state_data);
    auto * expert_to_cache = reinterpret_cast<int32_t *>(state_data + params.expert_to_cache_offset);
    auto * cache_to_expert = reinterpret_cast<int32_t *>(state_data + params.cache_to_expert_offset);
    auto * last_used = reinterpret_cast<uint64_t *>(state_data + params.last_used_offset);
    auto * fill_expert = reinterpret_cast<int32_t *>(state_data + params.fill_expert_offset);
    auto * fill_slot = reinterpret_cast<int32_t *>(state_data + params.fill_slot_offset);

    for (uint32_t expert = threadIdx.x; expert < params.n_expert; expert += blockDim.x) {
        requested[expert] = 0;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int32_t n_unique = 0;
        int32_t n_fills = 0;

        for (int64_t i = 0; i < n_ids; ++i) {
            const int32_t expert = ids[i];
            assert(expert >= 0 && expert < (int32_t) params.n_expert);
            if (!requested[expert]) {
                requested[expert] = 1;
                unique[n_unique++] = expert;
            }
        }
        assert(n_unique <= (int32_t) params.n_cache);

        header->stats.resolve_calls++;

        for (int32_t i = 0; i < n_unique; ++i) {
            const int32_t expert = unique[i];
            if (expert_to_cache[expert] >= 0) {
                header->stats.cache_hits++;
                continue;
            }

            header->stats.cache_misses++;

            int32_t victim = -1;
            for (uint32_t slot = 0; slot < params.n_cache; ++slot) {
                if (cache_to_expert[slot] < 0) {
                    victim = slot;
                    break;
                }
            }

            if (victim < 0) {
                uint64_t oldest = UINT64_MAX;
                for (uint32_t slot = 0; slot < params.n_cache; ++slot) {
                    const int32_t resident = cache_to_expert[slot];
                    if (!requested[resident] && last_used[slot] < oldest) {
                        oldest = last_used[slot];
                        victim = slot;
                    }
                }
            }
            assert(victim >= 0);

            const int32_t evicted = cache_to_expert[victim];
            if (evicted >= 0) {
                expert_to_cache[evicted] = -1;
                header->stats.evictions++;
            }
            cache_to_expert[victim] = expert;
            fill_expert[n_fills] = expert;
            fill_slot[n_fills] = victim;
            n_fills++;

            for (uint32_t weight = 0; weight < params.n_weights; ++weight) {
                header->stats.h2d_bytes += params.weights[weight].size;
            }
        }

        for (int32_t i = 0; i < n_fills; ++i) {
            const int32_t expert = fill_expert[i];
            const int32_t slot = fill_slot[i];
            expert_to_cache[expert] = slot;
            cache_to_expert[slot] = expert;
        }

        const uint64_t use_clock = ++header->use_clock;
        for (int32_t i = 0; i < n_unique; ++i) {
            const int32_t expert = unique[i];
            const int32_t slot = expert_to_cache[expert];
            assert(slot >= 0);
            last_used[slot] = use_clock;
        }

        for (int64_t i = 0; i < n_ids; ++i) {
            const int32_t slot = expert_to_cache[ids[i]];
            assert(slot >= 0);
            cache_ids[i] = slot;
        }

        header->n_fills = n_fills;
        header->copy_blocks_done = 0;
        if (n_fills > 0) {
            header->fill_start_ticks = wall_clock64();
        }
    }
}

static __global__ void expert_cache_copy_kernel(
        uint8_t * state_data,
        expert_cache_params params) {
    auto * header = reinterpret_cast<ggml_backend_cuda_expert_cache_state *>(state_data);
    const auto * fill_expert = reinterpret_cast<const int32_t *>(state_data + params.fill_expert_offset);
    const auto * fill_slot = reinterpret_cast<const int32_t *>(state_data + params.fill_slot_offset);
    const uint32_t n_fills = header->n_fills;

    uint64_t vectors_per_fill = 0;
    for (uint32_t weight = 0; weight < params.n_weights; ++weight) {
        vectors_per_fill += params.weights[weight].size / sizeof(uint4);
    }

    const uint64_t n_vectors = n_fills * vectors_per_fill;
    for (uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
            i < n_vectors;
            i += (uint64_t) gridDim.x * blockDim.x) {
        const uint32_t fill = i / vectors_per_fill;
        uint64_t weight_offset = i - fill * vectors_per_fill;
        uint32_t weight = 0;
        while (weight_offset >= params.weights[weight].size / sizeof(uint4)) {
            weight_offset -= params.weights[weight].size / sizeof(uint4);
            weight++;
        }

        const expert_cache_weight cached = params.weights[weight];
        const uint64_t expert = fill_expert[fill];
        const uint64_t slot = fill_slot[fill];
        const auto * src = reinterpret_cast<const uint4 *>(cached.host + expert * cached.size);
        auto * dst = reinterpret_cast<uint4 *>(cached.cache + slot * cached.size);
        dst[weight_offset] = src[weight_offset];
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        const uint32_t completed = atomicAdd(&header->copy_blocks_done, 1);
        if (completed + 1 == gridDim.x && n_fills > 0) {
            header->stats.fill_ticks += wall_clock64() - header->fill_start_ticks;
        }
    }
}

bool ggml_cuda_expert_cache_supported(const ggml_tensor * dst) {
    ggml_backend_cuda_expert_cache_desc * desc = expert_cache_desc(dst);
    if (desc == nullptr || desc->n_weights == 0 || desc->n_weights > GGML_CUDA_EXPERT_CACHE_MAX_WEIGHTS) {
        return false;
    }
    if (dst->type != GGML_TYPE_I32 || dst->src[0] == nullptr || dst->src[0]->type != GGML_TYPE_I32) {
        return false;
    }
    if (dst->src[1] == nullptr || dst->src[1]->type != GGML_TYPE_I8 || !ggml_is_contiguous(dst->src[1])) {
        return false;
    }
    if (desc->n_expert == 0 || desc->n_cache == 0 || desc->n_cache > desc->n_expert) {
        return false;
    }
    if (ggml_nelements(dst->src[0]) > desc->n_cache || ggml_nbytes(dst->src[1]) < desc->state_size) {
        return false;
    }
    for (uint32_t i = 0; i < desc->n_weights; ++i) {
        if (dst->src[2 + i] == nullptr ||
                desc->weights[i].host_data == nullptr ||
                desc->weights[i].expert_size == 0 ||
                desc->weights[i].expert_size % sizeof(uint4) != 0) {
            return false;
        }
    }
    return true;
}

void ggml_cuda_expert_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_backend_cuda_expert_cache_desc * desc = expert_cache_desc(dst);
    GGML_ASSERT(desc != nullptr);

    expert_cache_params params = {};
    params.n_expert = desc->n_expert;
    params.n_cache = desc->n_cache;
    params.n_weights = desc->n_weights;
    params.expert_to_cache_offset = desc->expert_to_cache_offset;
    params.cache_to_expert_offset = desc->cache_to_expert_offset;
    params.last_used_offset = desc->last_used_offset;
    params.fill_expert_offset = desc->fill_expert_offset;
    params.fill_slot_offset = desc->fill_slot_offset;

    for (uint32_t i = 0; i < desc->n_weights; ++i) {
        if (desc->weights[i].device_data == nullptr) {
            void * device_data = nullptr;
            const hipError_t error = hipHostGetDevicePointer(
                &device_data, const_cast<void *>(desc->weights[i].host_data), 0);
            if (error != hipSuccess) {
                GGML_ABORT("failed to map expert cache host weight: %s", hipGetErrorString(error));
            }
            desc->weights[i].device_data = device_data;
        }

        params.weights[i] = {
            static_cast<const uint8_t *>(desc->weights[i].device_data) + desc->weights[i].host_offset,
            static_cast<uint8_t *>(dst->src[2 + i]->data),
            desc->weights[i].expert_size,
        };
    }

    if (desc->wall_clock_hz == 0) {
        int device = 0;
        int wall_clock_khz = 0;
        hipError_t error = hipGetDevice(&device);
        if (error != hipSuccess) {
            GGML_ABORT("failed to get expert cache device: %s", hipGetErrorString(error));
        }
        error = hipDeviceGetAttribute(&wall_clock_khz, hipDeviceAttributeWallClockRate, device);
        if (error != hipSuccess) {
            GGML_ABORT("failed to get expert cache wall clock: %s", hipGetErrorString(error));
        }
        desc->wall_clock_hz = (uint64_t) wall_clock_khz * 1000;
    }

    const int64_t n_ids = ggml_nelements(dst->src[0]);
    const size_t shared_size = sizeof(int32_t) * (desc->n_expert + n_ids);
    expert_cache_resolve_kernel<<<1, 256, shared_size, ctx.stream()>>>(
        static_cast<const int32_t *>(dst->src[0]->data),
        static_cast<int32_t *>(dst->data),
        static_cast<uint8_t *>(dst->src[1]->data),
        n_ids,
        params);
    expert_cache_copy_kernel<<<32, 256, 0, ctx.stream()>>>(
        static_cast<uint8_t *>(dst->src[1]->data),
        params);
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
