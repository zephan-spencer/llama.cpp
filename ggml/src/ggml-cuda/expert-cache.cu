#include "expert-cache.cuh"
#include "mmid.cuh"

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
    uint32_t n_routes;
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

static __global__ void expert_cache_record_plan_kernel(
        uint8_t * state_data,
        expert_cache_params params) {
    auto * header = reinterpret_cast<ggml_backend_cuda_expert_cache_state *>(state_data);
    if (threadIdx.x != 0) {
        return;
    }

    uint64_t bytes_per_fill = 0;
    for (uint32_t weight = 0; weight < params.n_weights; ++weight) {
        bytes_per_fill += params.weights[weight].size;
    }

    header->stats.resolve_calls++;
    header->stats.update_touches += header->n_active;
    header->stats.read_only_touches += header->n_read_only;
    header->stats.resident_routes += params.n_routes;
    header->stats.resident_routes -= header->n_streamed;
    header->stats.streamed_routes += header->n_streamed;
    header->stats.cache_hits += header->n_resident;
    header->stats.cache_misses += header->n_misses;
    header->stats.evictions += header->n_evictions;
    header->stats.h2d_bytes += header->n_fills * bytes_per_fill;
    header->stats.host_expert_bytes += header->n_host_experts * bytes_per_fill;
    header->copy_blocks_done = 0;
    if (header->n_fills > 0) {
        header->fill_start_ticks = wall_clock64();
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
    if (desc == nullptr || desc->n_weights == 0 ||
            desc->n_weights > GGML_CUDA_EXPERT_CACHE_MAX_WEIGHTS) {
        return false;
    }
    if (dst->type != GGML_TYPE_I32 || dst->src[0] == nullptr || dst->src[0]->type != GGML_TYPE_I32 || !ggml_is_contiguous(dst->src[0])) {
        return false;
    }
    if (dst->src[1] == nullptr || dst->src[1]->type != GGML_TYPE_I8 || !ggml_is_contiguous(dst->src[1])) {
        return false;
    }
    if (desc->n_expert == 0 || desc->n_cache == 0 || desc->n_cache > desc->n_expert) {
        return false;
    }
    int state_index = 1;
    if (dst->src[2] != nullptr && dst->src[2]->type == GGML_TYPE_I8 &&
            ggml_is_contiguous(dst->src[2]) && dst->src[2]->ne[0] == (int64_t) desc->state_size &&
            dst->src[2]->ne[1] == 1 && dst->src[2]->ne[2] == 1 && dst->src[2]->ne[3] == 1) {
        state_index = 2;
        if (dst->src[1]->ne[0] != dst->src[0]->ne[1] || dst->src[1]->ne[1] != 1 ||
                dst->src[1]->ne[2] != 1 || dst->src[1]->ne[3] != 1) {
            return false;
        }
    }
    if (dst->src[state_index] == nullptr || ggml_nbytes(dst->src[state_index]) < desc->state_size) {
        return false;
    }
    for (uint32_t i = 0; i < desc->n_weights; ++i) {
        if (dst->src[state_index + 1 + i] == nullptr ||
                desc->weights[i].host_data == nullptr ||
                desc->weights[i].expert_size == 0 ||
                desc->weights[i].expert_size % sizeof(uint4) != 0) {
            return false;
        }
    }
    return true;
}

static void expert_cache_prepare_clock(ggml_backend_cuda_expert_cache_desc * desc) {
    if (desc->wall_clock_hz != 0) {
        return;
    }

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

void ggml_cuda_expert_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_backend_cuda_expert_cache_desc * desc = expert_cache_desc(dst);
    GGML_ASSERT(desc != nullptr);

    expert_cache_prepare_clock(desc);

    int state_index = 1;
    const ggml_tensor * policy_src = nullptr;
    if (dst->src[2] != nullptr && dst->src[2]->type == GGML_TYPE_I8 &&
            dst->src[2]->ne[0] == (int64_t) desc->state_size && dst->src[2]->ne[1] == 1 &&
            dst->src[2]->ne[2] == 1 && dst->src[2]->ne[3] == 1) {
        state_index = 2;
        policy_src = dst->src[1];
    }

    expert_cache_params params = {};
    params.n_expert = desc->n_expert;
    params.n_cache = desc->n_cache;
    params.n_weights = desc->n_weights;
    params.n_routes = ggml_nelements(dst->src[0]);
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
            static_cast<uint8_t *>(dst->src[state_index + 1 + i]->data),
            desc->weights[i].expert_size,
        };
    }

    uint8_t * state_data = static_cast<uint8_t *>(dst->src[state_index]->data);
    auto * header = reinterpret_cast<ggml_backend_cuda_expert_cache_state *>(state_data);

    ggml_cuda_expert_plan plan = {};
    plan.fill_expert = reinterpret_cast<int32_t *>(state_data + params.fill_expert_offset);
    plan.fill_slot = reinterpret_cast<int32_t *>(state_data + params.fill_slot_offset);
    plan.cache_ids = static_cast<int32_t *>(dst->data);
    plan.expert_to_cache = reinterpret_cast<int32_t *>(state_data + params.expert_to_cache_offset);
    plan.cache_to_expert = reinterpret_cast<int32_t *>(state_data + params.cache_to_expert_offset);
    plan.last_used = reinterpret_cast<uint64_t *>(state_data + params.last_used_offset);
    plan.use_clock = &header->use_clock;
    plan.n_active = &header->n_active;
    plan.n_resident = &header->n_resident;
    plan.n_miss = &header->n_misses;
    plan.n_fill = &header->n_fills;
    plan.n_evictions = &header->n_evictions;
    plan.n_read_only = &header->n_read_only;
    plan.n_streamed = &header->n_streamed;
    plan.n_host_experts = &header->n_host_experts;
    plan.policy = policy_src != nullptr ? static_cast<const uint8_t *>(policy_src->data) : nullptr;

    const int n_expert_used = dst->src[0]->ne[0];
    const int n_tokens = ggml_nelements(dst->src[0]) / n_expert_used;
    ggml_cuda_launch_expert_cache_plan(
        static_cast<const int32_t *>(dst->src[0]->data),
        plan,
        desc->n_expert,
        n_tokens,
        n_expert_used,
        desc->n_cache,
        ctx.stream());
    CUDA_CHECK(cudaGetLastError());

    expert_cache_record_plan_kernel<<<1, 1, 0, ctx.stream()>>>(
        state_data,
        params);
    CUDA_CHECK(cudaGetLastError());

    expert_cache_copy_kernel<<<32, 256, 0, ctx.stream()>>>(
        state_data,
        params);
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
