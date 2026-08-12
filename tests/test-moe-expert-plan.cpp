#include "expert-cache-private.cuh"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cuda.h"

#include <hip/hip_runtime_api.h>
using cudaStream_t = hipStream_t;
#include "mmid.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static void hip_check(hipError_t error, const char * call) {
    if (error != hipSuccess) {
        throw std::runtime_error(std::string(call) + ": " + hipGetErrorString(error));
    }
}

template <typename T> class device_buffer {
  public:
    explicit device_buffer(size_t size) : size(size) { hip_check(hipMalloc(&ptr, size * sizeof(T)), "hipMalloc"); }

    ~device_buffer() {
        if (ptr != nullptr) {
            (void) hipFree(ptr);
        }
    }

    device_buffer(const device_buffer &)             = delete;
    device_buffer & operator=(const device_buffer &) = delete;

    void set(const std::vector<T> & values) {
        if (values.size() != size) {
            throw std::runtime_error("device buffer size mismatch");
        }
        hip_check(hipMemcpy(ptr, values.data(), size * sizeof(T), hipMemcpyHostToDevice), "hipMemcpyHostToDevice");
    }

    std::vector<T> get() const {
        std::vector<T> values(size);
        hip_check(hipMemcpy(values.data(), ptr, size * sizeof(T), hipMemcpyDeviceToHost), "hipMemcpyDeviceToHost");
        return values;
    }

    T *    ptr = nullptr;
    size_t size;
};

struct test_case {
    std::string           name;
    int32_t               n_expert;
    int32_t               n_cache;
    int32_t               n_tokens;
    int32_t               n_expert_used;
    std::vector<int32_t>  ids;
    std::vector<int32_t>  priority;
    std::vector<int32_t>  cache_to_expert;
    std::vector<uint64_t> last_used;
    uint64_t              use_clock = 0;
    std::vector<uint64_t> protected_epoch;
    int32_t               epoch = 1;
};

struct plan_result {
    std::vector<int32_t>  ids_src;
    std::vector<int32_t>  ids_dst;
    std::vector<int32_t>  expert_bounds;
    std::vector<int32_t>  route_first;
    std::vector<int32_t>  route_priority;
    std::vector<int32_t>  expert_order;
    std::vector<int32_t>  fill_expert;
    std::vector<int32_t>  fill_slot;
    std::vector<int32_t>  cache_ids;
    std::vector<int32_t>  expert_to_cache;
    std::vector<int32_t>  cache_to_expert;
    std::vector<uint64_t> last_used;
    std::vector<uint64_t> protected_epoch;
    uint64_t              use_clock      = 0;
    uint32_t              n_active       = 0;
    uint32_t              n_resident     = 0;
    uint32_t              n_miss         = 0;
    uint32_t              n_fill         = 0;
    uint32_t              n_evictions    = 0;
    uint32_t              n_streamed     = 0;
    uint32_t              n_host_experts = 0;
};

static std::vector<int32_t> make_expert_to_cache(const test_case & test) {
    std::vector<int32_t> result(test.n_expert, -1);
    for (int32_t slot = 0; slot < test.n_cache; ++slot) {
        const int32_t expert = test.cache_to_expert[slot];
        if (expert >= 0) {
            result[expert] = slot;
        }
    }
    return result;
}

static plan_result make_reference(const test_case & test) {
    const int32_t n_ids = test.n_tokens * test.n_expert_used;

    plan_result result;
    result.ids_src.resize(n_ids);
    result.ids_dst.resize(n_ids);
    result.expert_bounds.resize(test.n_expert + 1);
    result.route_first.resize(test.n_expert, n_ids);
    result.route_priority.resize(test.n_expert, 0);
    result.expert_order.resize(test.n_expert);
    result.fill_expert.resize(test.n_cache);
    result.fill_slot.resize(test.n_cache);
    result.cache_ids.resize(n_ids);
    result.expert_to_cache = make_expert_to_cache(test);
    result.cache_to_expert = test.cache_to_expert;
    result.last_used       = test.last_used;
    result.protected_epoch =
        test.protected_epoch.empty() ? std::vector<uint64_t>(test.n_cache, 0) : test.protected_epoch;
    result.use_clock = test.use_clock;

    int32_t compact = 0;
    for (int32_t expert = 0; expert < test.n_expert; ++expert) {
        result.expert_bounds[expert] = compact;
        for (int32_t token = 0; token < test.n_tokens; ++token) {
            for (int32_t used = 0; used < test.n_expert_used; ++used) {
                const int32_t route = token * test.n_expert_used + used;
                if (test.ids[route] == expert) {
                    result.ids_src[compact] = route;
                    result.ids_dst[compact] = route;
                    compact++;
                }
            }
        }
    }
    result.expert_bounds[test.n_expert] = compact;

    std::vector<bool>    requested(test.n_expert, false);
    std::vector<bool>    priority_requested(test.n_expert, false);
    std::vector<int32_t> unique;
    for (int32_t route = 0; route < n_ids; ++route) {
        const int32_t expert = test.ids[route];
        if (!requested[expert]) {
            requested[expert] = true;
            unique.push_back(expert);
            result.route_first[expert] = route;
        }
        if (!test.priority.empty() && test.priority[route / test.n_expert_used] != 0) {
            priority_requested[expert]    = true;
            result.route_priority[expert] = 1;
        }
    }

    result.n_active = unique.size();

    for (int pass = 0; pass < 2; ++pass) {
        for (int32_t expert : unique) {
            if ((pass == 0) != priority_requested[expert]) {
                continue;
            }
            if (result.expert_to_cache[expert] >= 0) {
                result.expert_order[result.n_resident + result.n_fill] = expert;
                result.n_resident++;
                if (pass == 0 && test.epoch != 0) {
                    result.protected_epoch[result.expert_to_cache[expert]] = test.epoch;
                }
                continue;
            }
            result.n_miss++;

            int32_t victim = -1;
            for (int32_t slot = 0; slot < test.n_cache; ++slot) {
                if (result.cache_to_expert[slot] < 0) {
                    victim = slot;
                    break;
                }
            }
            if (victim < 0 && pass == 0) {
                uint64_t oldest = UINT64_MAX;
                for (int32_t slot = 0; slot < test.n_cache; ++slot) {
                    const int32_t resident   = result.cache_to_expert[slot];
                    const bool protected_now = test.epoch != 0 && result.protected_epoch[slot] == (uint64_t) test.epoch;
                    const bool active_now    = requested[resident];
                    if (!protected_now && !active_now && result.last_used[slot] < oldest) {
                        oldest = result.last_used[slot];
                        victim = slot;
                    }
                }
            }
            if (victim < 0) {
                continue;
            }

            const int32_t evicted = result.cache_to_expert[victim];
            if (evicted >= 0) {
                result.expert_to_cache[evicted] = -1;
                result.n_evictions++;
            }
            result.cache_to_expert[victim]                         = expert;
            result.protected_epoch[victim]                         = pass == 0 ? test.epoch : 0;
            result.expert_order[result.n_resident + result.n_fill] = expert;
            result.fill_expert[result.n_fill]                      = expert;
            result.fill_slot[result.n_fill]                        = victim;
            result.n_fill++;
        }
    }

    for (uint32_t i = 0; i < result.n_fill; ++i) {
        const int32_t expert           = result.fill_expert[i];
        const int32_t slot             = result.fill_slot[i];
        result.expert_to_cache[expert] = slot;
        result.cache_to_expert[slot]   = expert;
    }

    if (!unique.empty()) {
        result.use_clock++;
    }
    for (int32_t expert : unique) {
        const int32_t slot = result.expert_to_cache[expert];
        if (slot >= 0) {
            result.last_used[slot] = result.use_clock;
        }
    }
    std::vector<bool> streamed_experts(test.n_expert, false);
    for (int32_t i = 0; i < n_ids; ++i) {
        const int32_t slot  = result.expert_to_cache[test.ids[i]];
        result.cache_ids[i] = slot >= 0 ? slot : -test.ids[i] - 1;
        result.n_streamed += slot < 0;
        if (slot < 0 && !streamed_experts[test.ids[i]]) {
            streamed_experts[test.ids[i]] = true;
            result.n_host_experts++;
        }
    }
    return result;
}

template <typename T>
static void expect_equal(const std::string &    name,
                         const char *           field,
                         const std::vector<T> & actual,
                         const std::vector<T> & expected,
                         size_t                 count) {
    for (size_t i = 0; i < count; ++i) {
        if (actual[i] != expected[i]) {
            throw std::runtime_error(name + ": " + field + "[" + std::to_string(i) + "] expected " +
                                     std::to_string(expected[i]) + ", got " + std::to_string(actual[i]));
        }
    }
}

static void run_case(const test_case & test, hipStream_t stream) {
    const int32_t n_ids = test.n_tokens * test.n_expert_used;
    if ((int32_t) test.ids.size() != n_ids || (int32_t) test.cache_to_expert.size() != test.n_cache ||
        (int32_t) test.last_used.size() != test.n_cache ||
        (!test.priority.empty() && (int32_t) test.priority.size() != test.n_tokens) ||
        (!test.protected_epoch.empty() && (int32_t) test.protected_epoch.size() != test.n_cache)) {
        throw std::runtime_error(test.name + ": invalid test data");
    }

    const plan_result          expected                = make_reference(test);
    const std::vector<int32_t> initial_expert_to_cache = make_expert_to_cache(test);

    device_buffer<int32_t>  ids(n_ids);
    device_buffer<int32_t>  ids_src(n_ids);
    device_buffer<int32_t>  ids_dst(n_ids);
    device_buffer<int32_t>  expert_bounds(test.n_expert + 1);
    device_buffer<int32_t>  saved_route_ids(n_ids);
    device_buffer<int32_t>  saved_route_bounds(test.n_expert + 1);
    device_buffer<int32_t>  saved_route_first(test.n_expert);
    device_buffer<int32_t>  saved_route_priority(test.n_expert);
    device_buffer<int32_t>  saved_route_tile_bounds(test.n_expert + 1);
    device_buffer<int32_t>  expert_order(test.n_expert);
    device_buffer<int32_t>  fill_expert(test.n_cache);
    device_buffer<int32_t>  fill_slot(test.n_cache);
    device_buffer<int32_t>  cache_ids(n_ids);
    device_buffer<int32_t>  expert_to_cache(test.n_expert);
    device_buffer<int32_t>  cache_to_expert(test.n_cache);
    device_buffer<uint64_t> last_used(test.n_cache);
    device_buffer<uint64_t> protected_epoch(test.n_cache);
    device_buffer<uint64_t> pending_priority_epoch(test.n_expert);
    device_buffer<uint64_t> use_clock(1);
    device_buffer<int32_t>  token_priority(test.n_tokens);
    device_buffer<int32_t>  epoch(1);
    device_buffer<uint32_t> n_active(1);
    device_buffer<uint32_t> n_resident(1);
    device_buffer<uint32_t> n_miss(1);
    device_buffer<uint32_t> n_fill(1);
    device_buffer<uint32_t> n_evictions(1);
    device_buffer<uint32_t> n_streamed(1);
    device_buffer<uint32_t> n_host_experts(1);

    ids.set(test.ids);
    expert_to_cache.set(initial_expert_to_cache);
    cache_to_expert.set(test.cache_to_expert);
    last_used.set(test.last_used);
    protected_epoch.set(test.protected_epoch.empty() ? std::vector<uint64_t>(test.n_cache, 0) : test.protected_epoch);
    pending_priority_epoch.set(std::vector<uint64_t>(test.n_expert, 0));
    use_clock.set({ test.use_clock });
    token_priority.set(test.priority.empty() ? std::vector<int32_t>(test.n_tokens, 0) : test.priority);
    epoch.set({ test.epoch });

    ggml_cuda_expert_plan plan  = {};
    plan.ids_src                = ids_src.ptr;
    plan.ids_dst                = ids_dst.ptr;
    plan.expert_bounds          = expert_bounds.ptr;
    plan.route_ids              = saved_route_ids.ptr;
    plan.route_bounds           = saved_route_bounds.ptr;
    plan.route_first            = saved_route_first.ptr;
    plan.route_priority         = saved_route_priority.ptr;
    plan.expert_order           = expert_order.ptr;
    plan.fill_expert            = fill_expert.ptr;
    plan.fill_slot              = fill_slot.ptr;
    plan.cache_ids              = cache_ids.ptr;
    plan.expert_to_cache        = expert_to_cache.ptr;
    plan.cache_to_expert        = cache_to_expert.ptr;
    plan.last_used              = last_used.ptr;
    plan.protected_epoch        = protected_epoch.ptr;
    plan.pending_priority_epoch = pending_priority_epoch.ptr;
    plan.use_clock              = use_clock.ptr;
    plan.token_priority         = token_priority.ptr;
    plan.epoch                  = epoch.ptr;
    plan.n_active               = n_active.ptr;
    plan.n_resident             = n_resident.ptr;
    plan.n_miss                 = n_miss.ptr;
    plan.n_fill                 = n_fill.ptr;
    plan.n_evictions            = n_evictions.ptr;
    plan.n_streamed             = n_streamed.ptr;
    plan.n_host_experts         = n_host_experts.ptr;

    ggml_cuda_launch_expert_plan(ids.ptr, plan, test.n_expert, test.n_tokens, test.n_expert_used, test.n_expert_used,
                                 test.n_expert_used, test.n_expert_used, false, test.n_cache, stream);
    hip_check(hipGetLastError(), "ggml_cuda_launch_expert_plan");
    constexpr int route_tile_width = 4;
    ggml_cuda_launch_expert_route_tiles(saved_route_bounds.ptr, saved_route_tile_bounds.ptr, test.n_expert,
                                        route_tile_width, stream);
    hip_check(hipGetLastError(), "ggml_cuda_launch_expert_route_tiles");
    hip_check(hipStreamSynchronize(stream), "hipStreamSynchronize");

    expect_equal(test.name, "ids_src", ids_src.get(), expected.ids_src, n_ids);
    expect_equal(test.name, "ids_dst", ids_dst.get(), expected.ids_dst, n_ids);
    expect_equal(test.name, "expert_bounds", expert_bounds.get(), expected.expert_bounds, test.n_expert + 1);
    expect_equal(test.name, "saved_route_ids", saved_route_ids.get(), expected.ids_dst, n_ids);
    expect_equal(test.name, "saved_route_bounds", saved_route_bounds.get(), expected.expert_bounds, test.n_expert + 1);
    expect_equal(test.name, "saved_route_first", saved_route_first.get(), expected.route_first, test.n_expert);
    expect_equal(test.name, "saved_route_priority", saved_route_priority.get(), expected.route_priority, test.n_expert);
    std::vector<int32_t> expected_route_tile_bounds(test.n_expert + 1, 0);
    for (int32_t expert = 0; expert < test.n_expert; ++expert) {
        const int32_t routes = expected.expert_bounds[expert + 1] - expected.expert_bounds[expert];
        expected_route_tile_bounds[expert + 1] =
            expected_route_tile_bounds[expert] + (routes + route_tile_width - 1) / route_tile_width;
    }
    expect_equal(test.name, "saved_route_tile_bounds", saved_route_tile_bounds.get(), expected_route_tile_bounds,
                 test.n_expert + 1);
    const int32_t route_tile_upper_bound =
        n_ids == 0 ? 0 : (n_ids + route_tile_width - 1) / route_tile_width + std::min(test.n_expert, n_ids) - 1;
    if (expected_route_tile_bounds.back() > route_tile_upper_bound) {
        throw std::runtime_error(test.name + ": route tile count exceeds the MMQ launch bound");
    }
    expect_equal(test.name, "expert_order", expert_order.get(), expected.expert_order,
                 expected.n_resident + expected.n_fill);
    expect_equal(test.name, "fill_expert", fill_expert.get(), expected.fill_expert, expected.n_fill);
    expect_equal(test.name, "fill_slot", fill_slot.get(), expected.fill_slot, expected.n_fill);
    expect_equal(test.name, "cache_ids", cache_ids.get(), expected.cache_ids, n_ids);
    expect_equal(test.name, "expert_to_cache", expert_to_cache.get(), expected.expert_to_cache, test.n_expert);
    expect_equal(test.name, "cache_to_expert", cache_to_expert.get(), expected.cache_to_expert, test.n_cache);
    expect_equal(test.name, "last_used", last_used.get(), expected.last_used, test.n_cache);
    expect_equal(test.name, "protected_epoch", protected_epoch.get(), expected.protected_epoch, test.n_cache);
    expect_equal(test.name, "use_clock", use_clock.get(), std::vector<uint64_t>{ expected.use_clock }, 1);
    expect_equal(test.name, "n_active", n_active.get(), std::vector<uint32_t>{ expected.n_active }, 1);
    expect_equal(test.name, "n_resident", n_resident.get(), std::vector<uint32_t>{ expected.n_resident }, 1);
    expect_equal(test.name, "n_miss", n_miss.get(), std::vector<uint32_t>{ expected.n_miss }, 1);
    expect_equal(test.name, "n_fill", n_fill.get(), std::vector<uint32_t>{ expected.n_fill }, 1);
    expect_equal(test.name, "n_evictions", n_evictions.get(), std::vector<uint32_t>{ expected.n_evictions }, 1);
    expect_equal(test.name, "n_streamed", n_streamed.get(), std::vector<uint32_t>{ expected.n_streamed }, 1);
    expect_equal(test.name, "n_host_experts", n_host_experts.get(), std::vector<uint32_t>{ expected.n_host_experts },
                 1);

    std::printf("%s: OK\n", test.name.c_str());
}

static void run_deferred_priority_case(hipStream_t stream) {
    constexpr int n_expert     = 3;
    constexpr int n_cache      = 2;
    constexpr int n_max_tokens = 3;

    device_buffer<int32_t>  ids(n_max_tokens);
    device_buffer<int32_t>  priority(n_max_tokens);
    device_buffer<int32_t>  route_ids(n_max_tokens);
    device_buffer<int32_t>  route_bounds(n_expert + 1);
    device_buffer<int32_t>  route_first(n_expert);
    device_buffer<int32_t>  route_priority(n_expert);
    device_buffer<int32_t>  fill_expert(n_cache);
    device_buffer<int32_t>  fill_slot(n_cache);
    device_buffer<int32_t>  cache_ids(n_max_tokens);
    device_buffer<int32_t>  expert_to_cache(n_expert);
    device_buffer<int32_t>  cache_to_expert(n_cache);
    device_buffer<uint64_t> last_used(n_cache);
    device_buffer<uint64_t> protected_epoch(n_cache);
    device_buffer<uint64_t> pending_priority_epoch(n_expert);
    device_buffer<uint64_t> use_clock(1);
    device_buffer<int32_t>  epoch(1);
    device_buffer<uint32_t> n_active(1);
    device_buffer<uint32_t> n_resident(1);
    device_buffer<uint32_t> n_miss(1);
    device_buffer<uint32_t> n_fill(1);
    device_buffer<uint32_t> n_evictions(1);
    device_buffer<uint32_t> n_streamed(1);
    device_buffer<uint32_t> n_host_experts(1);

    ggml_cuda_expert_plan plan  = {};
    plan.fill_expert            = fill_expert.ptr;
    plan.fill_slot              = fill_slot.ptr;
    plan.cache_ids              = cache_ids.ptr;
    plan.route_ids              = route_ids.ptr;
    plan.route_bounds           = route_bounds.ptr;
    plan.route_first            = route_first.ptr;
    plan.route_priority         = route_priority.ptr;
    plan.expert_to_cache        = expert_to_cache.ptr;
    plan.cache_to_expert        = cache_to_expert.ptr;
    plan.last_used              = last_used.ptr;
    plan.protected_epoch        = protected_epoch.ptr;
    plan.pending_priority_epoch = pending_priority_epoch.ptr;
    plan.use_clock              = use_clock.ptr;
    plan.token_priority         = priority.ptr;
    plan.epoch                  = epoch.ptr;
    plan.n_active               = n_active.ptr;
    plan.n_resident             = n_resident.ptr;
    plan.n_miss                 = n_miss.ptr;
    plan.n_fill                 = n_fill.ptr;
    plan.n_evictions            = n_evictions.ptr;
    plan.n_streamed             = n_streamed.ptr;
    plan.n_host_experts         = n_host_experts.ptr;

    auto reset = [&] {
        expert_to_cache.set({ 0, 1, -1 });
        cache_to_expert.set({ 0, 1 });
        last_used.set({ 0, 0 });
        protected_epoch.set({ 0, 0 });
        pending_priority_epoch.set({ 0, 0, 0 });
        use_clock.set({ 0 });
    };
    auto invoke = [&](const std::vector<int32_t> & route_ids, const std::vector<int32_t> & route_priority,
                      int32_t route_epoch) {
        std::vector<int32_t> ids_data(n_max_tokens, 0);
        std::vector<int32_t> priority_data(n_max_tokens, 0);
        std::copy(route_ids.begin(), route_ids.end(), ids_data.begin());
        std::copy(route_priority.begin(), route_priority.end(), priority_data.begin());
        ids.set(ids_data);
        priority.set(priority_data);
        epoch.set({ route_epoch });
        ggml_cuda_launch_expert_cache_plan(ids.ptr, plan, n_expert, route_ids.size(), 1, n_cache, stream);
        hip_check(hipGetLastError(), "deferred priority launch");
        hip_check(hipStreamSynchronize(stream), "deferred priority synchronize");
    };

    reset();
    invoke({ 0, 1, 2 }, { 0, 0, 1 }, 11);
    expect_equal("deferred-first", "cache_ids", cache_ids.get(), { 0, 1, -3 }, 3);
    expect_equal("deferred-first", "cache_to_expert", cache_to_expert.get(), { 0, 1 }, 2);
    expect_equal("deferred-first", "pending_priority_epoch", pending_priority_epoch.get(), { 0, 0, 11 }, 3);
    expect_equal("deferred-first", "protected_epoch", protected_epoch.get(), { 0, 0 }, 2);

    invoke({ 1 }, { 0 }, 11);
    expect_equal("deferred-second", "cache_ids", cache_ids.get(), { 1 }, 1);
    expect_equal("deferred-second", "cache_to_expert", cache_to_expert.get(), { 2, 1 }, 2);
    expect_equal("deferred-second", "pending_priority_epoch", pending_priority_epoch.get(), { 0, 0, 0 }, 3);
    expect_equal("deferred-second", "protected_epoch", protected_epoch.get(), { 11, 0 }, 2);
    expect_equal("deferred-second", "last_used", last_used.get(), { 2, 2 }, 2);
    expect_equal("deferred-second", "use_clock", use_clock.get(), { 2 }, 1);
    expect_equal("deferred-second", "n_miss", n_miss.get(), { 0 }, 1);
    expect_equal("deferred-second", "n_fill", n_fill.get(), { 1 }, 1);

    reset();
    invoke({ 0, 1, 2 }, { 0, 0, 1 }, 11);
    invoke({ 1 }, { 0 }, 12);
    expect_equal("deferred-epoch-change", "cache_to_expert", cache_to_expert.get(), { 0, 1 }, 2);
    expect_equal("deferred-epoch-change", "pending_priority_epoch", pending_priority_epoch.get(), { 0, 0, 0 }, 3);

    std::printf("deferred-priority: OK\n");
}

static void run_backend_source_case(ggml_backend_t backend, ggml_type type, int32_t n_tokens) {
    const int64_t n_expert      = 8;
    const int64_t n_cache       = 3;
    const int64_t n_expert_used = 4;
    const int64_t n_rows        = 3;
    const int64_t n_cols        = 256;
    const size_t  row_size      = ggml_row_size(type, n_cols);
    const size_t  expert_size   = row_size * n_rows;
    const size_t  host_size     = expert_size * n_expert;

    void * host_data = nullptr;
    hip_check(hipHostMalloc(&host_data, host_size, hipHostMallocMapped), "hipHostMalloc");

    try {
        std::vector<float> rows(n_rows * n_cols);
        for (int64_t expert = 0; expert < n_expert; ++expert) {
            for (int64_t row = 0; row < n_rows; ++row) {
                for (int64_t col = 0; col < n_cols; ++col) {
                    rows[row * n_cols + col] = 0.25f * (expert + 1) + 0.003f * row + 0.0001f * col;
                }
            }
            const size_t written =
                ggml_quantize_chunk(type, rows.data(), static_cast<uint8_t *>(host_data) + expert * expert_size, 0,
                                    n_rows, n_cols, nullptr);
            if (written != expert_size) {
                throw std::runtime_error("source quantization size mismatch");
            }
        }

        void * host_device_data = nullptr;
        hip_check(hipHostGetDevicePointer(&host_device_data, host_data, 0), "hipHostGetDevicePointer");

        ggml_backend_cuda_expert_cache_weight source_weight = {
            host_data,
            host_device_data,
            0,
            expert_size,
        };
        ggml_backend_cuda_expert_source source_desc = {
            GGML_CUDA_EXPERT_SOURCE_MAGIC,
            static_cast<uint32_t>(n_expert),
            nullptr,
            &source_weight,
        };

        ggml_init_params params = {
            ggml_tensor_overhead() * 32 + 2 * ggml_graph_overhead_custom(32, false),
            nullptr,
            true,
        };
        ggml_context_ptr ctx(ggml_init(params));
        if (!ctx) {
            throw std::runtime_error("failed to create backend source test context");
        }

        ggml_tensor * slots      = ggml_new_tensor_3d(ctx.get(), type, n_cols, n_rows, n_cache);
        ggml_tensor * full       = ggml_new_tensor_3d(ctx.get(), type, n_cols, n_rows, n_expert);
        ggml_tensor * input      = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, n_cols, n_expert_used, n_tokens);
        ggml_tensor * ids        = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, n_expert_used, n_tokens);
        ggml_tensor * source_ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, n_expert_used, n_tokens);

        const std::array<int32_t, 3> slot_experts = { 5, 2, 7 };
        std::vector<uint8_t>         slots_data(n_cache * expert_size);
        for (int64_t slot = 0; slot < n_cache; ++slot) {
            std::memcpy(slots_data.data() + slot * expert_size,
                        static_cast<const uint8_t *>(host_data) + slot_experts[slot] * expert_size, expert_size);
        }

        std::vector<float>            input_data(n_cols * n_expert_used * n_tokens);
        std::vector<int32_t>          ids_data(n_expert_used * n_tokens);
        std::vector<int32_t>          source_ids_data(n_expert_used * n_tokens);
        const std::array<int32_t, 16> route_pattern = {
            5, 5, 1, 7, 2, 3, 2, 6, 0, 7, 5, 4, 6, 2, 1, 5,
        };
        for (int64_t token = 0; token < n_tokens; ++token) {
            for (int64_t used = 0; used < n_expert_used; ++used) {
                const int32_t expert   = route_pattern[(token * n_expert_used + used) % route_pattern.size()];
                const int64_t route    = token * n_expert_used + used;
                ids_data[route]        = expert;
                source_ids_data[route] = -expert - 1;
                for (int64_t col = 0; col < n_cols; ++col) {
                    input_data[route * n_cols + col] = 0.01f * (1 + col % 17) + 0.002f * used;
                }
                for (int64_t slot = 0; slot < n_cache; ++slot) {
                    if (slot_experts[slot] == expert) {
                        source_ids_data[route] = slot;
                        break;
                    }
                }
            }
        }

        slots->extra             = &source_desc;
        ggml_tensor * source_out = ggml_mul_mat_id(ctx.get(), slots, input, ids);
        source_out->src[3]       = source_ids;
        ggml_tensor * full_out   = ggml_mul_mat_id(ctx.get(), full, input, ids);

        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
        if (!buffer) {
            throw std::runtime_error("failed to allocate backend source test tensors");
        }
        ggml_backend_tensor_set(full, host_data, 0, host_size);
        ggml_backend_tensor_set(slots, slots_data.data(), 0, slots_data.size());
        ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
        ggml_backend_tensor_set(ids, ids_data.data(), 0, ids_data.size() * sizeof(int32_t));
        ggml_backend_tensor_set(source_ids, source_ids_data.data(), 0, source_ids_data.size() * sizeof(int32_t));

        std::vector<uint8_t> slots_check(slots_data.size());
        ggml_backend_tensor_get(slots, slots_check.data(), 0, slots_check.size());
        if (slots_check != slots_data) {
            throw std::runtime_error("slot tensor data mismatch");
        }
        std::vector<int32_t> source_ids_check(source_ids_data.size());
        ggml_backend_tensor_get(source_ids, source_ids_check.data(), 0, source_ids_check.size() * sizeof(int32_t));
        if (source_ids_check != source_ids_data) {
            throw std::runtime_error("source selector data mismatch");
        }

        ggml_cgraph * source_graph = ggml_new_graph_custom(ctx.get(), 32, false);
        ggml_build_forward_expand(source_graph, source_out);
        if (ggml_backend_graph_compute(backend, source_graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("backend source graph execution failed");
        }

        std::vector<float> source_result(ggml_nelements(source_out));
        std::vector<float> full_result(ggml_nelements(full_out));
        ggml_backend_tensor_get(source_out, source_result.data(), 0, source_result.size() * sizeof(float));

        ggml_cgraph * full_graph = ggml_new_graph_custom(ctx.get(), 32, false);
        ggml_build_forward_expand(full_graph, full_out);
        if (ggml_backend_graph_compute(backend, full_graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("backend full graph execution failed");
        }
        ggml_backend_tensor_get(full_out, full_result.data(), 0, full_result.size() * sizeof(float));

        float max_error = 0.0f;
        float max_value = 0.0f;
        for (size_t i = 0; i < source_result.size(); ++i) {
            max_error = std::max(max_error, std::fabs(source_result[i] - full_result[i]));
            max_value = std::max(max_value, std::fabs(full_result[i]));
        }
        if (max_error > 2e-3f * std::max(1.0f, max_value)) {
            throw std::runtime_error("source backend mismatch for " + std::string(ggml_type_name(type)) +
                                     " tokens=" + std::to_string(n_tokens) + " max_error=" + std::to_string(max_error));
        }

        std::printf("source-%s-%d: OK\n", ggml_type_name(type), n_tokens);
    } catch (...) {
        hip_check(hipHostFree(host_data), "hipHostFree");
        throw;
    }

    hip_check(hipHostFree(host_data), "hipHostFree");
}

static void run_backend_cached_source_case(ggml_backend_t backend, ggml_type type) {
    const int64_t n_expert      = 8;
    const int64_t n_cache       = 3;
    const int64_t n_expert_used = 4;
    const int64_t n_tokens      = 128;
    const int64_t n_rows        = 8;
    const int64_t n_cols        = 256;
    const size_t  row_size      = ggml_row_size(type, n_cols);
    const size_t  expert_size   = row_size * n_rows;
    const size_t  host_size     = expert_size * n_expert;

    void * host_data = nullptr;
    hip_check(hipHostMalloc(&host_data, host_size, hipHostMallocMapped), "hipHostMalloc");

    try {
        std::vector<float> rows(n_rows * n_cols);
        for (int64_t expert = 0; expert < n_expert; ++expert) {
            for (int64_t row = 0; row < n_rows; ++row) {
                for (int64_t col = 0; col < n_cols; ++col) {
                    rows[row * n_cols + col] = 0.25f * (expert + 1) + 0.003f * row + 0.0001f * col;
                }
            }
            const size_t written =
                ggml_quantize_chunk(type, rows.data(), static_cast<uint8_t *>(host_data) + expert * expert_size, 0,
                                    n_rows, n_cols, nullptr);
            if (written != expert_size) {
                throw std::runtime_error("cached source quantization size mismatch");
            }
        }

        void * host_device_data = nullptr;
        hip_check(hipHostGetDevicePointer(&host_device_data, host_data, 0), "hipHostGetDevicePointer");

        ggml_backend_cuda_expert_cache_weight source_weight = {
            host_data,
            host_device_data,
            0,
            expert_size,
        };
        ggml_backend_cuda_expert_cache_desc cache_desc = {};
        cache_desc.magic                               = GGML_CUDA_EXPERT_CACHE_MAGIC;
        cache_desc.version                             = GGML_CUDA_EXPERT_CACHE_VERSION;
        cache_desc.n_expert                            = n_expert;
        cache_desc.n_cache                             = n_cache;
        cache_desc.n_weights                           = 1;
        cache_desc.weights[0]                          = source_weight;

        const auto state_layout                  = ggml_cuda_expert_cache_layout(n_expert, n_cache);
        cache_desc.expert_to_cache_offset        = state_layout.expert_to_cache_offset;
        cache_desc.cache_to_expert_offset        = state_layout.cache_to_expert_offset;
        cache_desc.last_used_offset              = state_layout.last_used_offset;
        cache_desc.protected_epoch_offset        = state_layout.protected_epoch_offset;
        cache_desc.pending_priority_epoch_offset = state_layout.pending_priority_epoch_offset;
        cache_desc.fill_expert_offset            = state_layout.fill_expert_offset;
        cache_desc.fill_slot_offset              = state_layout.fill_slot_offset;
        cache_desc.state_size                    = state_layout.state_size;

        ggml_init_params params = {
            ggml_tensor_overhead() * 64 + 2 * ggml_graph_overhead_custom(64, false),
            nullptr,
            true,
        };
        ggml_context_ptr ctx(ggml_init(params));
        if (!ctx) {
            throw std::runtime_error("failed to create cached source test context");
        }

        ggml_tensor * slots    = ggml_new_tensor_3d(ctx.get(), type, n_cols, n_rows, n_cache);
        ggml_tensor * full     = ggml_new_tensor_3d(ctx.get(), type, n_cols, n_rows, n_expert);
        ggml_tensor * input    = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, n_cols, n_expert_used, n_tokens);
        ggml_tensor * ids      = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, n_expert_used, n_tokens);
        ggml_tensor * priority = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, n_tokens);
        ggml_tensor * epoch    = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
        ggml_tensor * state    = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, cache_desc.state_size);

        const int64_t n_routes           = n_tokens * n_expert_used;
        const auto    route_layout       = ggml_cuda_expert_cache_route_layout(n_routes, n_expert);
        const int64_t route_storage_size = route_layout.size / sizeof(int32_t);
        ggml_tensor * cache_args[]       = { ids, priority, epoch, state, slots };
        ggml_tensor * cache_plan = ggml_custom_4d(ctx.get(), GGML_TYPE_I32, route_storage_size, 1, 1, 1, cache_args, 5,
                                                  nullptr, 1, &cache_desc);
        ggml_tensor * cache_ids  = ggml_view_4d(ctx.get(), cache_plan, ids->ne[0], ids->ne[1], ids->ne[2], ids->ne[3],
                                                ids->nb[1], ids->nb[2], ids->nb[3], 0);
        ggml_backend_moe_cache cache                = {};
        cache.desc                                  = cache_desc;
        ggml_backend_cuda_expert_source source_desc = {
            GGML_CUDA_EXPERT_SOURCE_MAGIC,
            static_cast<uint32_t>(n_expert),
            &cache,
            &source_weight,
        };
        slots->extra                                       = &source_desc;
        ggml_backend_cuda_expert_plan_binding plan_binding = {
            GGML_CUDA_EXPERT_PLAN_MAGIC,
            &cache,
        };
        cache_ids->extra = &plan_binding;

        ggml_tensor * source_out = ggml_mul_mat_id(ctx.get(), slots, input, ids);
        source_out->src[3]       = cache_ids;
        ggml_tensor * full_out   = ggml_mul_mat_id(ctx.get(), full, input, ids);

        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
        if (!buffer) {
            throw std::runtime_error("failed to allocate cached source test tensors");
        }

        std::vector<float>            input_data(n_cols * n_expert_used * n_tokens);
        std::vector<int32_t>          ids_data(n_routes);
        const std::array<int32_t, 16> route_pattern = {
            5, 5, 1, 7, 2, 3, 2, 6, 0, 7, 5, 4, 6, 2, 1, 5,
        };
        for (int64_t token = 0; token < n_tokens; ++token) {
            for (int64_t used = 0; used < n_expert_used; ++used) {
                const int64_t route = token * n_expert_used + used;
                ids_data[route]     = route_pattern[route % route_pattern.size()];
                for (int64_t col = 0; col < n_cols; ++col) {
                    input_data[route * n_cols + col] = 0.01f * (1 + col % 17) + 0.002f * used;
                }
            }
        }

        std::vector<uint8_t> state_data(cache_desc.state_size, 0);
        std::fill(reinterpret_cast<int32_t *>(state_data.data() + cache_desc.expert_to_cache_offset),
                  reinterpret_cast<int32_t *>(state_data.data() + cache_desc.expert_to_cache_offset) + n_expert, -1);
        std::fill(reinterpret_cast<int32_t *>(state_data.data() + cache_desc.cache_to_expert_offset),
                  reinterpret_cast<int32_t *>(state_data.data() + cache_desc.cache_to_expert_offset) + n_cache, -1);

        ggml_backend_tensor_set(full, host_data, 0, host_size);
        ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
        ggml_backend_tensor_set(ids, ids_data.data(), 0, ids_data.size() * sizeof(int32_t));
        std::vector<int32_t> priority_data(n_tokens, 0);
        const int32_t        epoch_data = 1;
        ggml_backend_tensor_set(priority, priority_data.data(), 0, priority_data.size() * sizeof(int32_t));
        ggml_backend_tensor_set(epoch, &epoch_data, 0, sizeof(epoch_data));
        ggml_backend_tensor_set(state, state_data.data(), 0, state_data.size());

        ggml_cgraph * source_graph = ggml_new_graph_custom(ctx.get(), 64, false);
        ggml_build_forward_expand(source_graph, source_out);
        ggml_cgraph * full_graph = ggml_new_graph_custom(ctx.get(), 64, false);
        ggml_build_forward_expand(full_graph, full_out);

        auto compare = [&](const char * label) {
            if (ggml_backend_graph_compute(backend, source_graph) != GGML_STATUS_SUCCESS ||
                ggml_backend_graph_compute(backend, full_graph) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error(std::string(label) + ": cached source graph execution failed");
            }

            std::vector<float> source_result(ggml_nelements(source_out));
            std::vector<float> full_result(ggml_nelements(full_out));
            ggml_backend_tensor_get(source_out, source_result.data(), 0, source_result.size() * sizeof(float));
            ggml_backend_tensor_get(full_out, full_result.data(), 0, full_result.size() * sizeof(float));

            float max_error = 0.0f;
            float max_value = 0.0f;
            for (size_t i = 0; i < source_result.size(); ++i) {
                max_error = std::max(max_error, std::fabs(source_result[i] - full_result[i]));
                max_value = std::max(max_value, std::fabs(full_result[i]));
            }
            if (max_error > 2e-3f * std::max(1.0f, max_value)) {
                throw std::runtime_error(std::string(label) + ": cached source mismatch for " + ggml_type_name(type) +
                                         " max_error=" + std::to_string(max_error));
            }
        };

        compare("cached-first");

        std::reverse(ids_data.begin(), ids_data.end());
        ggml_backend_tensor_set(ids, ids_data.data(), 0, ids_data.size() * sizeof(int32_t));
        compare("cached-reuse");

        std::printf("cached-source-%s: OK\n", ggml_type_name(type));
    } catch (...) {
        hip_check(hipHostFree(host_data), "hipHostFree");
        throw;
    }

    hip_check(hipHostFree(host_data), "hipHostFree");
}

static void run_backend_interface_lifecycle_case(ggml_backend_t backend) {
    const ggml_backend_dev_t device        = ggml_backend_get_device(backend);
    const ggml_backend_reg_t registry      = ggml_backend_dev_backend_reg(device);
    const auto               get_interface = reinterpret_cast<ggml_backend_moe_cache_get_interface_t>(
        ggml_backend_reg_get_proc_address(registry, "ggml_backend_moe_cache_get_interface"));
    if (get_interface == nullptr) {
        throw std::runtime_error("MoE cache interface is unavailable");
    }

    const ggml_backend_moe_cache_i * api = get_interface();
    if (api == nullptr || api->version != GGML_BACKEND_MOE_CACHE_INTERFACE_VERSION || !api->supports(device)) {
        throw std::runtime_error("MoE cache interface is invalid");
    }

    if (api->get_state_size(8, 0, 1) != 0 || api->get_state_size(8, 8, 1) != 0 || api->get_state_size(8, 3, 0) != 0) {
        throw std::runtime_error("MoE cache interface accepted malformed dimensions");
    }

    ggml_context_ptr source_ctx(ggml_init({ ggml_tensor_overhead(), nullptr, true }));
    ggml_context_ptr device_ctx(ggml_init({ 6 * ggml_tensor_overhead(), nullptr, true }));
    if (!source_ctx || !device_ctx) {
        throw std::runtime_error("MoE cache interface context allocation failed");
    }

    const int64_t           source_ne[3] = { 256, 8, 8 };
    ggml_tensor *           source       = ggml_new_tensor(source_ctx.get(), GGML_TYPE_Q4_K, 3, source_ne);
    ggml_backend_buffer_ptr source_buffer(
        ggml_backend_buft_alloc_buffer(ggml_backend_dev_host_buffer_type(device), ggml_nbytes(source)));
    if (!source_buffer ||
        ggml_backend_tensor_alloc(source_buffer.get(), source, ggml_backend_buffer_get_base(source_buffer.get())) !=
            GGML_STATUS_SUCCESS) {
        throw std::runtime_error("MoE cache interface source allocation failed");
    }

    const int64_t           slots_ne[3] = { 256, 8, 3 };
    ggml_tensor *           slots       = ggml_new_tensor(device_ctx.get(), GGML_TYPE_Q4_K, 3, slots_ne);
    ggml_tensor *           state    = ggml_new_tensor_1d(device_ctx.get(), GGML_TYPE_I8, api->get_state_size(8, 3, 1));
    ggml_tensor *           ids      = ggml_new_tensor_2d(device_ctx.get(), GGML_TYPE_I32, 2, 4);
    ggml_tensor *           priority = ggml_new_tensor_1d(device_ctx.get(), GGML_TYPE_I32, 4);
    ggml_tensor *           epoch    = ggml_new_tensor_1d(device_ctx.get(), GGML_TYPE_I32, 1);
    ggml_backend_buffer_ptr device_buffer(ggml_backend_alloc_ctx_tensors(device_ctx.get(), backend));
    if (!device_buffer) {
        throw std::runtime_error("MoE cache interface device allocation failed");
    }

    ggml_tensor * sources[]     = { source };
    ggml_tensor * slots_array[] = { slots };
    if (api->create(backend, nullptr, sources, slots_array, 8, 3, 1) != nullptr ||
        api->create(backend, state, sources, slots_array, 8, 8, 1) != nullptr) {
        throw std::runtime_error("MoE cache interface accepted malformed creation");
    }

    ggml_backend_moe_cache_t cache = api->create(backend, state, sources, slots_array, 8, 3, 1);
    if (cache == nullptr || slots->extra == nullptr) {
        throw std::runtime_error("MoE cache interface creation failed");
    }

    ggml_context_ptr                  graph_ctx(ggml_init({ 8 * ggml_tensor_overhead(), nullptr, true }));
    const ggml_backend_moe_cache_plan plan = api->build_plan(cache, graph_ctx.get(), ids, priority, epoch);
    if (plan.selectors == nullptr || plan.execution == nullptr ||
        ggml_nelements(plan.selectors) != ggml_nelements(ids) ||
        ggml_nelements(plan.execution) != 2 * ggml_nelements(ids) + 34) {
        api->destroy(cache);
        throw std::runtime_error("MoE cache interface plan shape mismatch");
    }

    api->destroy(cache);
    if (slots->extra != nullptr) {
        throw std::runtime_error("MoE cache interface left a slot binding after destroy");
    }

    std::printf("backend-interface-lifecycle: OK\n");
}

static std::vector<int32_t> identity_experts(int32_t n_expert) {
    std::vector<int32_t> result(n_expert);
    for (int32_t i = 0; i < n_expert; ++i) {
        result[i] = i;
    }
    return result;
}

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) {
        std::fprintf(stderr, "failed to initialize ROCm0\n");
        return 1;
    }

    hipStream_t stream = nullptr;
    try {
        run_backend_interface_lifecycle_case(backend);
        hip_check(hipStreamCreate(&stream), "hipStreamCreate");

        run_case(
            {
                "cold",
                16,
                8,
                4,
                2,
                { 3, 7, 1, 5, 3, 1, 7, 5 },
                {},
                std::vector<int32_t>(8, -1),
                std::vector<uint64_t>(8, 0),
                0,
                {},
                1,
        },
            stream);

        run_case(
            {
                "partial",
                16,
                8,
                8,
                2,
                { 8, 2, 0, 3, 8, 3, 2, 0, 9, 8, 0, 9, 3, 2, 8, 0 },
                {},
                { 0, 8, 10, 12, 14, 15, 6, 7 },
                { 20, 10, 1, 2, 3, 4, 5, 6 },
                20,
                {},
                1,
        },
            stream);

        run_case(
            {
                "full",
                16,
                16,
                8,
                2,
                identity_experts(16),
                {},
                identity_experts(16),
                std::vector<uint64_t>(16, 4),
                4,
                {},
                1,
            },
            stream);

        std::vector<int32_t> duplicate_ids;
        for (int32_t token = 0; token < 128; ++token) {
            duplicate_ids.insert(duplicate_ids.end(), { 3, 7, 11, 15 });
        }
        run_case(
            {
                "duplicate-heavy",
                32,
                8,
                128,
                4,
                std::move(duplicate_ids),
                {},
                std::vector<int32_t>(8, -1),
                std::vector<uint64_t>(8, 0),
                0,
                {},
                1,
            },
            stream);

        run_case(
            {
                "all-256",
                256,
                256,
                32,
                8,
                identity_experts(256),
                {},
                std::vector<int32_t>(256, -1),
                std::vector<uint64_t>(256, 0),
                0,
                {},
                1,
            },
            stream);

        run_case(
            {
                "overflow",
                8,
                2,
                2,
                2,
                { 0, 1, 2, 3 },
                {},
                { -1, -1 },
                { 0, 0 },
                0,
                {},
                1,
        },
            stream);

        run_case(
            {
                "first-use-admission",
                8,
                2,
                2,
                2,
                { 6, 4, 1, 0 },
                {},
                { -1, -1 },
                { 0, 0 },
                0,
                {},
                1,
        },
            stream);

        run_case(
            {
                "priority-first",
                4,
                2,
                2,
                1,
                { 2, 3 },
                { 0, 1 },
                { 2, 1 },
                { 0, 1 },
                1,
                {},
                1,
        },
            stream);

        run_case(
            {
                "current-epoch-protection",
                4,
                2,
                1,
                1,
                { 2 },
                { 1 },
                { 0, 1 },
                { 0, 1 },
                1,
                { 7, 0 },
                7,
        },
            stream);

        run_case(
            {
                "expired-epoch-protection",
                4,
                2,
                1,
                1,
                { 2 },
                { 1 },
                { 0, 1 },
                { 0, 1 },
                1,
                { 7, 0 },
                8,
        },
            stream);

        run_case(
            {
                "active-route-retention",
                4,
                2,
                2,
                1,
                { 0, 2 },
                { 1, 1 },
                { 0, 1 },
                { 1, 9 },
                9,
                {},
                13,
        },
            stream);

        run_case(
            {
                "equal-timestamp-victim",
                4,
                2,
                1,
                1,
                { 2 },
                { 1 },
                { 0, 1 },
                { 5, 5 },
                5,
                {},
                13,
        },
            stream);

        run_case(
            {
                "prompt-residency-stability",
                4,
                2,
                1,
                1,
                { 2 },
                { 0 },
                { 0, 1 },
                { 3, 7 },
                7,
                {},
                13,
        },
            stream);

        run_case(
            {
                "route-tile-upper-bound",
                8,
                4,
                5,
                2,
                { 0, 0, 0, 7, 0, 0, 0, 7, 0, 7 },
                {},
                { -1, -1, -1, -1 },
                { 0, 0, 0, 0 },
                0,
                {},
                1,
        },
            stream);

        run_deferred_priority_case(stream);

        for (ggml_type type : { GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K }) {
            run_backend_source_case(backend, type, 4);
            run_backend_source_case(backend, type, 8);
            run_backend_source_case(backend, type, 16);
            run_backend_cached_source_case(backend, type);
        }
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        if (stream != nullptr) {
            (void) hipStreamDestroy(stream);
        }
        ggml_backend_free(backend);
        return 1;
    }

    hip_check(hipStreamDestroy(stream), "hipStreamDestroy");
    ggml_backend_free(backend);
    std::printf("test-moe-expert-plan: all tests OK\n");
    return 0;
}
