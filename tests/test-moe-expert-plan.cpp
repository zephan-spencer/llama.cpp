#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <hip/hip_runtime_api.h>
using cudaStream_t = hipStream_t;
#include "mmid.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

static void hip_check(hipError_t error, const char * call) {
    if (error != hipSuccess) {
        throw std::runtime_error(std::string(call) + ": " + hipGetErrorString(error));
    }
}

template<typename T>
class device_buffer {
public:
    explicit device_buffer(size_t size) : size(size) {
        hip_check(hipMalloc(&ptr, size * sizeof(T)), "hipMalloc");
    }

    ~device_buffer() {
        if (ptr != nullptr) {
            (void) hipFree(ptr);
        }
    }

    device_buffer(const device_buffer &) = delete;
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

    T * ptr = nullptr;
    size_t size;
};

struct test_case {
    std::string name;
    int32_t n_expert;
    int32_t n_cache;
    int32_t n_tokens;
    int32_t n_expert_used;
    std::vector<int32_t> ids;
    std::vector<int32_t> cache_to_expert;
    std::vector<uint64_t> last_used;
    uint64_t use_clock = 0;
};

struct plan_result {
    std::vector<int32_t> ids_src;
    std::vector<int32_t> ids_dst;
    std::vector<int32_t> expert_bounds;
    std::vector<int32_t> expert_order;
    std::vector<int32_t> miss_expert;
    std::vector<int32_t> miss_slot;
    std::vector<int32_t> cache_ids;
    std::vector<int32_t> expert_to_cache;
    std::vector<int32_t> cache_to_expert;
    std::vector<uint64_t> last_used;
    uint64_t use_clock = 0;
    uint32_t n_active = 0;
    uint32_t n_resident = 0;
    uint32_t n_miss = 0;
    uint32_t n_evictions = 0;
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
    result.expert_order.resize(test.n_cache);
    result.miss_expert.resize(test.n_cache);
    result.miss_slot.resize(test.n_cache);
    result.cache_ids.resize(n_ids);
    result.expert_to_cache = make_expert_to_cache(test);
    result.cache_to_expert = test.cache_to_expert;
    result.last_used = test.last_used;
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

    std::vector<bool> requested(test.n_expert, false);
    std::vector<int32_t> unique;
    for (int32_t expert : test.ids) {
        if (!requested[expert]) {
            requested[expert] = true;
            unique.push_back(expert);
        }
    }

    result.n_active = unique.size();
    for (int32_t expert : unique) {
        if (result.expert_to_cache[expert] >= 0) {
            result.expert_order[result.n_resident++] = expert;
        }
    }

    for (int32_t expert : unique) {
        if (result.expert_to_cache[expert] >= 0) {
            continue;
        }

        int32_t victim = -1;
        for (int32_t slot = 0; slot < test.n_cache; ++slot) {
            if (result.cache_to_expert[slot] < 0) {
                victim = slot;
                break;
            }
        }
        if (victim < 0) {
            uint64_t oldest = UINT64_MAX;
            for (int32_t slot = 0; slot < test.n_cache; ++slot) {
                const int32_t resident = result.cache_to_expert[slot];
                if (!requested[resident] && result.last_used[slot] < oldest) {
                    oldest = result.last_used[slot];
                    victim = slot;
                }
            }
        }
        if (victim < 0) {
            throw std::runtime_error(test.name + ": CPU reference could not select a victim");
        }

        const int32_t evicted = result.cache_to_expert[victim];
        if (evicted >= 0) {
            result.expert_to_cache[evicted] = -1;
            result.n_evictions++;
        }
        result.cache_to_expert[victim] = expert;
        result.expert_order[result.n_resident + result.n_miss] = expert;
        result.miss_expert[result.n_miss] = expert;
        result.miss_slot[result.n_miss] = victim;
        result.n_miss++;
    }

    for (uint32_t i = 0; i < result.n_miss; ++i) {
        const int32_t expert = result.miss_expert[i];
        const int32_t slot = result.miss_slot[i];
        result.expert_to_cache[expert] = slot;
        result.cache_to_expert[slot] = expert;
    }

    result.use_clock++;
    for (int32_t expert : unique) {
        result.last_used[result.expert_to_cache[expert]] = result.use_clock;
    }
    for (int32_t i = 0; i < n_ids; ++i) {
        result.cache_ids[i] = result.expert_to_cache[test.ids[i]];
    }
    return result;
}

template<typename T>
static void expect_equal(
        const std::string & name,
        const char * field,
        const std::vector<T> & actual,
        const std::vector<T> & expected,
        size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (actual[i] != expected[i]) {
            throw std::runtime_error(
                name + ": " + field + "[" + std::to_string(i) + "] expected " +
                std::to_string(expected[i]) + ", got " + std::to_string(actual[i]));
        }
    }
}

static void run_case(const test_case & test, hipStream_t stream) {
    const int32_t n_ids = test.n_tokens * test.n_expert_used;
    if ((int32_t) test.ids.size() != n_ids ||
            (int32_t) test.cache_to_expert.size() != test.n_cache ||
            (int32_t) test.last_used.size() != test.n_cache) {
        throw std::runtime_error(test.name + ": invalid test data");
    }

    const plan_result expected = make_reference(test);
    const std::vector<int32_t> initial_expert_to_cache = make_expert_to_cache(test);

    device_buffer<int32_t> ids(n_ids);
    device_buffer<int32_t> ids_src(n_ids);
    device_buffer<int32_t> ids_dst(n_ids);
    device_buffer<int32_t> expert_bounds(test.n_expert + 1);
    device_buffer<int32_t> expert_order(test.n_cache);
    device_buffer<int32_t> miss_expert(test.n_cache);
    device_buffer<int32_t> miss_slot(test.n_cache);
    device_buffer<int32_t> cache_ids(n_ids);
    device_buffer<int32_t> expert_to_cache(test.n_expert);
    device_buffer<int32_t> cache_to_expert(test.n_cache);
    device_buffer<uint64_t> last_used(test.n_cache);
    device_buffer<uint64_t> use_clock(1);
    device_buffer<uint32_t> n_active(1);
    device_buffer<uint32_t> n_resident(1);
    device_buffer<uint32_t> n_miss(1);
    device_buffer<uint32_t> n_evictions(1);

    ids.set(test.ids);
    expert_to_cache.set(initial_expert_to_cache);
    cache_to_expert.set(test.cache_to_expert);
    last_used.set(test.last_used);
    use_clock.set({ test.use_clock });

    ggml_cuda_expert_plan plan = {};
    plan.ids_src = ids_src.ptr;
    plan.ids_dst = ids_dst.ptr;
    plan.expert_bounds = expert_bounds.ptr;
    plan.expert_order = expert_order.ptr;
    plan.miss_expert = miss_expert.ptr;
    plan.miss_slot = miss_slot.ptr;
    plan.cache_ids = cache_ids.ptr;
    plan.expert_to_cache = expert_to_cache.ptr;
    plan.cache_to_expert = cache_to_expert.ptr;
    plan.last_used = last_used.ptr;
    plan.use_clock = use_clock.ptr;
    plan.n_active = n_active.ptr;
    plan.n_resident = n_resident.ptr;
    plan.n_miss = n_miss.ptr;
    plan.n_evictions = n_evictions.ptr;

    ggml_cuda_launch_expert_plan(
        ids.ptr,
        plan,
        test.n_expert,
        test.n_tokens,
        test.n_expert_used,
        test.n_expert_used,
        test.n_expert_used,
        test.n_expert_used,
        false,
        test.n_cache,
        stream);
    hip_check(hipGetLastError(), "ggml_cuda_launch_expert_plan");
    hip_check(hipStreamSynchronize(stream), "hipStreamSynchronize");

    expect_equal(test.name, "ids_src", ids_src.get(), expected.ids_src, n_ids);
    expect_equal(test.name, "ids_dst", ids_dst.get(), expected.ids_dst, n_ids);
    expect_equal(test.name, "expert_bounds", expert_bounds.get(), expected.expert_bounds, test.n_expert + 1);
    expect_equal(test.name, "expert_order", expert_order.get(), expected.expert_order, expected.n_active);
    expect_equal(test.name, "miss_expert", miss_expert.get(), expected.miss_expert, expected.n_miss);
    expect_equal(test.name, "miss_slot", miss_slot.get(), expected.miss_slot, expected.n_miss);
    expect_equal(test.name, "cache_ids", cache_ids.get(), expected.cache_ids, n_ids);
    expect_equal(test.name, "expert_to_cache", expert_to_cache.get(), expected.expert_to_cache, test.n_expert);
    expect_equal(test.name, "cache_to_expert", cache_to_expert.get(), expected.cache_to_expert, test.n_cache);
    expect_equal(test.name, "last_used", last_used.get(), expected.last_used, test.n_cache);
    expect_equal(test.name, "use_clock", use_clock.get(), std::vector<uint64_t>{ expected.use_clock }, 1);
    expect_equal(test.name, "n_active", n_active.get(), std::vector<uint32_t>{ expected.n_active }, 1);
    expect_equal(test.name, "n_resident", n_resident.get(), std::vector<uint32_t>{ expected.n_resident }, 1);
    expect_equal(test.name, "n_miss", n_miss.get(), std::vector<uint32_t>{ expected.n_miss }, 1);
    expect_equal(test.name, "n_evictions", n_evictions.get(), std::vector<uint32_t>{ expected.n_evictions }, 1);

    std::printf("%s: OK\n", test.name.c_str());
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
        hip_check(hipStreamCreate(&stream), "hipStreamCreate");

        run_case({
            "cold",
            16,
            8,
            4,
            2,
            { 3, 7, 1, 5, 3, 1, 7, 5 },
            std::vector<int32_t>(8, -1),
            std::vector<uint64_t>(8, 0),
            0,
        }, stream);

        run_case({
            "partial",
            16,
            8,
            8,
            2,
            { 8, 2, 0, 3, 8, 3, 2, 0, 9, 8, 0, 9, 3, 2, 8, 0 },
            { 0, 8, 10, 12, 14, 15, 6, 7 },
            { 20, 10, 1, 2, 3, 4, 5, 6 },
            20,
        }, stream);

        run_case({
            "full",
            16,
            16,
            8,
            2,
            identity_experts(16),
            identity_experts(16),
            std::vector<uint64_t>(16, 4),
            4,
        }, stream);

        std::vector<int32_t> duplicate_ids;
        for (int32_t token = 0; token < 64; ++token) {
            duplicate_ids.insert(duplicate_ids.end(), { 3, 7, 11, 15 });
        }
        run_case({
            "duplicate-heavy",
            32,
            8,
            64,
            4,
            std::move(duplicate_ids),
            std::vector<int32_t>(8, -1),
            std::vector<uint64_t>(8, 0),
            0,
        }, stream);

        run_case({
            "all-256",
            256,
            256,
            32,
            8,
            identity_experts(256),
            std::vector<int32_t>(256, -1),
            std::vector<uint64_t>(256, 0),
            0,
        }, stream);
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
