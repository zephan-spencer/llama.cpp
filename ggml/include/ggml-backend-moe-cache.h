#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

    typedef struct ggml_backend_moe_cache * ggml_backend_moe_cache_t;

    #define GGML_BACKEND_MOE_CACHE_MAX_WEIGHTS 4

    struct ggml_backend_moe_cache_plan {
        struct ggml_tensor * selectors;
        struct ggml_tensor * execution;
    };

    struct ggml_backend_moe_cache_plan_layout {
        enum ggml_type execution_type;
        int64_t        execution_ne[GGML_MAX_DIMS];
        enum ggml_type selectors_type;
        int64_t        selectors_ne[GGML_MAX_DIMS];
        size_t         selectors_nb[GGML_MAX_DIMS];
        size_t         selectors_offset;
    };

    struct ggml_backend_moe_cache_i {
        ggml_backend_buffer_type_t (*get_source_buffer_type)(ggml_backend_dev_t device);
        size_t (*get_state_size)(ggml_backend_dev_t device, uint32_t n_expert, uint32_t n_cache, uint32_t n_weights);
        ggml_backend_moe_cache_t (*create)(
            ggml_backend_t backend,
            struct ggml_tensor * state,
            struct ggml_tensor * const * source,
            struct ggml_tensor * const * slots,
            uint32_t n_expert,
            uint32_t n_cache,
            uint32_t n_weights);
        void (*destroy)(ggml_backend_moe_cache_t cache);
        bool (*get_plan_layout)(
            ggml_backend_moe_cache_t cache,
            const struct ggml_tensor * logical_ids,
            struct ggml_backend_moe_cache_plan_layout * layout);
        struct ggml_backend_moe_cache_plan (*build_plan)(
            ggml_backend_moe_cache_t cache,
            struct ggml_context * ctx,
            struct ggml_tensor * logical_ids,
            struct ggml_tensor * token_priority,
            struct ggml_tensor * epoch);
    };

    typedef const struct ggml_backend_moe_cache_i * (*ggml_backend_moe_cache_get_interface_t)(void);

    GGML_API const struct ggml_backend_moe_cache_i * ggml_backend_moe_cache_get_interface(ggml_backend_dev_t device);

#ifdef __cplusplus
}
#endif
