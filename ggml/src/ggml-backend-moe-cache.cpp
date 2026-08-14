#include "ggml-backend-moe-cache.h"

const ggml_backend_moe_cache_i * ggml_backend_moe_cache_get_interface(ggml_backend_dev_t device) {
    if (device == NULL) {
        return NULL;
    }

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
    if (reg == NULL) {
        return NULL;
    }

    ggml_backend_moe_cache_get_interface_t get_interface =
        (ggml_backend_moe_cache_get_interface_t) ggml_backend_reg_get_proc_address(
            reg, "ggml_backend_moe_cache_get_interface");
    return get_interface != NULL ? get_interface() : NULL;
}
