#include "ggml-backend-moe-cache.h"

#include "ggml-backend-moe-cache-impl.h"

const ggml_backend_moe_cache_i * ggml_backend_moe_cache_get_interface(ggml_backend_dev_t device) {
    if (device == NULL) {
        return NULL;
    }
    if (ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_META) {
        return ggml_backend_meta_moe_cache_get_interface();
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
