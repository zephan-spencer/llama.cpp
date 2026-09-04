#pragma once

#include "ggml-backend-moe-cache.h"
#include "llama.h"
#include "llama-arch.h"

#include <map>
#include <memory>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct llama_model;
struct llama_moe_cache_group;

bool llama_moe_cache_is_routed_weight(llm_tensor tensor, const char * suffix);
void llama_moe_cache_validate_model(
    const llama_model & model,
    uint32_t n_cache_experts,
    enum llama_moe_cache_policy policy);

class llama_moe_cache_layer {
  public:
    ggml_tensor * mul_mat(ggml_tensor * weight, ggml_tensor * input) const;

  private:
    friend class llama_moe_expert_cache;

    llama_moe_cache_layer(ggml_context *          ctx,
                          llama_moe_cache_group * group,
                          ggml_tensor *           logical_ids,
                          ggml_tensor *           plan);

    ggml_context *          ctx;
    llama_moe_cache_group * group;
    ggml_tensor *           logical_ids;
    ggml_tensor *           plan;
};

class llama_moe_expert_cache {
  public:
    llama_moe_expert_cache(const llama_model &                 model,
                           const std::vector<ggml_backend_t> & backends,
                           uint32_t                            n_cache_experts,
                           enum llama_moe_cache_policy         policy);
    ~llama_moe_expert_cache();

    llama_moe_cache_layer build_layer(ggml_context *       ctx,
                                      ggml_backend_sched_t sched,
                                      int                  il,
                                      ggml_tensor *        logical_ids);

    void print_info();

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

  private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
