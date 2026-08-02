#pragma once

#include "ggml-backend.h"

#include <map>
#include <memory>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct llama_model;

struct llama_moe_cache_binding {
    ggml_tensor * up;
    ggml_tensor * gate;
    ggml_tensor * down;
    ggml_tensor * gate_up;
    ggml_tensor * ids;
};

class llama_moe_expert_cache {
public:
    llama_moe_expert_cache(
            const llama_model & model,
            const std::vector<ggml_backend_t> & backends,
            uint32_t n_cache_experts);
    ~llama_moe_expert_cache();

    llama_moe_cache_binding bind(
            ggml_context * ctx,
            ggml_backend_sched_t sched,
            int il,
            ggml_tensor * ids,
            ggml_tensor * policy,
            ggml_tensor * up,
            ggml_tensor * gate,
            ggml_tensor * down,
            ggml_tensor * gate_up);

    void synchronize();
    void print_info();
    void print_stats() const;
    void reset_stats();

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
