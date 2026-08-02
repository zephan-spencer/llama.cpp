#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include "llama-ext.h"
#include "log.h"

#include <nlohmann/json.hpp>

#include <array>
#include <chrono>
#include <clocale>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <map>
#include <sstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

struct trace_options {
    std::string prompt_path;
    std::string trace_path;
    std::string summary_path;
    int32_t n_prompt = 8192;
    bool trace = true;
};

struct prompt_ubatch {
    int32_t token_start = 0;
    int32_t n_tokens = 0;
    std::map<int32_t, std::vector<int32_t>> layers;
};

class prompt_route_trace {
public:
    void set_expert_count(int32_t value) {
        if (n_experts != 0 && n_experts != value) {
            throw std::runtime_error("model expert count changed during route tracing");
        }
        n_experts = value;
        if (n_experts <= 0) {
            throw std::runtime_error("model does not define routed experts");
        }
    }

    static bool callback(ggml_tensor * tensor, bool ask, void * user_data) {
        auto & trace = *static_cast<prompt_route_trace *>(user_data);

        int32_t il = -1;
        const bool selected =
            trace.capture &&
            parse_layer(tensor->name, il);
        if (ask) {
            return selected;
        }
        if (selected) {
            trace.record(*tensor, il);
        }
        return true;
    }

    void begin() {
        ubatches.clear();
        current = {};
        capture = true;
    }

    void end() {
        capture = false;
        finish_ubatch();
    }

    int32_t n_tokens() const {
        int32_t result = 0;
        for (const prompt_ubatch & ubatch : ubatches) {
            result += ubatch.n_tokens;
        }
        return result;
    }

    int32_t expert_count() const {
        return n_experts;
    }

    int32_t expert_used_count() const {
        return n_expert_used;
    }

    void validate(int32_t expected_tokens) const {
        if (n_tokens() != expected_tokens) {
            throw std::runtime_error(
                "route trace covers " + std::to_string(n_tokens()) +
                " tokens, expected " + std::to_string(expected_tokens));
        }
        if (ubatches.empty()) {
            throw std::runtime_error("route trace did not observe any prompt ubatches");
        }
        if (n_expert_used <= 0) {
            throw std::runtime_error("route trace did not observe an expert top-k width");
        }

        const std::vector<int32_t> expected_layers = routed_layers();
        for (size_t iub = 0; iub < ubatches.size(); ++iub) {
            const prompt_ubatch & ubatch = ubatches[iub];
            std::vector<int32_t> actual_layers;
            for (const auto & [il, experts] : ubatch.layers) {
                GGML_UNUSED(experts);
                actual_layers.push_back(il);
            }
            if (actual_layers != expected_layers) {
                throw std::runtime_error(
                    "ubatch " + std::to_string(iub) +
                    " has a different routed-layer set");
            }
        }
    }

    std::vector<int32_t> routed_layers() const {
        std::vector<int32_t> result;
        if (ubatches.empty()) {
            return result;
        }
        for (const auto & [il, experts] : ubatches.front().layers) {
            GGML_UNUSED(experts);
            result.push_back(il);
        }
        return result;
    }

    void write_jsonl(std::ostream & output, const json & metadata) const {
        output << metadata.dump() << '\n';
        for (size_t iub = 0; iub < ubatches.size(); ++iub) {
            const prompt_ubatch & ubatch = ubatches[iub];
            for (const auto & [il, experts] : ubatch.layers) {
                output << json({
                    { "type", "routes" },
                    { "ubatch", iub },
                    { "token_start", ubatch.token_start },
                    { "n_tokens", ubatch.n_tokens },
                    { "layer", il },
                    { "experts", experts },
                }).dump() << '\n';
            }
        }
    }

    size_t n_ubatches() const {
        return ubatches.size();
    }

private:
    int32_t n_experts = 0;
    int32_t n_expert_used = 0;
    bool capture = false;
    std::vector<prompt_ubatch> ubatches;
    prompt_ubatch current;

    static bool parse_layer(const char * name, int32_t & il) {
        static const std::string prefix = "ffn_moe_topk-";
        const std::string value(name);
        if (value.compare(0, prefix.size(), prefix) != 0) {
            return false;
        }

        char * end = nullptr;
        const long parsed = std::strtol(value.c_str() + prefix.size(), &end, 10);
        if (end == value.c_str() + prefix.size() ||
                *end != '\0' ||
                parsed < 0 ||
                parsed > INT32_MAX) {
            return false;
        }
        il = parsed;
        return true;
    }

    void record(const ggml_tensor & tensor, int32_t il) {
        if (tensor.type != GGML_TYPE_I32 ||
                tensor.ne[0] <= 0 ||
                tensor.ne[0] > n_experts ||
                tensor.src[0] == nullptr ||
                tensor.src[0]->ne[0] != n_experts) {
            throw std::runtime_error(
                "unexpected MoE routing tensor: type=" +
                std::string(ggml_type_name(tensor.type)) +
                ", shape=[" +
                std::to_string(tensor.ne[0]) + "," +
                std::to_string(tensor.ne[1]) + "," +
                std::to_string(tensor.ne[2]) + "," +
                std::to_string(tensor.ne[3]) + "]");
        }
        if (n_expert_used == 0) {
            n_expert_used = tensor.ne[0];
        } else if (tensor.ne[0] != n_expert_used) {
            throw std::runtime_error("routed layers disagree on expert top-k width");
        }

        if (tensor.nb[0] != sizeof(int32_t)) {
            throw std::runtime_error("unexpected MoE routing tensor element stride");
        }

        const int64_t n_tokens_64 = tensor.ne[1] * tensor.ne[2] * tensor.ne[3];
        if (n_tokens_64 <= 0 || n_tokens_64 > INT32_MAX) {
            throw std::runtime_error("invalid MoE routing tensor shape");
        }
        const int32_t n_tokens = n_tokens_64;

        if (!current.layers.empty() && current.layers.count(il) != 0) {
            finish_ubatch();
        }
        if (current.layers.empty()) {
            current.n_tokens = n_tokens;
        } else if (current.n_tokens != n_tokens) {
            throw std::runtime_error("routed layers disagree on ubatch token count");
        }

        const size_t storage_size =
            (tensor.ne[1] - 1) * tensor.nb[1] +
            (tensor.ne[2] - 1) * tensor.nb[2] +
            (tensor.ne[3] - 1) * tensor.nb[3] +
            tensor.ne[0] * tensor.nb[0];
        std::vector<uint8_t> storage(storage_size);
        ggml_backend_tensor_get(
            &tensor,
            storage.data(),
            0,
            storage.size());

        std::vector<int32_t> experts((size_t) n_tokens * n_expert_used);
        int32_t it = 0;
        for (int64_t i3 = 0; i3 < tensor.ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < tensor.ne[2]; ++i2) {
                for (int64_t i1 = 0; i1 < tensor.ne[1]; ++i1, ++it) {
                    const size_t offset =
                        i1 * tensor.nb[1] +
                        i2 * tensor.nb[2] +
                        i3 * tensor.nb[3];
                    std::memcpy(
                        experts.data() + (size_t) it * n_expert_used,
                        storage.data() + offset,
                        n_expert_used * sizeof(experts[0]));
                }
            }
        }

        for (it = 0; it < n_tokens; ++it) {
            std::set<int32_t> unique;
            for (int32_t iex = 0; iex < n_expert_used; ++iex) {
                const int32_t expert = experts[it * n_expert_used + iex];
                if (expert < 0 ||
                        expert >= n_experts ||
                        !unique.insert(expert).second) {
                    throw std::runtime_error("invalid routed expert ID");
                }
            }
        }

        current.layers.emplace(il, std::move(experts));
    }

    void finish_ubatch() {
        if (current.layers.empty()) {
            return;
        }
        current.token_start = n_tokens();
        ubatches.push_back(std::move(current));
        current = {};
    }
};

static std::vector<char *> parse_trace_options(
        int argc,
        char ** argv,
        trace_options & options) {
    std::vector<char *> common_argv = { argv[0] };

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--prompt-file") {
            if (++i >= argc) {
                throw std::runtime_error("--prompt-file requires a path");
            }
            options.prompt_path = argv[i];
        } else if (arg == "--prompt-tokens") {
            if (++i >= argc) {
                throw std::runtime_error("--prompt-tokens requires a count");
            }
            options.n_prompt = std::stoi(argv[i]);
        } else if (arg == "--trace-output") {
            if (++i >= argc) {
                throw std::runtime_error("--trace-output requires a path");
            }
            options.trace_path = argv[i];
        } else if (arg == "--summary-output") {
            if (++i >= argc) {
                throw std::runtime_error("--summary-output requires a path");
            }
            options.summary_path = argv[i];
        } else if (arg == "--no-route-trace") {
            options.trace = false;
        } else {
            common_argv.push_back(argv[i]);
        }
    }

    if (options.prompt_path.empty()) {
        throw std::runtime_error("--prompt-file is required");
    }
    if (options.n_prompt <= 0) {
        throw std::runtime_error("--prompt-tokens must be positive");
    }
    if (options.trace && options.trace_path.empty()) {
        throw std::runtime_error("--trace-output is required unless --no-route-trace is set");
    }
    return common_argv;
}

static std::string read_file(const std::string & path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open prompt file: " + path);
    }
    return {
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>(),
    };
}

static std::string prompt_token_hash(const std::vector<llama_token> & tokens) {
    uint64_t hash = 14695981039346656037ULL;
    for (llama_token token : tokens) {
        const uint32_t value = token;
        for (int shift = 0; shift < 32; shift += 8) {
            hash ^= (value >> shift) & 0xff;
            hash *= 1099511628211ULL;
        }
    }

    std::ostringstream output;
    output << "fnv1a64:" << std::hex << std::setfill('0') << std::setw(16) << hash;
    return output.str();
}

static int run(
        common_params & params,
        const trace_options & options,
        prompt_route_trace & trace) {
    params.cb_eval = options.trace ? prompt_route_trace::callback : nullptr;
    params.cb_eval_user_data = options.trace ? &trace : nullptr;
    params.warmup = false;
    params.n_parallel = 1;
    params.n_sequences = 1;

    auto llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    llama_context * ctx = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        throw std::runtime_error("failed to initialize model or context");
    }
    trace.set_expert_count(llama_model_n_expert(model));

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> prompt = common_tokenize(
        ctx,
        read_file(options.prompt_path),
        llama_vocab_get_add_bos(vocab),
        true);
    if ((int32_t) prompt.size() < options.n_prompt) {
        throw std::runtime_error(
            "prompt file contains only " + std::to_string(prompt.size()) +
            " tokens, requested " + std::to_string(options.n_prompt));
    }
    prompt.resize(options.n_prompt);
    if ((int32_t) prompt.size() > params.n_batch) {
        throw std::runtime_error(
            "prompt token count exceeds batch size; pass -b " +
            std::to_string(prompt.size()) + " or larger");
    }

    if (options.trace) {
        trace.begin();
    }
    const auto begin = std::chrono::steady_clock::now();
    const int decode_result = llama_decode(
        ctx,
        llama_batch_get_one(prompt.data(), prompt.size()));
    const auto end = std::chrono::steady_clock::now();
    if (options.trace) {
        trace.end();
    }
    if (decode_result != 0) {
        throw std::runtime_error("prompt decode failed");
    }

    const double elapsed = std::chrono::duration<double>(end - begin).count();
    if (options.trace) {
        trace.validate(prompt.size());
    }

    std::array<char, 1024> model_description = {};
    llama_model_desc(model, model_description.data(), model_description.size());
    const std::vector<int32_t> routed_layers =
        options.trace ? trace.routed_layers() : std::vector<int32_t>();

    const json summary = {
        { "type", "metadata" },
        { "schema", 2 },
        { "model_path", params.model.path },
        { "model_description", model_description.data() },
        { "model_size", llama_model_size(model) },
        { "prompt_file", options.prompt_path },
        { "prompt_tokens", prompt.size() },
        { "prompt_token_hash", prompt_token_hash(prompt) },
        { "n_batch", params.n_batch },
        { "n_ubatch", params.n_ubatch },
        { "n_ubatches", options.trace ? trace.n_ubatches() : 0 },
        { "n_model_layers", llama_model_n_layer(model) },
        { "routed_layers", routed_layers },
        { "n_routed_layers", routed_layers.size() },
        { "n_experts", trace.expert_count() },
        { "n_expert_used", trace.expert_used_count() },
        { "seconds", elapsed },
        { "tokens_per_second", prompt.size() / elapsed },
        { "traced", options.trace },
    };

    if (options.trace) {
        std::ofstream output(options.trace_path);
        if (!output) {
            throw std::runtime_error("failed to open trace output: " + options.trace_path);
        }
        trace.write_jsonl(output, summary);
    }
    if (!options.summary_path.empty()) {
        std::ofstream output(options.summary_path);
        if (!output) {
            throw std::runtime_error("failed to open summary output: " + options.summary_path);
        }
        output << summary.dump(2) << '\n';
    }

    LOG_INF("%s\n", summary.dump().c_str());
    return 0;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    try {
        trace_options options;
        std::vector<char *> common_argv = parse_trace_options(argc, argv, options);

        common_params params;
        common_init();
        if (!common_params_parse(
                common_argv.size(),
                common_argv.data(),
                params,
                LLAMA_EXAMPLE_COMPLETION)) {
            return 1;
        }

        prompt_route_trace trace;
        llama_backend_init();
        const int result = run(params, options, trace);
        llama_backend_free();
        return result;
    } catch (const std::exception & error) {
        LOG_ERR("%s\n", error.what());
        return 1;
    }
}
