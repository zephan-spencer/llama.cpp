#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/bench-moe-cache.sh --model MODEL --quant QUANT [options]

Run a fully-resident baseline and MoE expert-cache benchmarks with llama-bench.

MODEL may be a local GGUF path or a Hugging Face repository. For a Hugging
Face repository, QUANT may be a quantization name (for example Q4_K_XL) or an
exact GGUF filename. For a local GGUF path, QUANT is only descriptive.

Options:
  --model PATH_OR_REPO       local GGUF path or Hugging Face repository (required)
  --quant QUANT_OR_FILE      quantization name or GGUF filename
  --hip-devices IDS          physical HIP IDs, comma-separated (default: 0)
  --tensor-parallel          use split-mode tensor across the selected devices
  --tensor-split SPLIT       tensor proportions, e.g. 1/1 (default: automatic)
  --main-gpu INDEX           logical visible GPU for small/intermediate tensors (default: 0)
  --cache-sizes LIST         comma-separated cache sizes (default: 32,64,128,196,255)
  --prompt TOKENS            prompt length (default: 16384; use 0 to disable)
  --generation TOKENS        generated tokens (default: 128; use 0 to disable)
  --batch-size TOKENS        logical batch size (default: 4096)
  --ubatch-size TOKENS       physical ubatch size (default: 4096)
  --repetitions N            repetitions per benchmark (default: 5)
  --flash-attn MODE          on, off, or auto (default: on)
  --threads N                CPU thread count (default: llama-bench default)
  --load-mode MODE           auto, none, mmap, mlock, mmap+mlock, or dio
  --binary PATH              llama-bench executable
                             (default: build-hip-tensor/bin/llama-bench)
  --output PATH              JSONL output (default: benchmark-results/moe-cache.jsonl)
  --force                    overwrite an existing output and log
  --no-warmup                pass --no-warmup to llama-bench
  -h, --help                 show this help

Examples:
  # One physical GPU: HIP device 1, with prompt and generation measurements.
  scripts/bench-moe-cache.sh \
    --model ggml-org/Qwen3.6-35B-A3B-GGUF --quant Q4_K_XL \
    --hip-devices 1 --output benchmark-results/qwen35-q4-single.jsonl

  # Two physical GPUs using tensor parallelism and equal tensor splits.
  scripts/bench-moe-cache.sh \
    --model ggml-org/Qwen3.6-35B-A3B-GGUF --quant Q4_K_XL \
    --hip-devices 0,1 --tensor-parallel --tensor-split 1/1 \
    --output benchmark-results/qwen35-q4-tensor.jsonl
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

need_value() {
    (($# >= 2)) || die "missing value for $1"
}

is_uint() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)

model=''
quant=''
hip_devices='0'
tensor_parallel=0
tensor_split=''
main_gpu=0
cache_sizes='32,64,128,196,255'
prompt_tokens=16384
generation_tokens=128
batch_size=4096
ubatch_size=4096
repetitions=5
flash_attn='on'
threads=''
load_mode=''
binary="$repo_root/build-hip-tensor/bin/llama-bench"
output='benchmark-results/moe-cache.jsonl'
force=0
no_warmup=0

while (($#)); do
    case "$1" in
        --model)
            need_value "$@"
            model=$2
            shift 2
            ;;
        --quant)
            need_value "$@"
            quant=$2
            shift 2
            ;;
        --hip-devices)
            need_value "$@"
            hip_devices=$2
            shift 2
            ;;
        --tensor-parallel)
            tensor_parallel=1
            shift
            ;;
        --tensor-split)
            need_value "$@"
            tensor_split=$2
            shift 2
            ;;
        --main-gpu)
            need_value "$@"
            main_gpu=$2
            shift 2
            ;;
        --cache-sizes)
            need_value "$@"
            cache_sizes=$2
            shift 2
            ;;
        --prompt)
            need_value "$@"
            prompt_tokens=$2
            shift 2
            ;;
        --generation)
            need_value "$@"
            generation_tokens=$2
            shift 2
            ;;
        --batch-size)
            need_value "$@"
            batch_size=$2
            shift 2
            ;;
        --ubatch-size)
            need_value "$@"
            ubatch_size=$2
            shift 2
            ;;
        --repetitions)
            need_value "$@"
            repetitions=$2
            shift 2
            ;;
        --flash-attn)
            need_value "$@"
            flash_attn=$2
            shift 2
            ;;
        --threads)
            need_value "$@"
            threads=$2
            shift 2
            ;;
        --load-mode)
            need_value "$@"
            load_mode=$2
            shift 2
            ;;
        --binary)
            need_value "$@"
            binary=$2
            shift 2
            ;;
        --output)
            need_value "$@"
            output=$2
            shift 2
            ;;
        --force)
            force=1
            shift
            ;;
        --no-warmup)
            no_warmup=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

[[ -n "$model" ]] || die "--model is required"
if [[ ! -f "$model" && -z "$quant" ]]; then
    die "--quant is required when --model is a Hugging Face repository"
fi
[[ -x "$binary" ]] || die "llama-bench is not executable: $binary"

for number_name in prompt_tokens generation_tokens batch_size ubatch_size repetitions main_gpu; do
    number_value=${!number_name}
    is_uint "$number_value" || die "$number_name must be a non-negative integer: $number_value"
done
(( prompt_tokens > 0 || generation_tokens > 0 )) || die "at least one of --prompt and --generation must be greater than zero"
(( batch_size >= ubatch_size )) || die "--batch-size must be at least --ubatch-size"
(( repetitions > 0 )) || die "--repetitions must be positive"
case "$flash_attn" in
    on|off|auto) ;;
    *) die "--flash-attn must be on, off, or auto" ;;
esac

IFS=',' read -r -a hip_ids <<< "$hip_devices"
(( ${#hip_ids[@]} > 0 )) || die "--hip-devices cannot be empty"

visible_devices=''
device_arg=''
for index in "${!hip_ids[@]}"; do
    hip_id=${hip_ids[$index]//[[:space:]]/}
    is_uint "$hip_id" || die "invalid HIP device ID: ${hip_ids[$index]}"
    if (( index > 0 )); then
        visible_devices+=','
        device_arg+='/'
    fi
    visible_devices+="$hip_id"
    device_arg+="ROCm${index}"
done

(( main_gpu < ${#hip_ids[@]} )) || die "--main-gpu must refer to a selected logical device"
if (( ${#hip_ids[@]} > 1 && !tensor_parallel )); then
    die "multiple HIP devices require --tensor-parallel"
fi

IFS=',' read -r -a cache_values <<< "$cache_sizes"
(( ${#cache_values[@]} > 0 )) || die "--cache-sizes cannot be empty"
for cache in "${cache_values[@]}"; do
    is_uint "$cache" || die "invalid cache size: $cache"
    (( cache > 0 )) || die "cache sizes must be positive: $cache"
done

split_mode='none'
if (( tensor_parallel )); then
    split_mode='tensor'
fi

model_args=()
if [[ -f "$model" ]]; then
    model_args=(-m "$model")
elif [[ "$model" == *.gguf || "$model" == /* || "$model" == ./* ]]; then
    die "local model does not exist: $model"
elif [[ "$quant" == *.gguf ]]; then
    model_args=(-hfr "$model" -hff "$quant")
else
    model_args=(-hfr "${model}:${quant}")
fi

output_dir=$(dirname -- "$output")
mkdir -p -- "$output_dir"
log="${output}.stderr.log"
if [[ -e "$output" || -e "$log" ]] && (( ! force )); then
    die "output or log already exists; choose another --output or use --force"
fi
: > "$output"
: > "$log"

common_args=(
    "${model_args[@]}"
    -p "$prompt_tokens"
    -n "$generation_tokens"
    -b "$batch_size"
    -ub "$ubatch_size"
    -r "$repetitions"
    -ctk f16
    -ctv f16
    -fa "$flash_attn"
    -o jsonl
    -dev "$device_arg"
    -sm "$split_mode"
    -mg "$main_gpu"
)
if [[ -n "$tensor_split" ]]; then
    common_args+=(-ts "$tensor_split")
fi
if [[ -n "$threads" ]]; then
    is_uint "$threads" || die "--threads must be a non-negative integer: $threads"
    common_args+=(-t "$threads")
fi
if [[ -n "$load_mode" ]]; then
    common_args+=(--load-mode "$load_mode")
fi
if (( no_warmup )); then
    common_args+=(--no-warmup)
fi

run_case() {
    local label=$1
    shift
    local -a command=("$binary" "${common_args[@]}" "$@")

    {
        printf '\n=== %s ===\n' "$label"
        printf 'HIP_VISIBLE_DEVICES=%s\n' "$visible_devices"
        printf 'physical_hip_devices=%s\n' "$hip_devices"
        printf 'command:'
        printf ' %q' "${command[@]}"
        printf '\n'
    } >> "$log"

    HIP_VISIBLE_DEVICES="$visible_devices" "${command[@]}" >> "$output" 2>> "$log"
}

printf 'Writing benchmark JSONL to %s\n' "$output"
printf 'Writing stderr and command log to %s\n' "$log"
printf 'Selected physical HIP devices: %s\n' "$hip_devices"
if (( tensor_parallel )); then
    printf 'Tensor parallelism: enabled\n'
else
    printf 'Tensor parallelism: disabled\n'
fi

run_case 'fully resident baseline'
run_case "expert cache sizes: $cache_sizes" --moe-cache-experts "$cache_sizes"

printf 'Done. The plotter can consume %s directly.\n' "$output"
