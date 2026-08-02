# Prompt MoE route tracer

`llama-moe-route-trace` records the selected expert IDs for every routed
layer and scheduler ubatch while processing a real prompt. The trace is
diagnostic: the evaluation callback synchronizes each observed routing tensor,
so its timing is not a production performance measurement.

Example:

```sh
env HIP_VISIBLE_DEVICES=1 \
  ./build-hip-tensor/bin/llama-moe-route-trace \
  -m model.gguf -ngl all -dev ROCm0 \
  -c 8192 -b 8192 -ub 128 \
  --moe-cache-experts 0 \
  --prompt-file wikitext-2-raw/wiki.test.raw \
  --prompt-tokens 8192 \
  --trace-output routes-ub128.jsonl

python3 tools/moe-route-trace/analyze.py \
  --capacities 64,128,192,256 \
  --logical-ubatches 8,16,32,64,128,512,2048,8192 \
  --expert-bytes 1900544 \
  routes-ub128.jsonl
```

Each JSONL file starts with a metadata record. The remaining records contain
one flattened `[n_tokens, n_expert_used]` expert-ID matrix for one layer of one
ubatch. `analyze.py` validates the complete trace and replays it with
resident-first execution, one load per missing active expert, and LRU
retention between ubatches. Logical ubatch replay preserves the captured
per-token routes while changing only the task boundaries. Passing multiple
independently captured traces also reports exact-route and Jaccard agreement,
which makes the effect of the real scheduler boundary visible.

Rechunked results are labeled `counterfactual_rechunk` and are sensitivity
analysis only. Capture a separate trace with the real `-ub` value for a primary
measurement. Replay uses the explicitly reported candidate
`resident-first-sequential-lru-v1` policy; it does not claim to reproduce the
current device resolver, which cannot execute an active set larger than its
cache. `--expert-bytes` has no default because it depends on the model
quantization and must be the measured sum of one gate, up, and down expert.
The example value is the Qwen3.6 Q4_K_XL shape used by this experiment.
