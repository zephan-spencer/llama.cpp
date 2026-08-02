# Host-backed MoE expert cache specification

## Requirement language

`MUST`, `REQUIRED`, `SHALL`, `SHOULD`, and `MAY` define requirement levels.

## Scope

Version 1 has this target matrix:

| Property | Required value |
| --- | --- |
| process count | 1 |
| tensor parallelism | 1 |
| expert parallelism | 1 |
| accelerator count | 1 |
| accelerator | AMD Radeon AI PRO R9700 |
| backend | HIP |
| reference model | `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` Q4_K_XL |
| expert weight types | Q4_K, Q5_K, Q6_K |
| projection layouts | gate and up, merged gate-up, down |
| matrix paths | MMQ and MMV |

Version 1 excludes tensor-parallel coordination, expert parallelism, DeepSeek-specific execution, and model-specific fused FFNs.

## Terms

| Term | Definition |
| --- | --- |
| logical expert | Expert selected by the model router. Its ID has range `[0, n_expert)`. |
| cache slot | Persistent GPU storage for one logical expert slice from every cached projection in one layer. |
| resident expert | Logical expert with a valid cache slot. |
| streamed expert | Logical expert executed from its mapped host tensor slice. |
| route | One `(token, expert-rank)` router selection. |
| update route | Route with permission to update cache policy state. |
| read-only route | Route with permission to inspect residency only. |
| policy state | `expert_to_slot`, `slot_to_expert`, `last_used`, and `use_clock`. |
| source selector | Execution metadata selecting a cache slot or the host-source sentinel. |
| plan | One layer's policy update and route classification for one physical ubatch. |

## User interface

`--moe-cache-experts N` controls persistent cache capacity per routed MoE layer.

| Value | Behavior |
| --- | --- |
| omitted | Normal model placement |
| `0` | Normal model placement |
| `n_expert_used <= N < n_expert` | Host-backed routed experts with `N` GPU cache slots per routed layer |

All other positive values are invalid.

Cache mode requires:

- every model layer assigned to one HIP device;
- split mode `none` or single-device `layer` mode;
- device-visible host backing for every routed expert weight tensor;
- complete GPU offload of model layers.

Cache mode rejects:

- tensor split mode;
- CPU MoE placement options;
- routed expert tensor overrides;
- `--no-host`;
- a model with zero routed experts;
- a model with incompatible routed expert dimensions.

Normal placement with the option omitted defines the all-resident performance ceiling and the exclusive full-residency configuration.

## Weight placement

Positive cache capacity assigns these tensors to device-visible host backing:

- routed gate weights;
- routed up weights;
- routed merged gate-up weights;
- routed down weights.

Dense weights, shared experts, routers, bias tensors, scale tensors, embeddings, attention weights, and output weights use normal placement rules.

The loader MUST produce model-lifetime host addresses visible to the target HIP device. Valid implementations include a HIP host buffer or a model-load copy into persistent pinned backing. Global mmap selection MAY supply file data to the load operation; forward execution MUST use the persistent device-visible backing.

Each expert slice occupies `tensor->nb[2]` bytes. Source tensors MUST have contiguous expert slices and `ne[3] = 1`. A slot tensor preserves the source tensor's type, `ne[0]`, `ne[1]`, `nb[0]`, and `nb[1]`, with `ne[2] = N` and `ne[3] = 1`.

## Token policy API

`llama_batch` exposes an optional policy array with one entry per batch token. `llama_ubatch` carries the matching entries for its tokens.

The policy values are:

```text
LLAMA_MOE_CACHE_POLICY_READ_ONLY
LLAMA_MOE_CACHE_POLICY_UPDATE
```

A null policy pointer selects compatibility mode for the complete logical batch. Batch allocation, slicing, sequence reordering, embedding input, token input, graph reuse, and logical-to-physical ubatch conversion MUST preserve policy-to-token association.

Server assignment uses:

| Token source | Policy |
| --- | --- |
| prompt ingestion | `READ_ONLY` |
| generated decode token | `UPDATE` |
| speculative decode token | `UPDATE` |
| accepted draft token | `UPDATE` |

Compatibility mode resolves independently for each layer plan:

| Unique routed working set | Effective policy |
| --- | --- |
| size `<= N` | `UPDATE` for every route |
| size `> N` | `READ_ONLY` for every route |

## Layer ownership and lifetime

Each routed layer owns:

- mapped host tensors;
- persistent GPU cache tensors;
- policy state;
- planner workspace;
- backend source-view descriptors;
- cumulative device counters.

These objects have model or context lifetime. Context destruction releases them after backend completion.

Policy state uses:

```text
expert_to_slot[n_expert] : int32
slot_to_expert[N]        : int32
last_used[N]             : uint64
use_clock                : uint64
```

Initialization sets every mapping entry to `-1`, every recency value to `0`, and `use_clock` to `0`.

For every resident pair `(expert, slot)`:

```text
expert_to_slot[expert] = slot
slot_to_expert[slot] = expert
```

Every other expert and slot has mapping value `-1`.

## Route order

Flattened route index uses token-major order:

```text
route_index = token_index * n_expert_used + expert_rank
```

Unique expert order uses the first flattened occurrence of each logical expert. This order governs admission and overflow selection.

## Planner inputs and outputs

One layer plan consumes:

- logical route IDs with shape `[n_expert_used, n_tokens]`;
- token policy with shape `[n_tokens]`, or the resolved compatibility policy;
- current policy state;
- `n_expert` and `N`.

One layer plan emits:

- compact expert-major route indices;
- expert bounds for compact execution;
- source selectors with the same logical shape as route IDs;
- fill expert IDs;
- fill slot IDs;
- updated policy state;
- counter deltas.

Source selector values use:

```text
selector >= 0 : persistent cache slot
selector = -1 : mapped host source
```

## Deterministic planning algorithm

The planner MUST execute these phases in order.

### 1. Collect update experts

Build `update_experts` from routes whose token policy is `UPDATE`. Deduplicate by logical expert ID and preserve first-route occurrence order.

### 2. Protect update hits

For each expert in `update_experts`, inspect the entry-state map present at plan start.

- A resident expert increments `hits` and protects its slot.
- An absent expert increments `misses`.

Protection lasts through the complete plan.

### 3. Admit update misses

Process absent update experts in `update_experts` order.

Slot selection uses:

1. the lowest-index empty slot;
2. the unprotected slot with the lowest `last_used` value;
3. the lowest slot index for equal `last_used` values;
4. host-source overflow when the eligible-slot set is empty.

Each admitted expert protects its selected slot. Replacing a resident expert increments `evictions`. A completed admission increments `fills`.

### 4. Update recency

A plan containing at least one update expert increments `use_clock` once. Every update expert resident after phase 3 receives the new clock value in its slot.

A plan containing zero update experts preserves the complete policy state byte-for-byte.

### 5. Classify routes

Classify every route from the policy state produced by phase 4.

- A resident logical expert receives its cache slot selector.
- An absent logical expert receives the host-source sentinel.

Route classification preserves route count and route order. Duplicate logical experts produce duplicate execution routes.

### 6. Publish fills

Each fill copies every cached projection slice for the selected logical expert. Backend dependencies gate every consumer of the selected slot on fill completion.

Residency used by execution requires both a valid mapping and completed fill dependency.

## Overflow semantics

An update working set MAY exceed cache capacity. Protected update hits retain their slots. Admissions consume eligible slots in deterministic order. Remaining update experts use host-source selectors.

A read-only route for an expert admitted during the same plan receives the new cache slot selector. A read-only route for an overflow expert receives the host-source sentinel.

Each input route produces one execution route. Overflow changes source selection only.

## Source-aware `mul_mat_id`

Logical route IDs retain model meaning. Source selectors carry physical execution placement.

HIP MMQ and MMV weight address selection uses:

```text
selector >= 0:
    cache_base + selector * cache_tensor.nb[2]

selector = -1:
    host_base + logical_expert_id * host_tensor.nb[2]
```

The source view supplies:

- logical host tensor metadata and device-visible base address;
- persistent cache tensor metadata and device base address;
- per-route source selectors;
- logical route IDs.

Gate, up, merged gate-up, and down projections use the same source-selection contract.

Graph semantics retain logical IDs for:

- routing weights;
- expert bias;
- expert scale;
- LoRA selection;
- tensor naming;
- graph callbacks;
- graph inspection.

The existing MoE graph defines activation, activation clamping, route-weight placement, projection order, expert bias, expert scale, LoRA, and reduction.

## Execution ordering

The backend scheduler MUST establish this dependency order:

```text
router output
    -> policy plan
    -> cache fills
    -> source-aware gate/up or gate-up mul_mat_id
    -> graph activation operations
    -> source-aware down mul_mat_id
    -> graph route weighting and reduction
```

Cache-slot reuse MUST wait for completion of every earlier operation reading that slot. Host mappings MUST exist before graph construction and persist through graph completion.

## Prohibited forward operations

The forward path prohibits:

- route-ID device-to-host transfer;
- policy device-to-host transfer;
- CPU admission, eviction, recency, or classification decisions;
- device synchronization for policy or source selection;
- expert-weight allocation;
- host mapping creation or destruction;
- concatenation of resident and host experts into a temporary weight tensor;
- logical-ID replacement in graph-visible tensors;
- model-specific FFN execution branches.

Backend graph workspace MAY contain activations, quantized activation tiles, route compaction, and operation scratch.

## Failure behavior

Initialization MUST fail before the first forward for:

- unavailable device-visible host backing;
- unsupported HIP weight type;
- incompatible expert dimensions;
- cache allocation failure;
- invalid capacity;
- unsupported device topology.

Planner overflow MUST complete through host-source execution.

A synchronous CPU resolver MAY support non-target backends as a fallback. HIP completion evidence excludes fallback execution.

## Invariants

For every layer and forward:

1. Every route has one logical expert ID and one source selector.
2. Every route executes exactly once.
3. Every cache selector resolves to the routed logical expert at execution time.
4. Every host selector resolves through the routed logical expert ID.
5. Read-only-only input preserves policy state byte-for-byte.
6. Update effects complete before route classification.
7. Duplicate routes preserve multiplicity and create one update touch per unique update expert.
8. Overflow preserves protected update hits.
9. Mapping tables are exact inverses for resident entries.
10. Cache capacity is smaller than routed expert count.
11. Fill completion precedes slot consumption.
12. Prior slot consumption precedes slot replacement.

## Counters

Counters are cumulative across all cached layers in one context. Context performance reset sets every counter to zero. Layer-local counters are valid storage; printed context values equal the sum of stored layer values.

| Counter | Unit | Definition |
| --- | --- | --- |
| `update_touches` | unique experts | Unique update experts processed by policy |
| `read_only_touches` | routes | Read-only routes classified |
| `resident_routes` | routes | Routes assigned a persistent cache slot |
| `streamed_routes` | routes | Routes assigned the host-source sentinel |
| `host_expert_bytes` | bytes | Sum of unique streamed expert slices presented to each projection operation |
| `fills` | experts | Successful admissions |
| `hits` | unique experts | Update experts resident at plan start |
| `misses` | unique experts | Update experts absent at plan start |
| `evictions` | experts | Resident experts displaced by admissions |

For `host_expert_bytes`, uniqueness uses `(layer, operation, logical_expert)` within one forward. The slice size is the source tensor's `nb[2]`. Hardware traffic counters and profiler measurements use separate names.

Required identities for every reporting interval:

```text
hits + misses = update_touches
resident_routes + streamed_routes = total_routes
fills <= misses
```

Counter retrieval MAY synchronize during an explicit statistics request. Execution scheduling excludes synchronization performed solely for reporting.

## Planner validation

GPU tests MUST cover:

| Case | Required assertion |
| --- | --- |
| read-only resident | cache selector; byte-identical policy state |
| read-only absent | host selector; byte-identical policy state |
| mixed policies | update phases precede classification |
| same-plan visibility | read-only route observes update admission |
| overflow | every route classified; excess update experts streamed |
| duplicate update routes | one update touch; full route multiplicity |
| duplicate read-only routes | one classification per route; zero policy mutation |
| empty slots | lowest empty slot selected |
| LRU replacement | lowest recency selected |
| recency tie | lowest slot index selected |
| protected hit | selected slot survives same-plan admissions |
| repeated plan | deterministic state and outputs |

The read-only preservation test compares the entire policy-state byte range before and after execution. Counter storage lies outside that byte range.

## Source-aware operation validation

Backend tests MUST compare source-aware `mul_mat_id` output with the normal logical-expert operation for:

| Dimension | Required cases |
| --- | --- |
| source distribution | all resident, all host, mixed |
| temporal behavior | stable residency, eviction between forwards, slot reuse |
| projection layout | separate gate/up, merged gate-up, down |
| weight type | Q4_K, Q5_K, Q6_K |
| dispatch | MMQ, MMV |
| route pattern | unique experts, duplicates, shared slot across consecutive forwards |

Comparisons use established backend tolerances. A slot-reuse case MUST detect execution with the previous logical expert's weights.

## End-to-end validation

The reference comparison uses normal all-resident placement with `--moe-cache-experts` omitted.

Required workloads:

1. pure prefill;
2. decode after cache warmup;
3. mixed server traffic containing prompt and decode tokens;
4. update working-set overflow;
5. all-read-only execution after cache warmup.

Every comparison records:

- prompt bytes and token IDs;
- random seed;
- sampler settings;
- context, batch, and ubatch sizes;
- cache capacity;
- generated token IDs;
- maximum absolute logit error;
- maximum relative logit error;
- tolerance;
- cache counters before and after the workload.

Deterministic decoding requires exact generated token equality. Logit comparison requires declared absolute and relative tolerances before measurement.

Mixed server validation MUST capture one physical ubatch containing at least one `READ_ONLY` token and at least one `UPDATE` token.

## Performance validation

### Fixed environment

| Property | Value |
| --- | --- |
| physical device | GPU 1 |
| process selector | `HIP_VISIBLE_DEVICES=1` |
| backend device | `ROCm0` |
| model | `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` Q4_K_XL |
| tensor parallelism | 1 |
| ubatch size | 128 |
| prompt lengths | 512, 2048 |
| cache capacities | 32, 64, 128, 192 |
| measured repetitions | 5 |

### Baselines

| Name | Definition |
| --- | --- |
| clean cache baseline | clean `bdd5eaa9b` build with matching options |
| candidate | source-aware implementation under evaluation |
| all-resident ceiling | candidate build with cache option omitted |

Commands, model file, device selection, batch values, context values, warmup, and repetition count MUST match across comparable runs.

### Evidence

Each run stores:

- exact command and environment;
- commit and worktree identity;
- build configuration;
- compiler, HIP, and driver versions;
- raw stdout and stderr;
- per-repetition throughput;
- expert-cache counters;
- `amd-smi` inventory;
- timestamped utilization, VRAM, power, and PCIe telemetry.

Medians use all five measured repetitions. Reports include every tested capacity.

Evidence resides under:

```text
.git/expert-cache-prefill-results/
```

An index maps each reported value to raw command output and telemetry files.

### Acceptance gates

One tested capacity MUST satisfy all gates:

1. maximum absolute and relative logit errors satisfy the declared tolerances;
2. deterministic generated tokens match the all-resident comparison;
3. median prompt throughput improvement is at least 10 percent over the clean cache baseline at 512 tokens;
4. median prompt throughput improvement is at least 10 percent over the clean cache baseline at 2048 tokens;
5. median cached decode throughput is at least 95 percent of the clean cache baseline.

Prompt improvement uses:

```text
(candidate_median / baseline_median - 1) * 100
```

## Route-trace evidence

Route tracing is diagnostic. Valid uses include route-distribution analysis, working-set measurement, and capacity selection.

Primary performance evidence requires the real scheduler boundary and a trace captured with the measured ubatch size. Rechunked traces are labeled `counterfactual_rechunk`.

Policy simulation, synchronized tensor callbacks, and counterfactual boundaries are excluded from production correctness and throughput evidence.

## Future tensor-parallel contract

Each rank owns one policy state and one source view per layer. A logical cache entry maps to one rank-local expert shard on every rank.

Future tensor-parallel execution requires:

- identical logical residency maps across ranks;
- rank-local fills for each admitted logical expert;
- publication after completion of every required rank-local fill;
- coordinated invalidation on eviction;
- rank-local transfer and wait counters.

Version 1 data structures SHOULD preserve this ownership boundary.

## Conformance

A conforming implementation MUST satisfy every requirement in this specification, including:

- token-policy propagation from `llama_batch` through every `llama_ubatch`;
- mixed policy values in a captured server ubatch;
- GPU planning with the specified ordering and overflow behavior;
- byte-identical policy state across all-read-only execution;
- route-level cache or host source selection in HIP MMQ and MMV;
- source-aware gate, up, merged gate-up, and down projections;
- Q4_K, Q5_K, and Q6_K backend coverage;
- pure prefill, decode, mixed, overflow, and warm read-only model validation;
- counters with the specified definitions and identities;
- one capacity satisfying every acceptance gate;
- recorded all-resident ceilings and telemetry;
- production execution with zero model-specific full-FFN cache branches.
