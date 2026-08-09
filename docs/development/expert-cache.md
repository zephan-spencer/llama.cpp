# MoE expert cache specification

## Purpose

The MoE expert cache keeps selected routed expert weight slices in GPU memory while complete routed weights remain in mapped host memory. Each routed layer owns an independent cache with a common capacity. The feature serves models whose complete expert weights exceed available GPU memory.

`--moe-cache-experts N` selects capacity `N`. Zero selects ordinary MoE execution. A positive value requires `n_expert_used <= N < n_expert`.

## Supported configuration

| Property | Current value |
| --- | --- |
| backend | ROCm/HIP |
| accelerator count | one |
| model placement | every model layer on the accelerator |
| split mode | none or single-device layer split |
| routed weight location | accelerator-visible host buffer |
| tested weight types | Q4_K, Q5_K, Q6_K |
| matrix paths | MMQ and MMV |
| projection layouts | gate and up, merged gate-up, down |

Cache initialization rejects CPU layer placement, multiple accelerator devices, tensor splitting, meta devices, unsupported backends, incomplete routed tensors, tensor buffer overrides for routed weights, invalid capacities, and models without routed experts.

CUDA, Vulkan, and meta backends expose cache support after implementing the backend interface. A meta implementation owns rank-local child handles and coordinates route plans after tensor-parallel execution semantics are defined.

## Ownership

`llama_moe_expert_cache` owns feature configuration, per-layer source tensors, cache-slot tensors, state tensors, device buffers, backend handles, synchronization, memory accounting, logical-batch epochs, and aggregate statistics.

One backend handle owns one layer cache. The handle owns cache-state layout, source bindings, slot bindings, selector encoding, route-plan layout, replacement state, transfer scheduling, and backend statistics. Slot and plan bindings remain valid through handle destruction. Handle destruction clears every slot binding before releasing storage.

Source tensors and backend handles have context lifetime. Graph plan tensors have graph lifetime. Cache state and slot storage have context lifetime.

## Backend interface

The backend registry exposes `ggml_backend_moe_cache_i` under `ggml_backend_moe_cache_get_interface`. ROCm supplies interface version 2.

`supports(device)` reports device capability. `get_state_size(n_expert, n_cache, n_weights)` returns required state bytes or zero for an invalid configuration.

`create` receives one backend, one allocated state tensor, ordered source and slot arrays, expert count, capacity, and projection count. Creation validates buffer ownership, host visibility, tensor types, expert dimensions, slot dimensions, and expert strides. Failure returns a null handle and leaves tensor bindings unchanged.

`build_plan(handle, context, logical_ids, token_priority, epoch)` returns:

- `selectors`: integer source selectors consumed by cached `MUL_MAT_ID` operations.
- `execution`: the graph operation that resolves routes, updates state, fills slots, and owns saved route storage.

The selector tensor carries a backend-private association with its handle. Matrix kernels use that association to obtain saved grouped routes. llama graph code treats both returned tensors as opaque backend products.

`get_stats` reads one handle. `reset_stats` clears one handle. The llama owner aggregates every layer.

## Tensor contract

Logical IDs use type I32 and shape `[n_expert_used, n_tokens]`. Flattened route order is token-major:

```text
route = token * n_expert_used + expert_rank
```

`token_priority` uses type I32 and shape `[n_tokens]`. Zero identifies prompt work. A nonzero value identifies output-priority work. `epoch` uses type I32 and shape `[1]`.

A routed source tensor has one expert slice along dimension two and `ne[3] = 1`. Each expert occupies `nb[2]` bytes. Its buffer type equals the target device host buffer type.

A slot tensor preserves source type, `ne[0]`, `ne[1]`, `nb[0]`, `nb[1]`, and `nb[2]`. It uses `ne[2] = N` and `ne[3] = 1`. Every projection in one layer uses the same expert-to-slot mapping.

Selectors have the logical-ID shape and route order:

```text
selector >= 0 : persistent cache slot
selector < 0  : host expert encoded as -logical_expert_id - 1
```

Logical expert IDs remain graph inputs for bias, scale, LoRA, route weighting, graph callbacks, and graph inspection.

## Workload classification

`llama_batch_allocr` classifies each exact sequence set once per logical batch. Output priority applies when every member token requests output, including logits-all prompts and single-token prompts. Sequence sets containing a token that omits an output request use ordinary admission.

Classification occurs before physical ubatch splitting. Every split operation preserves each token's output-priority value. `llama_batch` remains unchanged.

Each `llama_decode` call receives a nonzero epoch. Every physical ubatch produced by that call carries the same epoch. Epoch rollover skips zero.

## State invariants

Each layer state contains:

```text
expert_to_slot[n_expert] : int32
slot_to_expert[N]        : int32
last_used[N]             : uint64
protected_epoch[N]       : uint64
pending_priority_epoch[n_expert] : uint64
use_clock                : uint64
```

Initialization sets mapping entries to `-1`, recency, protection, and pending values to zero, and `use_clock` to zero.

For each resident pair `(expert, slot)`:

```text
expert_to_slot[expert] = slot
slot_to_expert[slot] = expert
```

Every resident mapping has an exact inverse. Each slot contains slices for the same logical expert across every cached projection.

## Admission and overflow

The planner records each expert once in first-occurrence order. Admission phases are active output-priority experts, pending output-priority experts, and active prompt experts. First-occurrence order controls active-phase admission. Logical-expert order controls pending admission.

An active output-priority expert already resident retains its slot and assigns the current epoch to `protected_epoch[slot]`. A missing active output-priority expert selects a slot in this order:

1. Lowest-index empty slot.
2. Slot outside the current epoch whose resident expert has no route in the physical plan, with the lowest `last_used` value.
3. Lowest slot index among equal timestamps.

A prompt expert already resident after output-priority admission retains its slot. A missing prompt expert selects the lowest-index empty slot. Missing prompt experts use the mapped host source after the cache reaches capacity. Prompt admission preserves every occupied slot.

Every expert requested by a physical ubatch retains its slot through that ubatch. A missing active output-priority expert streams from mapped host memory when active routes and current-epoch protection exhaust eviction candidates. The planner records that expert as pending for the current epoch. Each later physical ubatch processes active output-priority routes, same-epoch pending output-priority experts in ascending logical-expert order, then active prompt routes. Pending admission requires an inactive, epoch-unprotected victim. Admission clears the pending entry, protects the new slot for the current epoch, and updates recency. Epoch rollover clears earlier pending entries.

Each fill updates both mapping arrays and copies every cached projection for the admitted expert. A plan containing routes increments `use_clock` once and assigns the new timestamp to every requested expert resident after planning.

When eligible slots consume the capacity, remaining missing experts receive host selectors. Overflow preserves route count, route order, and route multiplicity. Every route executes once.

## Saved route plan and MMQ schedule

One physical ubatch creates one plan per cached layer. The execution tensor stores source selectors, route indices grouped by logical expert, one start/end route range per expert, and one compact route-tile prefix range per expert.

Gate, up, merged gate-up, and down projections consume the same grouped route IDs and expert bounds. A projection derives its activation index with one linear pass over saved route indices.

Cached MMQ selects a route tile width from `n_expert_used` and supported kernel configurations. A GPU kernel converts expert route counts into tile-prefix bounds for that width. For `routes > 0`, the compact launch bound is `ceil(routes / tile_width) + min(experts, routes) - 1`; zero routes use a zero bound. Each block maps its route-tile index to an expert through the prefix bounds. Blocks beyond the populated tile count return before weight or activation reads. Stream-K architectures retain the full-route MMQ bound. Cached MMV retains its existing schedule.

The scheduler establishes this order:

```text
logical route IDs and token priority
    -> priority-aware cache plan and slot fills
    -> compact cached gate/up or gate-up MUL_MAT_ID
    -> activation
    -> compact cached down MUL_MAT_ID
    -> route weighting and reduction
```

Every fill completes before a consumer reads its slot. Every earlier slot consumer completes before replacement writes that slot.

## Forward-path constraints

`llama_batch_allocr` performs classification before physical-ubatch splitting. Accelerator execution begins with route planning, admission, eviction, saved-route construction, and compact tile construction. The forward path prohibits route transfers to the CPU, CPU replacement decisions, source-selection synchronization, expert-weight allocation, host mapping changes, temporary tensors containing the complete expert set, and replacement of graph-visible logical IDs.

Mapped host addresses remain stable through graph completion. Graph execution performs host-source reads directly for streamed experts.

## Failure behavior

Positive cache capacity requires a matching backend interface. Context construction fails before the first forward for unsupported topology, unsupported backend, incompatible routed tensors, unavailable accelerator-visible host backing, cache allocation failure, and backend handle creation failure.

Backend plan construction failure stops graph construction. A cache request with a failed requirement produces an initialization error.

## Counters

`resolve_calls` counts plans. `update_touches` counts unique requested experts. `resident_routes` and `streamed_routes` count routes by selector class. `cache_hits`, `cache_misses`, and `evictions` count expert planning results. `h2d_bytes` counts bytes copied into slots. `host_expert_bytes` counts unique streamed expert slices across cached projections. `fill_ticks` uses each handle's `wall_clock_hz`; aggregate display converts each handle to milliseconds before summation.

Context performance reset clears every layer counter. Printed context values equal the sum of layer values.

## Acceptance

RDNA4 ROCm correctness acceptance covers Q4_K, Q5_K, and Q6_K through MMQ and MMV. Tests exercise resident routes, streamed routes, repeated plans, duplicate routes, complete expert activation, active-route retention, deferred output-priority admission, epoch protection, epoch release, equal-timestamp victim selection, and route-tile launch bounds. Cached matrix results use fully resident routed matrices as the numerical reference.

Interface tests obtain version 2 from the backend registry, reject invalid dimensions and malformed creation, construct a handle with backend-owned buffers, build selector and execution tensors, destroy the handle, and verify slot-binding cleanup.

A configured HIP build includes server, benchmark, batched benchmark, and perplexity targets. `test-moe-expert-plan` and `test-batch-alloc` complete successfully.

Performance acceptance uses synthetic 8196-token inputs for isolated llama-bench PP and TG after a complete warmup. Mixed server measurement uses the WikiText prompt with four established TG streams and one PP stream. Measurements cover cache capacities 32, 64, 128, and 192, plus ordinary all-resident execution, and report throughput, fills, copied bytes, resident routes, and streamed routes. N192 PP targets at least 95 percent of all-resident PP throughput. Cached TG targets at least 95 percent of the checkpoint result.
