# MoE expert cache

## Purpose

The MoE expert cache lowers GPU memory use for routed expert weights. Full routed weights stay in mapped host memory. A fixed number of experts stay in GPU memory for each routed layer.

This feature only controls weight residency. It keeps model math, routing, batching, and output rules the same.

The cache is off by default. `--moe-cache-experts N` turns it on. `N` must be at least the number of experts used for one token and smaller than the model's expert count.

`--moe-cache-policy lru` selects the residency policy. LRU is the first policy. The public policy field and backend policy contract allow more policies to be added later.

## Scope

The cache applies to routed expert projections in main decoder layers. It covers separate gate and up weights, merged gate-up weights, and down weights.

Shared experts use the normal weight path. Dense layers use the normal weight path. MTP layers use the normal weight path. Each active routed projection must pass backend validation before model loading finishes.

The first backend is ROCm/HIP. It supports one HIP device and the tensor-parallel meta backend over HIP devices. Each layer uses the device chosen by normal model placement.

## Module boundaries

`llama_moe_expert_cache` is the llama-side module. It owns cache settings, one cache group for each routed layer, storage, backend handles, and memory reports.

The model loader asks the backend whether each routed weight can use the cache. It then places eligible full weights in the backend's mapped host buffer. Other weights follow the normal loader path.

The shared graph builder asks `llama_moe_expert_cache` for a graph-scoped layer operation. The operation accepts a logical routed weight and input and builds the corresponding `MUL_MAT_ID`. Cache slots, route plans, and backend bindings remain inside the cache module.

The backend cache interface owns these details:

- source and slot buffer rules
- cache state layout
- policy state
- route plan layout
- slot fills
- selector encoding
- cached matrix dispatch

The llama graph does not read or change backend cache state.

## Backend interface

`ggml_backend_moe_cache_get_interface(device)` returns a cache interface for a supported device.

`supports_weight(device, tensor)` checks one routed weight. This is the one source of truth for backend weight support.

`get_source_buffer_type(device)` returns the mapped host buffer type for full expert weights.

`get_state_size(device, policy, n_expert, n_cache, n_weights)` returns the state size. Zero means the request is invalid.

`create` binds one state tensor, one or more full source tensors, matching GPU slot tensors, a policy, and cache sizes. One handle covers all routed projections in one layer. All projections share the same expert-to-slot map.

`build_plan(handle, context, logical_ids)` returns one opaque graph tensor. Computing it updates residency, fills slots, and saves the backend's route plan. Cached matrix code consumes that tensor together with the logical IDs.

The plan is built once per routed layer and physical batch. Gate, up, gate-up, and down operations share it.

## Tensor contract

Logical IDs use I32 and shape `[n_expert_used, n_tokens]`. Route order is token first:

```text
route = token * n_expert_used + expert_rank
```

A routed weight has one expert in dimension two and uses `ne[3] = 1`. Each expert slice has `nb[2]` bytes.

A slot tensor keeps the source type, row shape, and expert stride. Dimension two changes from the model expert count to the cache size.

The backend's internal selectors use the same route order as logical IDs:

```text
selector >= 0 : GPU cache slot
selector < 0  : host expert, encoded as -logical_expert_id - 1
```

Logical IDs stay in the graph. Bias, scale, LoRA, route weights, callbacks, and graph tools continue to see logical expert IDs.

## LRU policy

Each layer starts with empty slots. The policy handles active experts in first-route order.

For each active expert:

1. A resident expert keeps its slot.
2. A missing expert takes the lowest empty slot.
3. A missing expert may evict the least recently used expert that has no route in this batch.
4. The lowest slot wins an age tie.
5. A missing expert reads from host when every slot is pinned by this batch.

Every active resident expert gets a new age in first-route order. Duplicate routes use one residency decision. Every route still runs once.

All projections for one admitted expert fill the same slot before matrix work reads that slot.

The age clock uses 64 bits. At wrap, the policy ranks occupied slots by old age and slot number. This keeps the same LRU order.

## State rules

Common state contains:

```text
expert_to_slot[n_expert] : int32
slot_to_expert[n_cache]  : int32
fill_expert[n_cache]     : int32
fill_slot[n_cache]       : int32
n_fills                  : uint32
n_active                 : uint32
policy_data              : bytes
```

LRU policy data contains a 64-bit clock and one 64-bit age per slot.

For every resident pair:

```text
expert_to_slot[expert] = slot
slot_to_expert[slot] = expert
```

Each map is the exact inverse of the other. Every projection in the layer has the same logical expert in a given slot.

## Execution order

The graph keeps this order:

```text
logical route IDs
    -> cache plan and slot fills
    -> cached gate/up or gate-up MUL_MAT_ID
    -> activation
    -> cached down MUL_MAT_ID
    -> route weights and reduction
```

The opaque plan tensor links cached `MUL_MAT_ID` work to its cache update. The HIP dispatcher detects this link once. Normal `MUL_MAT_ID` uses the normal dispatch. Cached work uses the cache source dispatch.

Mapped host addresses stay valid for the context lifetime. Host overflow routes read the full expert slice from that mapped storage.

## Failure rules

Cache setup fails with a clear error when:

- the capacity is invalid
- a routed projection has no cache backend
- a routed weight type or layout is unsupported
- mapped host storage is unavailable
- state or slot allocation fails
- backend handle creation fails
- plan creation fails

These checks happen during model or context setup. A cache request does not silently fall back to another feature layout.

## Tests

The shared backend operation test uses the public cache interface. HIP cases cover Q4_K, Q5_K, and MXFP4 weights. Across repeated route patterns and graph reuse, they compare cached numerical results with full GPU weights.

The normal `MUL_MAT_ID` test set continues to cover the standard path.
