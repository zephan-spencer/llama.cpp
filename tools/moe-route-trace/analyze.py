#!/usr/bin/env python3

import argparse
import json
import statistics
from collections import OrderedDict
from pathlib import Path


DEFAULT_CAPACITIES = (64, 128, 192, 256)


def load_trace(path):
    with Path(path).open() as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if not rows or rows[0].get("type") != "metadata":
        raise ValueError(f"{path}: missing metadata record")

    metadata = rows[0]
    if metadata.get("schema") != 2:
        raise ValueError(f"{path}: unsupported trace schema")
    routed_layers = metadata["routed_layers"]
    if (
        not routed_layers
        or routed_layers != sorted(set(routed_layers))
        or len(routed_layers) != metadata["n_routed_layers"]
    ):
        raise ValueError(f"{path}: invalid routed-layer metadata")
    n_expert_used = metadata["n_expert_used"]
    n_experts = metadata["n_experts"]
    ubatches = OrderedDict()

    for row in rows[1:]:
        if row.get("type") != "routes":
            raise ValueError(f"{path}: unexpected record type")
        key = row["ubatch"]
        ubatch = ubatches.setdefault(key, {
            "token_start": row["token_start"],
            "n_tokens": row["n_tokens"],
            "layers": {},
        })
        if (
            ubatch["token_start"] != row["token_start"]
            or ubatch["n_tokens"] != row["n_tokens"]
        ):
            raise ValueError(f"{path}: inconsistent ubatch {key}")
        layer = row["layer"]
        if layer in ubatch["layers"]:
            raise ValueError(f"{path}: duplicate ubatch {key}, layer {layer}")
        experts = row["experts"]
        if len(experts) != row["n_tokens"] * n_expert_used:
            raise ValueError(f"{path}: wrong route count in ubatch {key}, layer {layer}")
        for offset in range(0, len(experts), n_expert_used):
            route = experts[offset:offset + n_expert_used]
            if (
                len(set(route)) != n_expert_used
                or min(route) < 0
                or max(route) >= n_experts
            ):
                raise ValueError(f"{path}: invalid route in ubatch {key}, layer {layer}")
        ubatch["layers"][layer] = experts

    expected_start = 0
    for expected_index, (index, ubatch) in enumerate(ubatches.items()):
        if index != expected_index:
            raise ValueError(f"{path}: non-contiguous ubatch indices")
        if ubatch["token_start"] != expected_start:
            raise ValueError(f"{path}: non-contiguous token ranges")
        if set(ubatch["layers"]) != set(routed_layers):
            raise ValueError(f"{path}: ubatch {index} does not contain every layer")
        expected_start += ubatch["n_tokens"]
    if expected_start != metadata["prompt_tokens"]:
        raise ValueError(f"{path}: trace does not cover the prompt")
    if len(ubatches) != metadata["n_ubatches"]:
        raise ValueError(f"{path}: metadata ubatch count mismatch")

    return metadata, list(ubatches.values())


def active_experts(experts):
    return list(dict.fromkeys(experts))


def flatten_routes(metadata, ubatches):
    routed_layers = metadata["routed_layers"]
    routes = [[] for _ in routed_layers]
    n_expert_used = metadata["n_expert_used"]
    for source in ubatches:
        for index, layer in enumerate(routed_layers):
            experts = source["layers"][layer]
            routes[index].extend(
                experts[offset:offset + n_expert_used]
                for offset in range(0, len(experts), n_expert_used)
            )
    return routes


def rechunk(metadata, ubatches, n_ubatch):
    routes = flatten_routes(metadata, ubatches)
    result = []
    for token_start in range(0, metadata["prompt_tokens"], n_ubatch):
        n_tokens = min(n_ubatch, metadata["prompt_tokens"] - token_start)
        layers = {}
        for index, layer in enumerate(metadata["routed_layers"]):
            layers[layer] = [
                expert
                for route in routes[index][token_start:token_start + n_tokens]
                for expert in route
            ]
        result.append({
            "token_start": token_start,
            "n_tokens": n_tokens,
            "layers": layers,
        })
    return result


def compare_traces(reference, reference_trace, candidate, candidate_trace):
    reference_metadata, reference_ubatches = reference_trace
    candidate_metadata, candidate_ubatches = candidate_trace
    compared_fields = (
        "model_path",
        "model_description",
        "model_size",
        "prompt_token_hash",
        "prompt_tokens",
        "n_model_layers",
        "routed_layers",
        "n_routed_layers",
        "n_experts",
        "n_expert_used",
    )
    for field in compared_fields:
        if reference_metadata[field] != candidate_metadata[field]:
            raise ValueError(
                f"cannot compare traces with different {field}: "
                f"{reference} and {candidate}")

    reference_routes = flatten_routes(reference_metadata, reference_ubatches)
    candidate_routes = flatten_routes(candidate_metadata, candidate_ubatches)
    ordered_matches = 0
    set_matches = 0
    jaccard_sum = 0
    n_routes = (
        reference_metadata["prompt_tokens"] *
        reference_metadata["n_routed_layers"]
    )

    for reference_layer, candidate_layer in zip(reference_routes, candidate_routes):
        for reference_route, candidate_route in zip(reference_layer, candidate_layer):
            ordered_matches += reference_route == candidate_route
            reference_set = set(reference_route)
            candidate_set = set(candidate_route)
            set_matches += reference_set == candidate_set
            jaccard_sum += len(reference_set & candidate_set) / len(reference_set | candidate_set)

    return {
        "reference": str(reference),
        "candidate": str(candidate),
        "route_count": n_routes,
        "ordered_exact_fraction": ordered_matches / n_routes,
        "set_exact_fraction": set_matches / n_routes,
        "mean_jaccard": jaccard_sum / n_routes,
    }


class LayerCache:
    def __init__(self, capacity, n_experts, preload):
        self.capacity = capacity
        self.clock = 0
        self.last_used = (
            {expert: 0 for expert in range(n_experts)}
            if preload
            else {}
        )

    def execute_ubatch(self, active):
        resident = [expert for expert in active if expert in self.last_used]
        missing = [expert for expert in active if expert not in self.last_used]

        for expert in resident:
            self.clock += 1
            self.last_used[expert] = self.clock
        for expert in missing:
            if len(self.last_used) == self.capacity:
                victim = min(self.last_used, key=self.last_used.get)
                del self.last_used[victim]
            self.clock += 1
            self.last_used[expert] = self.clock
        return len(resident), len(missing)


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return 0
    index = round((len(values) - 1) * fraction)
    return values[index]


def simulate(metadata, ubatches, capacity, expert_bytes):
    preload = capacity == metadata["n_experts"]
    caches = [
        LayerCache(capacity, metadata["n_experts"], preload)
        for _ in metadata["routed_layers"]
    ]
    unique_counts = []
    hits = 0
    misses = 0
    fitting = 0

    for ubatch in ubatches:
        for index, layer in enumerate(metadata["routed_layers"]):
            active = active_experts(ubatch["layers"][layer])
            unique_counts.append(len(active))
            fitting += len(active) <= capacity
            n_hit, n_miss = caches[index].execute_ubatch(active)
            hits += n_hit
            misses += n_miss

    tasks = hits + misses
    prompt_tokens = metadata["prompt_tokens"]
    result = {
        "capacity": capacity,
        "initial_residency": "preloaded" if preload else "empty",
        "layer_ubatches": len(unique_counts),
        "unique_experts": {
            "mean": statistics.fmean(unique_counts),
            "p50": percentile(unique_counts, 0.50),
            "p95": percentile(unique_counts, 0.95),
            "max": max(unique_counts),
        },
        "ubatches_fitting_cache": fitting,
        "ubatches_fitting_cache_fraction": fitting / len(unique_counts),
        "resident_tasks": hits,
        "missing_tasks": misses,
        "task_hit_rate": hits / tasks,
        "missing_tasks_per_token": misses / prompt_tokens,
    }
    if expert_bytes is not None:
        result["transfer_bytes"] = misses * expert_bytes
        result["transfer_bytes_per_token"] = misses * expert_bytes / prompt_tokens
    return result


def analyze(path, trace, capacities, logical_ubatches, expert_bytes):
    metadata, ubatches = trace
    if logical_ubatches is None:
        logical_ubatches = (metadata["n_ubatch"],)
    results = []
    for n_ubatch in logical_ubatches:
        captured = n_ubatch == metadata["n_ubatch"]
        logical_routes = ubatches if captured else rechunk(metadata, ubatches, n_ubatch)
        results.append({
            "n_ubatch": n_ubatch,
            "n_ubatches": len(logical_routes),
            "boundary_source": "captured" if captured else "counterfactual_rechunk",
            "capacities": [
                simulate(metadata, logical_routes, capacity, expert_bytes)
                for capacity in capacities
            ],
        })
    return {
        "trace": str(path),
        "metadata": metadata,
        "logical_ubatches": results,
    }


def self_test():
    cache = LayerCache(2, 4, False)
    assert cache.execute_ubatch([0, 1]) == (0, 2)
    assert cache.execute_ubatch([1, 2]) == (1, 1)
    assert set(cache.last_used) == {1, 2}
    assert cache.execute_ubatch([3, 0, 1]) == (1, 2)
    assert set(cache.last_used) == {0, 3}

    cache = LayerCache(2, 2, True)
    assert cache.execute_ubatch([0, 1]) == (2, 0)
    assert set(cache.last_used) == {0, 1}

    assert active_experts([2, 1, 2, 3, 1]) == [2, 1, 3]
    assert percentile([1, 2, 3, 4, 5], 0.95) == 5

    metadata = {
        "routed_layers": [0],
        "n_expert_used": 2,
        "prompt_tokens": 3,
    }
    ubatches = [{
        "n_tokens": 3,
        "layers": {0: [0, 1, 1, 2, 2, 3]},
    }]
    split = rechunk(metadata, ubatches, 2)
    assert [item["n_tokens"] for item in split] == [2, 1]
    assert split[0]["layers"][0] == [0, 1, 1, 2]
    assert split[1]["layers"][0] == [2, 3]

    variable_metadata = {
        "routed_layers": [0],
        "n_expert_used": 2,
        "n_experts": 4,
        "prompt_tokens": 4,
        "n_ubatch": 2,
    }
    variable_ubatches = [
        {"n_tokens": 1, "layers": {0: [0, 1]}},
        {"n_tokens": 1, "layers": {0: [1, 2]}},
        {"n_tokens": 2, "layers": {0: [2, 3, 3, 0]}},
    ]
    captured = analyze(
        "self-test", (variable_metadata, variable_ubatches), (2,), None, None)
    assert captured["logical_ubatches"][0]["boundary_source"] == "captured"
    assert captured["logical_ubatches"][0]["n_ubatches"] == 3


def main():
    parser = argparse.ArgumentParser(
        description="Replay prompt MoE routes through a resident-first expert cache")
    parser.add_argument("traces", nargs="*")
    parser.add_argument(
        "--capacities",
        default=",".join(map(str, DEFAULT_CAPACITIES)))
    parser.add_argument(
        "--logical-ubatches",
        help="comma-separated replay boundaries; defaults to the recorded ubatch size")
    parser.add_argument(
        "--expert-bytes",
        type=int,
        help="measured gate+up+down bytes for one expert; omit to report task counts only")
    parser.add_argument("--output")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        if not args.traces:
            return
    if not args.traces:
        parser.error("at least one trace is required")

    capacities = tuple(int(value) for value in args.capacities.split(","))
    if not capacities or min(capacities) <= 0:
        parser.error("capacities must be positive")
    if args.expert_bytes is not None and args.expert_bytes <= 0:
        parser.error("expert bytes must be positive")
    if args.logical_ubatches:
        logical_ubatches = tuple(
            int(value) for value in args.logical_ubatches.split(","))
        if min(logical_ubatches) <= 0:
            parser.error("logical ubatches must be positive")
    else:
        logical_ubatches = None

    loaded = [
        (path, load_trace(path))
        for path in args.traces
    ]
    for path, (metadata, _) in loaded:
        if max(capacities) > metadata["n_experts"]:
            parser.error(
                f"cache capacity exceeds the {metadata['n_experts']} experts in {path}")
    report = {
        "expert_bytes": args.expert_bytes,
        "cache_policy": {
            "name": "resident-first-sequential-lru-v1",
            "status": "candidate Sprint 3 policy; not the current device resolver",
            "semantics": [
                "execute every active resident expert first",
                "load each expert missing at ubatch entry at most once",
                "execute missing experts in first-route order",
                "update recency after each complete expert task",
                "retain the last capacity experts after sequential LRU eviction",
            ],
            "full_capacity_initial_residency": "preloaded by model initialization",
            "partial_capacity_initial_residency": "empty",
        },
        "traces": [
            analyze(
                path,
                trace,
                capacities,
                logical_ubatches,
                args.expert_bytes)
            for path, trace in loaded
        ],
        "comparisons": [
            compare_traces(loaded[0][0], loaded[0][1], candidate, trace)
            for candidate, trace in loaded[1:]
        ],
    }
    output = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
