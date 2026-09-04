# Repository engineering guidance

## MoE expert cache

- Keep expert-cache behavior in cache-owned files. General llama.cpp and GGML files should contain only the minimum integration hooks needed to call the cache.
- Before changing an upstream file, identify the cache knowledge currently present there and make sure the change removes that knowledge or is required for correct integration. Do not move cache policy, state, layouts, selectors, planning, or lifecycle decisions into upstream code.
- Avoid change amplification. Do not introduce general frameworks, new GGML operations, scheduler behavior, backend ownership models, or execution-mode support solely to rearrange the cache interface.
- Preserve existing cache behavior unless a behavior change is explicitly requested. Do not silently change batching, scheduling, concurrency, residency policy, host fallback, or tensor-parallel behavior during an architectural refactor.
- Prefer a small interface that performs the routed operation over an interface that returns cache implementation pieces for callers to assemble.
- Keep graph-local route data separate from persistent cache residency state, and do not retain graph-owned tensors beyond their graph lifetime.

## Tests

- Add tests only when they verify observable behavior or a concrete regression with the same semantic value as the existing test suite.
- Do not add tests for call wiring, tensor names, private fields, mocks, implementation shape, or other details that can pass while behavior is wrong.
- If no valuable test is available, do not add a performative test.
- Obtain explicit approval before adding or materially expanding test scenarios for this work.
