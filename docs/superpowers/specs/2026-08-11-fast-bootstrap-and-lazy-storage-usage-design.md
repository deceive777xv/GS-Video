# Fast bootstrap and lazy storage usage

## Problem

Both desktop and browser hosts can start the local service correctly and still report that it is unavailable. A measured authenticated bootstrap returned HTTP 200 in 16.4-18.6 seconds, after the Web client had already enforced its 15-second request timeout.

The dominant synchronous costs are:

- `StorageLayoutManager.snapshot()` recursively inventories the 640 MB project library and 2.99 GB cache library on every bootstrap; the current cache takes about 12.1 seconds to scan.
- The environment doctor probes the renderer and EdgeTAM worker during bootstrap; the EdgeTAM probe alone takes about 5.5 seconds.

The connection screen currently catches every bootstrap error and reduces it to a generic connection failure, so a slow or failed initialization looks like a bad port or token.

## Considered approaches

### A. Increase the global HTTP timeout

Raising 15 seconds to 30 or 60 seconds is mechanically small, but storage scanning grows with the cache and would eventually cross the new threshold. It would also make unrelated stalled requests take longer to fail. Rejected as the primary fix.

### B. Cache recursively computed byte totals in the storage manager

Bootstrap could return cached values and invalidate them after every artifact, asset, migration, and cleanup mutation. This keeps the existing response shape, but correct invalidation spans many writers and stale values would be difficult to explain. Rejected for now.

### C. Separate quick status from measured usage

Bootstrap returns only constant-time storage topology and editability. Opening Settings automatically requests the existing full storage snapshot, which performs the recursive measurement there. Environment probing is cached separately and can be explicitly refreshed. This is recommended because it removes work from the connection boundary and gives expensive operations an explicit owner.

## Design

### Storage contracts

- Add a lightweight storage status DTO containing project/cache roots and IDs, restart-required state, editability, and blocked reason.
- Keep the full storage snapshot DTO for the storage endpoint and storage mutations. It extends the status with project-library bytes/free bytes and cache bytes/free bytes.
- Add `StorageLayoutManager.status()` as a constant-time operation. It validates no directory inventory and performs no recursive traversal.
- Bootstrap returns `status()`. It must not call `snapshot()`.
- `GET /api/v1/runtime/storage-layout` remains the authoritative full measurement endpoint.
- Storage switch and cleanup responses continue returning measured snapshots because those are explicit settings operations rather than application connection.

### Settings flow

- Bootstrap storage status is enough to decide whether the Settings feature exists and whether editing is blocked.
- On navigation to Settings, the Web app automatically calls `getStorageLayout()` once for that page visit.
- While it runs, the capacity area displays `正在统计…`; other application pages remain usable.
- A failed measurement stays local to Settings, displays the backend error, and offers a retry. It never returns the user to the connection screen.
- Successful storage mutations replace both the measured settings state and the lightweight bootstrap status.

### Environment flow

- Introduce a process-local environment report cache around the existing doctor. The first bootstrap may perform the probe and cache its result; later bootstrap/catalog refreshes reuse it.
- Add an explicit environment refresh API used by the existing “重新检测环境” flow. Refresh replaces the cached report only after the doctor completes successfully as a call; an unhealthy but valid report is still cacheable.
- Environment repair completion must use the explicit refresh path, so cached pre-repair results cannot survive a requested recheck.
- This design does not weaken GPU admission, worker probing, or environment readiness checks.

### Client timeout and errors

- Keep the 15-second default for ordinary REST requests.
- Give only initial bootstrap a 30-second timeout as a bounded safety margin for the first environment probe on slower machines.
- Desktop and browser composition roots preserve the structured bootstrap error. They show the backend message and code when available, while authentication/network failures retain concise connection guidance.
- Session tokens remain memory-only and are never included in errors or logs.

## Verification

1. API regression: bootstrap succeeds when a storage double fails the test if `snapshot()` is called, and returns lightweight status.
2. API regression: the full storage endpoint still calls `snapshot()` and returns measured bytes.
3. API regression: repeated bootstrap calls reuse one environment report; explicit refresh invokes the doctor again.
4. Web regression: entering Settings renders a loading state, requests the measured snapshot automatically, and then renders capacities.
5. Web regression: a failed settings measurement remains on Settings and can retry.
6. Web regression: bootstrap uses its dedicated timeout and surfaces a structured failure instead of only `Desktop local service unavailable.`
7. Focused Python and Web tests run red before implementation and green afterward.
8. Final verification includes Ruff, Mypy, the non-GPU Python suite, the Web suite, typecheck, build, and real fresh-process bootstrap timing against the current 2.99 GB cache.

## Non-goals

- Changing project/cache roots or migration semantics.
- Deleting or rewriting existing cache artifacts.
- Adding persistent telemetry or storing session tokens.
- Weakening path, reparse-point, hard-link, or storage marker validation.
- Making task execution endpoints inherit the longer bootstrap timeout.
