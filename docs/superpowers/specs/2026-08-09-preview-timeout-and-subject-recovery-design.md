# Preview workspace, subject overlay, and task-error recovery

## Problem

Three user-visible failures currently overlap:

1. A project whose Gaussian scene is a shared asset cannot open the camera preview after project artifacts were moved to the configurable cache library. The scene is valid, but the persistent preview session still requires `<project_root>/previews` even though project directories no longer own cache output.
2. The subject page performs a single Alpha-overlay request. If that request races stage publication or fails with a retryable local transport error, the page remains without an overlay until it is unmounted and opened again.
3. A deterministic unsupported-material camera failure can be visually hidden by an unrelated transport-timeout banner. The user sees `180/758` and a timeout instead of the authoritative camera-solver failure.

The existing `38aa2bf` task-admission fix remains authoritative. It must be verified from a newly started backend; this design does not replace it with longer global HTTP timeouts.

## Considered approaches

### A. Recreate `previews/` inside every project

This is the smallest code change, but it violates the approved storage boundary: project configuration and shared inputs live in the project library, while generated and transient artifacts live in the cache library. Rejected.

### B. Treat the cache root as the existing `project_root`

The route could resolve every scene to an absolute path and pass the cache project root through parameters still named `project_root`. This is mechanically small but hides two different authorities behind one argument and makes legacy relative-scene handling fragile. Rejected.

### C. Separate scene authority from preview workspace

Keep `project_root` for project-owned relative input resolution and pass a distinct cache-owned preview workspace to the preview service/session. This is the recommended approach because the interface states both ownership boundaries explicitly and supports shared absolute PLY paths and legacy project-relative PLY paths.

## Design

### Preview workspace

- Preview service and persistent-session calls receive a separate preview workspace root.
- With an `ArtifactStore`, the workspace is `artifact_store.project_root(project_id)`. Without one, tests and legacy single-project assembly fall back to the existing project root.
- The persistent preview session creates and validates only its owned `previews/` child under that workspace. It continues to resolve relative scene inputs against the real project root.
- Shared-asset resolution, size/SHA-256 authority, reparse-point checks, single-link checks, latest-only scheduling, and suspend/drain/resume behavior remain unchanged.
- No cache directory is created inside a managed project merely to satisfy rendering.

### Subject Alpha recovery

- Proxy and Alpha media loads use the same bounded retry policy.
- Retry `subject_media_not_ready` and errors explicitly marked retryable, including local request timeouts and network interruptions.
- Use the existing short delays (250 ms, 500 ms, 1,000 ms); after exhaustion, surface the final error once.
- Key Alpha loading by the segment cache key and scalar prompt coordinates/frame index so a newly authoritative artifact revision triggers a fresh request.
- Abort timers and requests on project/page change, and continue revoking replaced object URLs.

### Authoritative camera error

- When the active task reaches `failed`, prefer its authoritative task-event error over an older request-level transport banner for the same workflow period.
- Map `unsupported_material` on `solve_camera` to a clear user-facing explanation that the camera trajectory is not trustworthy for this material. Do not claim that resolution is the cause.
- Preserve progress (`180/758`) as diagnostic context and keep retry behavior driven by the backend's `retryable` flag.

## Public verification seams

1. API/runtime seam: with separate project and cache roots, a shared-PLY project can request a live/authoritative preview while the project directory has no `previews/` child; preview work appears only under the cache project root.
2. React subject seam: when the first Alpha request is not ready or retryable, the overlay appears after bounded retry or after a new segment cache key becomes authoritative, without remounting the page.
3. React workflow seam: a failed `solve_camera` event with `unsupported_material` displays the material error rather than an older transport timeout.

Each slice will be implemented red first, then minimally made green. Focused Python/Web tests run after each slice, followed by TypeScript, Ruff, Mypy, the non-GPU Python suite, the Web suite/build, and a fresh-process desktop check.

## Non-goals

- Changing camera-solver thresholds or downscaling policy.
- Increasing the global 15-second HTTP timeout.
- Weakening scene identity, reparse-point, or file-link validation.
- Moving shared source video or PLY files back into project directories.
