# Task 14 Frontend Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the ten independent-review gaps in the Task 14 React workflow without assembling real `WorkflowServices` or inventing a composite-video artifact.

**Architecture:** Keep `App` as the single task/project authority and pass one shared busy/admission state into feature pages. Bind preview interactions to an immutable camera fingerprint, isolate resumable upload metadata behind a session-scoped store, and expose only explicit user-initiated browser saves. Every asynchronous effect owns an abort flag/timer cleanup so React StrictMode and stale responses cannot mutate newer state.

**Tech Stack:** React 19, TypeScript, Vitest 4, Testing Library, Web Crypto primitives, browser `sessionStorage`, Vite, Playwright CLI.

## Global Constraints

- Preserve the existing `.gitignore` modification.
- Do not implement real `WorkflowServices` or a composite-preview artifact.
- Do not persist tokens, local paths, file bytes, or other secrets.
- Keep platform-specific behavior behind `PlatformBridge`; feature modules must not import Tauri APIs.
- Use test-first RED/GREEN cycles and project-local caches only.

---

### Task 1: Task ownership recovery and event replacement

**Files:**
- Modify: `apps/web/src/app/app.tsx`
- Modify: `apps/web/src/api/task-events.ts`
- Test: `apps/web/src/features/workflow/guided-workflow.test.tsx`
- Test: `apps/web/src/api/task-events.test.ts`

**Interfaces:**
- Consumes: `ProjectDto.workflow.active_task_id`, `TaskEvent`, `TaskStore.replaceFromRest`.
- Produces: bounded initial recovery retries with cleanup and newest-event task ownership.

- [ ] Add tests where the initial `getTask` fails once then succeeds after controlled timer advancement, unmount prevents later retries, and the recovered marker is written only after success.
- [ ] Add an old-task/new-task event-gap test proving a newer event task ID replaces the stale task before REST resync and that App never queries the stale owner afterward.
- [ ] Run the focused tests and confirm failures identify the premature recovery marker and stale-owner selection.
- [ ] Implement a bounded retry schedule (`250`, `500`, `1000` ms), clear timers/abort on cleanup, and prefer `task_event.task_id` for replacement/resync authority.
- [ ] Re-run the focused tests until green.

### Task 2: Preview camera authority and shared admission

**Files:**
- Modify: `apps/web/src/features/camera/scene-viewport.tsx`
- Modify: `apps/web/src/features/camera/scene-viewport.test.tsx`
- Modify: `apps/web/src/app/app.tsx`
- Modify: `apps/web/src/features/import/import-page.tsx`
- Modify: `apps/web/src/features/subject/subject-page.tsx`
- Modify: `apps/web/src/features/preview/preview-page.tsx`
- Modify: `apps/web/src/features/export/export-page.tsx`
- Modify: `apps/web/src/features/workflow/guided-workflow.test.tsx`

**Interfaces:**
- Produces: `cameraFingerprint(CameraInput): string`, `busy: boolean`, and a central `runStage` guard rejecting any queued/running owner.

- [ ] Add viewport tests showing any camera mutation immediately disables both confirm and pick, submit guards reject stale frames, cumulative sub-threshold pointer moves suppress the click, and null preview authority revokes/clears the frame URL.
- [ ] Add admission tests for double click and a different active owner disabling ingest, segment, composite, and export controls.
- [ ] Run focused tests and record expected failures.
- [ ] Store the fingerprint that produced each accepted frame; compare it at render and submit time, clear stale frame authority on camera mutation/null preview, and measure drag displacement from pointer-down origin.
- [ ] Compute shared busy state in `App`, reject from `runStage`, and pass `busy` to every task-starting page/control.
- [ ] Re-run focused tests until green.

### Task 3: Incremental hashing and resumable uploads

**Files:**
- Create: `apps/web/src/features/import/incremental-sha256.ts`
- Create: `apps/web/src/features/import/incremental-sha256.test.ts`
- Create: `apps/web/src/features/import/upload-resume.ts`
- Modify: `apps/web/src/features/import/import-page.tsx`
- Test: `apps/web/src/features/workflow/guided-workflow.test.tsx`

**Interfaces:**
- Produces: `sha256Hex(blob, chunkSize?)`, fixed maximum chunk reads, and non-secret `UploadResumeRecord` session persistence.

- [ ] Add SHA-256 known-vector, chunk-boundary, bounded-slice and forbidden-whole-`arrayBuffer` tests.
- [ ] Add reload/reselection tests proving a matching server session resumes missing chunks, mismatch is cancelled and cleared before creation, unmount only aborts the request, and explicit cancel removes both server and session metadata.
- [ ] Run focused tests and record expected failures.
- [ ] Implement an incremental SHA-256 state machine fed from bounded Blob slices/stream chunks and preserve the exact lowercase digest contract.
- [ ] Persist only upload ID, kind, name, MIME type, byte size, digest and chunk size under a project-scoped `sessionStorage` key; validate `getUpload` before resume and clear mismatch before `createUpload`.
- [ ] Re-run focused tests until green.

### Task 4: Progress, recovery actions, connection card and explicit save

**Files:**
- Modify: `apps/web/src/app/app.tsx`
- Modify: `apps/web/src/app/app.css`
- Modify: `apps/web/src/composition-root.tsx`
- Modify: `apps/web/src/features/preview/preview-page.tsx`
- Modify: `apps/web/src/features/export/export-page.tsx`
- Modify: `apps/web/src/features/export/export-page.test.tsx`
- Modify: `apps/web/src/features/workflow/guided-workflow.test.tsx`
- Modify: `apps/web/src/platform/platform-boundary.test.ts`

**Interfaces:**
- Consumes: latest `task_event` stage/progress and `TaskDto.error` code/category/retryability.
- Produces: determinate accessible progress when event data exists, targeted recovery controls, styled browser connection card, and explicit browser save after verification.

- [ ] Add tests for determinate `progressbar` event rendering, indeterminate REST state, subject/render/camera recovery selection, retry eligibility, responsive connection-card semantics, no token persistence, and browser export that never saves until the explicit button is clicked.
- [ ] Run focused tests and record expected failures.
- [ ] Render progress from the latest matching event only; otherwise expose status text without a fabricated percentage.
- [ ] Derive recovery actions from target stage plus stable error code/category and show retry only for retryable failed tasks.
- [ ] Add centered responsive connection-card classes/help/error copy and remove unjustified `role="application"`.
- [ ] Separate export task completion from browser saving; reveal `保存已验证视频` after metadata verification while retaining Tauri native save flow.
- [ ] Re-run focused tests until green.

### Task 5: Verification, visual QA, records and commit

**Files:**
- Modify: `.superpowers/sdd/task-14-report.md` (ignored ledger)
- Modify: `.superpowers/sdd/progress.md` (ignored ledger)

- [ ] Run all focused tests touched by this fix.
- [ ] Run full `npm run test:web -- --run --reporter=dot`, `npm run typecheck:web`, and `npm run build:web` when the environment permits; report `spawn EPERM` without bypassing the usage limit.
- [ ] Run source/bundle secret, fixed-port, storage and Tauri-boundary scans plus `git diff --check`.
- [ ] Serve the built UI with a disposable local API, inspect desktop and narrow Edge layouts, and store screenshots only under ignored `output/playwright/`.
- [ ] Update the ignored Task 14 report/ledger with exact RED/GREEN evidence.
- [ ] Stage only intended tracked files, preserve `.gitignore`, and commit with a focused hardening message.
