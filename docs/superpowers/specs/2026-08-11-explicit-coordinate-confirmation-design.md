# Explicit Coordinate Confirmation and Visual Feedback

## Context

The subject-segmentation page and the Gaussian-scene camera page both accept
image-space pixel coordinates. Their current text fields do not reveal the valid
range or the corresponding location in the image. In addition, clicking the
subject representative frame immediately persists the prompt and starts the
segmentation stage, so an accidental click can begin expensive work.

The Gaussian foot-point flow already separates camera confirmation from camera
movement, but a viewport click immediately submits the depth pick. Both flows
need the same visible draft-and-confirm interaction without merging their
different business operations.

## Goals

- Make the valid X and Y ranges, origin, and axis directions visible.
- Keep clicking and direct numeric entry synchronized with a visible marker.
- Require an explicit confirmation before either segmentation or Gaussian
  foot-point picking performs backend work.
- Restore already-persisted points when a page is reopened.
- Fail closed when image, preview, or camera authority changes.
- Preserve the existing backend pixel-coordinate contracts and workflow
  ownership.

## Non-goals

- Replacing pixels with normalized or percentage coordinates.
- Adding drag-to-place behavior; dragging remains camera rotation in the
  Gaussian viewport.
- Changing segmentation, depth picking, preview rendering, or camera-solving
  algorithms.
- Adding persisted draft fields or new backend status fields.
- Automatically clamping, rounding, or otherwise correcting invalid input.

## Interaction Design

### Shared coordinate model

Coordinates use the source image's integer pixel grid:

- the origin is the top-left pixel;
- X increases to the right and is valid from `0` through `width - 1`;
- Y increases downward and is valid from `0` through `height - 1`.

Each coordinate editor displays these ranges using the actual representative
frame or authoritative preview dimensions. The inputs use numeric semantics,
`min`, `max`, and `step=1`, while still allowing an incomplete text-editing
state such as an empty field.

The page holds a local draft made of the two input strings and, only when both
strings describe valid in-range integers, a candidate point. An empty,
fractional, non-numeric, negative, or out-of-range value produces an inline
validation message, hides the candidate marker, and disables confirmation. The
application does not silently clamp or round it.

### Visual marker

A valid candidate is shown as a crosshair centered on the selected pixel, with
a compact `X n · Y n` label. The marker has `pointer-events: none` and therefore
does not interfere with picking or camera gestures. A text status distinguishes
`待确认` from `已确认` or `已验证`; color is supplementary rather than the only
status signal.

Marker placement uses the same contained-image rectangle as click conversion.
It accounts for letterboxing and responds to viewport resizing, so a portrait
representative frame inside a 16:9 surface does not place the marker relative to
the surrounding black bars.

### Subject segmentation

Loading the representative frame initializes the draft from
`workflow.subject_prompt` only when its `frame_index` matches that frame.
Otherwise the draft starts empty.

Clicking inside the representative frame only updates the local draft. Directly
editing either coordinate input updates the same draft and marker. Neither path
updates the project nor starts segmentation.

`确认人物位置并开始分割` is enabled only for a valid draft while the workflow is
idle. Activating it performs the existing ordered operation:

1. persist `subject_prompt` for the representative frame;
2. publish the returned project snapshot;
3. start the `segment` stage.

The control shows its submitting state and prevents duplicate confirmation. If
the project update succeeds but segmentation fails, the persisted prompt remains
available for an explicit retry. Existing Alpha output continues to represent
the last completed segmentation until a new segmentation succeeds; the draft
status makes clear when a newly selected point has not yet been applied.

### Gaussian foot point

The viewport keeps its existing pointer-drag threshold and click suppression.
A non-drag click inside the current authoritative image only updates the local
draft. Direct numeric input behaves identically.

The Gaussian draft authority is the tuple of preview artifact ID, camera
revision, and pick-buffer revision. A change to any member clears an uncommitted
draft. A persisted `workflow.foot_point` is restored only when all three members
match the current authoritative preview; otherwise no marker is shown.

`确认场景落脚点` is enabled only when:

- the draft is valid;
- the displayed frame matches the current camera;
- that camera revision and preview artifact are confirmed;
- no authoritative preview update is in progress.

Activating it calls the existing foot-point pick endpoint with the candidate
pixel and current authority tuple. Only a successful response changes the point
status to `已验证`. Camera movement immediately revokes confirmation in the
existing workflow and therefore disables picking; a replacement authoritative
preview also clears the old draft.

## Module Design

The coordinate seam lives under `apps/web/src/features/coordinates/` rather
than under the camera feature. This removes the current dependency in which the
subject page imports coordinate geometry from `scene-viewport.tsx`.

The shared module owns:

- contained-image rectangle calculation;
- viewport-event to source-pixel conversion;
- integer parsing and bounds validation;
- local draft synchronization and comparison with an initial persisted point;
- resize-aware marker placement;
- the consistent coordinate fields, range help, validation message, marker,
  and textual confirmation state.

Its interface accepts image dimensions, an authority identity, an optional
initial point, and controlled disabled/status labels. It returns or emits only a
valid candidate point. Callers do not need to reproduce parsing, range, dirty,
or marker-position logic.

The subject page continues to own project patching and starting segmentation.
The camera page and scene viewport continue to own camera gestures, preview
authority, camera confirmation, and the depth-pick request. The shared module
does not call the backend.

`CameraPage` passes the current confirmed camera revision and persisted foot
point into `SceneViewport`; this supplies the authority information required to
restore a valid point and to disable confirmation before the camera is
confirmed.

## State and Error Handling

- Client validation is inline and produces no backend request.
- Clicking letterbox space leaves the draft unchanged and reports that the
  click is outside image content.
- A subject representative-frame change resets or restores the draft according
  to matching `frame_index`.
- A Gaussian preview-authority change resets or restores the draft according to
  the full preview authority tuple.
- A backend stale-preview or unconfirmed-camera response is authoritative. The
  UI does not retry the old coordinate against a different frame; it refreshes
  project authority and requires a new valid confirmation.
- Existing global error handling remains responsible for transport and backend
  failures. A failed confirmation leaves the current valid local draft visible
  when its authority is still current, allowing the user to retry explicitly.

## Accessibility

- Coordinate fields remain the keyboard path and expose numeric bounds to
  assistive technology.
- Range/origin help and validation are text, not marker-only information.
- Validation and confirmation status use an appropriate live text region.
- The marker is decorative; the inputs contain the authoritative accessible
  value.
- Existing focus styling and Gaussian keyboard-accessible camera controls are
  preserved.

## Verification

Pure geometry and draft tests cover:

- contained-image mapping in both landscape and portrait letterboxed surfaces;
- click rejection in black bars;
- inverse marker placement at representative pixels and after resize;
- integer parsing, exact `0..width-1` / `0..height-1` bounds, and invalid input;
- initial-point restoration, dirty-state comparison, and authority reset.

Subject-page tests cover:

- a click updates fields and marker without calling `updateProject` or starting
  segmentation;
- direct input updates the marker;
- only the explicit button persists the prompt and starts segmentation;
- invalid drafts disable confirmation and issue no request;
- a matching persisted prompt is restored.

Scene-viewport tests cover:

- a click updates the draft without calling `pickFootPoint`;
- drag suppression still prevents a camera drag from becoming a candidate;
- direct input and the marker remain synchronized;
- the pick button remains disabled until the matching camera is confirmed;
- confirmation binds to the latest artifact and revision tuple;
- camera or preview authority changes clear an uncommitted draft;
- a matching persisted foot point is restored.

The relevant frontend test suite and production web build must pass. A real Edge
or equivalent Chromium run then checks both complete workflows at desktop and a
narrow responsive width, including click, manual entry, validation, explicit
confirmation, and marker alignment.

## Expected File Scope

- new shared coordinate module files under
  `apps/web/src/features/coordinates/`;
- `apps/web/src/features/subject/subject-page.tsx` and its tests;
- `apps/web/src/features/camera/camera-page.tsx`;
- `apps/web/src/features/camera/scene-viewport.tsx` and its tests;
- `apps/web/src/app/app.css` for marker, range, validation, and status styling;
- no backend schema, route, domain-model, or storage changes.
