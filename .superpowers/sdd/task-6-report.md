# Task 6 Report: Gaussian PLY scene model and orbit camera

## Status

Implemented strict Graphdeco Gaussian PLY loading, conservative scene VRAM sizing and admission,
and orbit-camera transforms/intrinsics with one documented OpenCV-style coordinate convention.

## RED / GREEN evidence

### Cycle 1: Gaussian PLY, VRAM, and camera contracts

- RED: `uv run --extra dev python -m pytest -p no:cacheprovider tests/unit/scene -v`
  failed during collection with `ModuleNotFoundError: No module named 'gs_video.scene'` for both
  scene test modules.
- The first GREEN run executed 32 tests. Production behavior passed, while three test assertions
  errored because `pytest.approx` does not accept nested lists. Replacing those assertion helpers
  with `numpy.testing.assert_allclose` preserved the intended assertions.
- GREEN: the focused scene suite passed `32 passed`.

### Cycle 2: malformed PLY syntax and non-scalar properties

- RED: the focused PLY suite had two expected failures: a list-valued `opacity` leaked a NumPy
  `ValueError`, and an invalid header leaked `plyfile.PlyHeaderParseError`.
- GREEN: scalar numeric dtype validation and `PlyParseError` mapping made both cases stable
  `UnsupportedMaterialError` failures; the expanded focused suite passed `34 passed`.

## API and documentation choices

- Context7 selected NumPy `/websites/numpy_doc_2_4` and plyfile
  `/dranjan/python-plyfile` as the current primary documentation.
- `PlyData.read(path)` handles ASCII and binary files. Vertex field discovery uses structured
  `dtype.names`, and every required/SH field must have a scalar integer or floating dtype.
- `GaussianScene.colors` is always `[N, K, 3]`. DC is inserted as coefficient zero. Continuous
  `f_rest_0..M` is read in Graphdeco channel-major order, reshaped to `[N, 3, K-1]`, and
  transposed to `[N, K-1, 3]`. Supported square SH bases are `K = 1, 4, 9, 16`.
- Means, raw log-scales, quaternions, raw opacity logits, and SH colors are finite, contiguous
  float32 arrays. The loader deliberately applies neither `exp` nor `sigmoid`.
- `estimate_scene_vram` preserves the brief's exact integer formula: tensor bytes plus RGBA
  float framebuffer bytes plus 48 projection bytes per Gaussian, multiplied by 1.5.
  `assess_scene_vram` applies the exact rational comparison `estimate * 5 <= available * 4` and
  recommends `downsample the scene or reduce resolution` on rejection.
- `OrbitCamera` stores the domain camera-to-world convention. Its rotation columns are camera
  right, down, and forward axes in world coordinates. `view_matrix()` returns the inverse in
  OpenCV x-right/y-down/z-forward coordinates. At zero yaw/pitch, the camera is at `(0, 0,
  -distance)`, looking along +z. Intrinsics use centered principal point and one focal length
  derived from vertical FOV.

## Fixture

- Path: `tests/fixtures/scene/tiny_gaussians.ply`
- Format: hand-authored ASCII PLY, two synthetic vertices, Graphdeco-required fields, degree-1 SH
  (`K = 4`), deliberately distinct raw negative/positive log-scales and opacity logits.
- Size: 749 bytes.
- SHA-256: `2F94B30DDAD6FF2644A75B35B4DC2485A479CCCFCF28B8EE9D1C4DF85BD8C1C4`.
- Tests generate additional tiny ASCII and binary PLYs through plyfile. No test reads or extracts
  the 14 GB official Graphdeco acceptance archive.

## Test coverage

- Plain XYZ rejection with every missing required property listed, including `opacity`.
- Missing and empty vertex elements; malformed PLY headers; non-finite values; list-valued fields.
- ASCII and binary Graphdeco PLY loading.
- Continuous `f_rest` enforcement, counts divisible by three, and supported square SH bases.
- Graphdeco channel-major SH mapping into `[N, K, 3]`.
- Raw log-scale/logit preservation, tensor shapes, float32 dtypes, and contiguous layout.
- Exact VRAM formula, exact 80% acceptance boundary, and actionable rejection recommendation.
- Camera-to-world/view inverse consistency, orthonormal right-handed rotation, OpenCV zero orbit,
  centered principal point, vertical-FOV focal behavior, and invalid distances/FOVs/dimensions.

## Verification

- Focused: `uv run --extra dev python -m pytest -p no:cacheprovider tests/unit/scene -q`:
  `34 passed in 0.19s`.
- Cumulative: `uv run --extra dev python -m pytest -p no:cacheprovider tests/unit
  tests/integration`: `135 passed in 0.83s`.
- `uv run --extra dev python -m ruff check src tests`: `All checks passed!`.
- `uv run --extra dev python -m mypy src/gs_video/scene`:
  `Success: no issues found in 3 source files`.
- `git diff --check`: clean.

## Files

- `src/gs_video/scene/__init__.py`
- `src/gs_video/scene/ply.py`
- `src/gs_video/scene/camera.py`
- `tests/unit/scene/test_ply.py`
- `tests/unit/scene/test_camera.py`
- `tests/fixtures/scene/tiny_gaussians.ply`
- `.superpowers/sdd/task-6-report.md`

## Self-review

- Confirmed the missing-field error is deterministically sorted and includes all absent names.
- Confirmed SH reshaping follows Graphdeco's channel-major storage rather than treating the flat
  properties as coefficient-major.
- Confirmed all returned tensors are float32 and C-contiguous after transpose/concatenation.
- Confirmed scale and opacity values remain byte-for-byte-equivalent float32 values, without
  renderer activation functions.
- Confirmed 80% admission uses integer arithmetic and accepts equality without rounding drift.
- Confirmed the camera basis has determinant +1 and the view transform is a true inverse.
- Confirmed tests only use the committed tiny fixture and per-test generated files.

## Concerns

- The loader intentionally supports SH degrees 0 through 3 (`K = 1, 4, 9, 16`), matching common
  Graphdeco exports. Higher square bases are rejected until the renderer contract explicitly
  supports them.
- VRAM sizing is a conservative admission estimate, not a driver-level allocation measurement;
  allocator fragmentation and backend-specific temporary buffers can still require headroom.
