# Camera Workbench Design QA

## Comparison target

- Source visual truth:
  - `C:/Users/31503/AppData/Local/Temp/codex-clipboard-978f05f4-a4ea-4e04-af3c-95d2d0a82dcd.png`
  - `C:/Users/31503/AppData/Local/Temp/codex-clipboard-4b83ce0f-b88d-4802-a6f9-d00528a32025.png`
  - `C:/Users/31503/AppData/Local/Temp/codex-clipboard-bdd0ceb5-9f5f-4c94-9917-ed4ed1af3acd.png`
  - `docs/superpowers/specs/2026-08-16-camera-workbench-ui-design.md`
- Browser-rendered implementation:
  - `output/playwright/camera-source-guides-1280x800.png`
  - `output/playwright/camera-scene-1600x900.png`
  - `output/playwright/camera-scene-2560x1440.png`
  - `output/playwright/camera-scene-2780x1440.png`
  - `output/playwright/camera-synthesis-1600x900.png`
  - `output/playwright/camera-source-centered-guides-1920x1010.png`
- Combined comparison evidence:
  - `output/playwright/camera-scene-reference-comparison.png`
  - `output/playwright/camera-scene-focused-comparison.png`
  - `output/playwright/camera-source-centering-typography-comparison.png`
- Browser: Microsoft Edge 151, headed Playwright CLI session.
- Theme/state: dark theme, project `test_1080`, source perspective saved as a visibly labeled low-confidence system prior, local ground not yet calibrated.

## Viewport and normalization

- Source scene screenshot: 2235 × 1446 physical pixels. The source device density and CSS viewport were not recorded, so it is treated as a visual product reference rather than a one-CSS-pixel fidelity contract.
- Full-view implementation capture: 2780 × 1440 physical pixels at a 2780 × 1440 CSS viewport, `deviceScaleFactor: 1`.
- Full-view comparison normalized the source proportionally to 1440 px high and placed it beside the 2780 × 1440 implementation without stretching.
- Focused comparison cropped the source's original scene workspace and the implementation's camera workbench, scaled both crops proportionally to 900 px high, then placed them side by side.
- Responsive captures were also checked at 1280 × 800, 1600 × 900, and 2560 × 1440.
- The later 3840 × 2160 user capture was cropped to its 3832 × 2020 application viewport and downsampled to 1920 × 1010, matching the Edge verification viewport before side-by-side comparison.

## Full-view comparison evidence

- The original single-page camera screen put every operation in one tall right rail. The implementation intentionally replaces this with three explicit subpages and a bounded, centered workbench.
- At 1600 × 900 and 2560 × 1440, the Gaussian viewport measured exactly 780.86 × 439.23 CSS px in both cases. Fullscreen therefore adds outer breathing room instead of scaling the internal workbench.
- At 1280 × 800 the workbench contracts to the available width without horizontal overflow; the workflow footer remains reachable and the page provides ordinary vertical scrolling for below-the-fold controls.
- The source, scene, and synthesis page containers preserve the existing dark/acid-green/cyan visual system and use the same header, workflow rail, borders, radii, and typography hierarchy as the surrounding product.

## Focused comparison evidence

- The 16:9 scene viewport remains the primary surface and the numeric control rail remains adjacent, but high-frequency movement moved onto the viewport as a compact game-style pad.
- Three-point ground actions moved below the viewport. This removes the tall button stack shown in the source while keeping direct X/Y/Z/Yaw/Pitch/Roll and FOV input visible.
- The source-perspective page replaces FOV/horizon sliders with two colored guide groups over the image, circular drag handles, keyboard adjustment, read-only derived FOV/Roll/gravity, and an explicit low-confidence fallback state.
- The portrait source frame is contained inside a 16:9 black presentation surface without cropping or stretching; the Gaussian image remains sharp and uses the authoritative project preview asset.

## Required fidelity surfaces

- Fonts and typography: the implementation retains the product's existing display/body/monospace stack and compact uppercase labels. The source calibration rail uses a deliberately quieter auxiliary hierarchy: labels measure 10.88 px at the 1920 × 1010 CSS viewport, weight 500, with muted slate foregrounds; no clipping or unintended wrapping was observed.
- Spacing and layout rhythm: the three-page navigation, 16:9 viewport/control pairing, bottom secondary actions, and fixed maximum workbench size create a consistent scan path. Cards, gaps, radii, and borders match existing product tokens.
- Colors and visual tokens: acid green remains the primary/active state, cyan identifies the second guide group and connected status, and muted slate surfaces preserve contrast. Disabled and selected states are distinct.
- Image quality and asset fidelity: real project source/alpha/Gaussian assets are used. Images are contained rather than stretched, and no placeholder illustration, handcrafted SVG replacement, or CSS-art product asset was introduced.
- Copy and content: labels explain that FOV is solved rather than estimated by the user, low-confidence priors are explicit, hidden feet can be skipped in pure-perspective mode, and perspective consistency is not presented as physical contact.
- Accessibility and interaction: subpages use links, fields are labeled, numeric values remain directly editable, guide handles expose semantic button names and focus rings, and guide endpoints support 5 px arrows / 1 px Shift+arrows.

## Primary interactions tested

- Navigated source, GS scene, and synthesis subpages through real routes.
- Selected orthogonal-guide evidence and adjusted a guide endpoint with Shift+ArrowRight.
- Verified the guide endpoint kept focus and the evidence method did not change.
- Verified the initial example guides cannot be confirmed until an endpoint is adjusted.
- Used the on-canvas movement pad and confirmed the camera pose changed.
- Entered Roll `12.5` directly and committed it with Enter.
- Scrolled the mouse wheel over the Gaussian viewport: camera Z changed from `-4.25` to `-4.5`, while `window.scrollY` remained `0`.
- Verified 16:9 dimensions and identical scene viewport size at 1600 × 900 and 2560 × 1440.
- Checked browser console: 0 errors. One transient WebSocket-close warning came from the first interrupted connection attempt; the verified page subsequently showed `实时事件已连接`.

## Findings and comparison history

### Iteration 1

- [P1] Clicking a vanishing-line endpoint did not retain keyboard focus.
  - Evidence: after clicking an endpoint, Shift+ArrowRight operated the still-focused evidence-method radio group and selected the low-confidence prior.
  - Impact: keyboard fine adjustment could silently switch calibration modes.
  - Fix: explicitly focus the guide handle during pointer-down before capturing the pointer.
  - Post-fix evidence: Edge snapshot marked `A 组第 1 条线端点 1` active; Shift+ArrowRight retained `两组正交方向`, enabled confirmation, and updated the derived solution from 60.4° to 59.9°.

- [P2] Very nearly parallel same-group guides could create an unstable far-away vanishing point.
  - Evidence: the browser and backend used an effectively exact-parallel threshold.
  - Impact: small pointer noise could yield numerically unstable perspective evidence.
  - Fix: normalize image lines and reject intersections with normalized determinant at or below `1e-4` in both frontend preview and authoritative backend solve; added regression tests.
  - Post-fix evidence: frontend and backend near-parallel regression tests pass.

### Iteration 2

- No remaining actionable P0, P1, or P2 visual, interaction, responsive, typography, spacing, color, image-quality, or copy findings were observed in the revised Edge captures.

### Iteration 3

- [P2] The 4K/high-density user capture made the camera workbench appear shifted toward the right side of the usable canvas, while inherited global control typography made the source-calibration rail too bright and visually heavy.
  - Evidence: `camera-source-centering-typography-comparison.png` places the normalized user capture beside the revised Edge render at the same 1920 × 1010 application size.
  - Fix: give the workbench an explicit `min(100%, 1120px)` width, auto inline margins, and centered grid self-alignment; add camera-scoped compact type, input, button, border, surface, and muted foreground rules for the right rail.
  - Post-fix evidence: at 1920 × 1010 the workbench center differs from the usable `.workflow-main` center by `0.0000076` CSS px, `scrollWidth` equals `clientWidth`, the source grid remains 1086.86 px wide, and the right rail remains 300 px wide. Edge reports 0 console errors.

- No remaining actionable P0, P1, or P2 visual, responsive, typography, spacing, or color findings were observed after the Iteration 3 side-by-side comparison.

## Residual test gaps

- The local project used for browser QA had no confirmed three-point ground, so the synthesis page's prerequisite state was captured rather than a fully rendered composite. The complete synthesis controls and interaction path remain covered by frontend integration tests.
- The source reference screenshots predate the approved three-subpage design and have unknown display density; exact pixel matching against their single-page scale is therefore not meaningful. The focused comparison checks hierarchy and product continuity instead.

## Implementation checklist

- [x] Split the camera workflow into source, scene, and synthesis routes.
- [x] Keep every relevant viewport at 16:9.
- [x] Keep internal workbench dimensions constant between default and fullscreen desktop sizes.
- [x] Put six-direction movement controls on the scene viewport.
- [x] Isolate Gaussian wheel input from page scrolling.
- [x] Move low-frequency ground actions below the viewport.
- [x] Replace user-estimated FOV with direct guide evidence and derived readouts.
- [x] Retain direct numeric X/Y/Z/Yaw/Pitch/Roll and FOV input.
- [x] Verify responsive layouts, keyboard behavior, production build, and automated regressions.

final result: passed
