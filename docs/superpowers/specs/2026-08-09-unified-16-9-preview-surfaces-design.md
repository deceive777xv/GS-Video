# Unified 16:9 Preview Surfaces Design

## Goal

Make every workflow media preview use a consistent 16:9 picture area:

- the subject representative-frame preview;
- the Gaussian scene preview;
- the composite-video or camera-reference preview.

The preview-page caption containing FOV and focus-distance information remains outside the 16:9 picture area.

## Current behavior

The three preview surfaces use viewport-height clamps and minimum heights. Their widths are controlled independently by page grids, so their displayed aspect ratios vary with window size. The preview page also applies its height contract to the complete card, which includes the caption.

## Design

Add one shared `preview-surface` class to the three elements that represent the actual picture area. The class will define a width-driven `aspect-ratio: 16 / 9`, remove fixed and minimum media heights, and align the surface to the top of its grid cell.

The existing semantic classes remain responsible for feature-specific behavior:

- `.subject-frame` keeps pointer interaction and the alpha overlay;
- `.viewport-frame` keeps camera gestures, picking, and its interaction hint;
- `.preview-media` keeps either the video player or camera-reference image;
- `.preview-card` keeps the caption below `.preview-media` and is not itself constrained to 16:9.

Images and videos continue to use `object-fit: contain`. Portrait or otherwise non-16:9 content therefore receives letterboxing without cropping or stretching. Loading and empty states fill the shared 16:9 surface.

The adjacent control panels retain their natural content height and align to the top. If a control panel is taller than its preview surface, the page uses its existing document scrolling; no nested control-panel scrolling is introduced.

## Responsive behavior

The same 16:9 contract applies at desktop, tablet, and narrow breakpoints. Existing one-column responsive grids remain unchanged. Width always determines preview height, so no breakpoint may reintroduce fixed media heights or minimum heights that override the ratio.

## Interaction safety

Subject and Gaussian picking continue to use the existing `toImagePoint` conversion. It calculates the contained image rectangle from the element bounds and source dimensions, rejects clicks in letterbox regions, and therefore remains valid when the outer surface changes to 16:9.

No backend requests, preview dimensions, media generation, video controls, project authority, or workflow state are changed.

## Alternatives considered

- Grouping the three existing CSS selectors would avoid JSX changes, but it would leave the shared contract implicit and make future preview surfaces easier to omit.
- Introducing a React preview component would provide a stronger abstraction, but it would combine unrelated interaction and rendering responsibilities for a CSS-only layout requirement.
- JavaScript measurement would duplicate native CSS sizing, add resize coordination, and offer no benefit over `aspect-ratio`.

The shared class is the smallest explicit contract.

## Verification

- At the default desktop viewport, measure `.subject-frame.preview-surface`, `.viewport-frame.preview-surface`, and `.preview-media.preview-surface`; each picture area must be 16:9 within one rendered pixel.
- Repeat at a wide or maximized viewport and at the existing `980px` and `700px` responsive boundaries.
- Confirm the preview caption is outside the measured media surface.
- Confirm portrait subject and composite media remain contained without cropping or stretching.
- Confirm loading and empty states fill the surface and the page has no horizontal overflow.
- Run the web test suite, TypeScript checks, the production web build, and `git diff --check`.
