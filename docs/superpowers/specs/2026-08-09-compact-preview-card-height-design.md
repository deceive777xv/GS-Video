# Compact Preview Card Height Design

## Goal

In the default desktop window, make the complete preview card—including the video region and its bottom caption—the same height as the subject-page preview frame. Preserve the current maximized-window appearance.

## Root cause

The subject frame uses `clamp(360px, calc(100vh - 360px), 560px)`, while the desktop preview card uses `clamp(320px, calc(100vh - 440px), 560px)`. The different viewport deductions make the preview card approximately `80px` shorter in the default window. In a maximized window, both reach the shared `560px` maximum, which hides the mismatch.

## Design

The subject frame and complete preview card will share the same height contract at each responsive range:

- Desktop: `clamp(360px, calc(100vh - 360px), 560px)`
- At or below `980px`: `clamp(360px, calc(100vh - 360px), 520px)` with a `360px` minimum
- At or below `700px`: retain the existing shared `clamp(280px, calc(100vh - 420px), 420px)` rule and `280px` minimum

The preview card keeps its two rows: a flexible media region and the caption row. The video itself remains `object-fit: contain`, so no cropping or stretching is introduced. React components, video loading, controls, and backend behavior are unchanged.

## Alternatives considered

- Giving only `.preview-media` the subject-frame height would make the complete card taller than the subject frame and would not match the approved maximized-window behavior.
- A fixed `aspect-ratio` would ignore available viewport height and could force unnecessary scrolling or cropping.

Sharing the complete card's height rule is the smallest consistent change.

## Verification

- In Microsoft Edge at the default desktop window size, measure `.subject-frame` and `.preview-card`; their heights must match within one pixel.
- Repeat at a wide/maximized viewport; both must remain at the current `560px` cap.
- At `980px` and `700px` responsive widths, confirm the paired height rules remain equal and the page has no horizontal overflow.
- Run the web test suite, TypeScript typecheck, production web build, and `git diff --check`.
