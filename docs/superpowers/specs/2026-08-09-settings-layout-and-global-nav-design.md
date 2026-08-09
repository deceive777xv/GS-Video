# Settings Layout and Global Navigation Design

## Goal

Keep the settings-page cards at the same width in the default desktop window and in a maximized window, and place the global `项目 / 素材库 / 设置` navigation on the left in every application view.

## Scope

This is a presentation-only change in `apps/web/src/app/app.css`. It does not change routing, React component structure, storage behavior, or responsive navigation behavior on narrow screens.

## Layout design

The settings heading, two-card grid, notice, restart card, and action card share one named content-width constraint. Its maximum inline size is `1120px`, matching the current default desktop window shown in the approved reference. At wider viewport sizes the content remains centered and only the surrounding whitespace grows. Below that available width, the content continues to shrink responsively; the existing single-column breakpoint remains intact.

The hub header uses the same desktop column geometry as the workflow header:

1. Brand
2. Global navigation
3. Flexible space
4. Current project

The global navigation therefore begins immediately after the brand, as it does inside a project. The current-project label remains aligned to the right. Existing narrow-window rules may collapse the header and hide the project label when space is constrained.

## Alternatives considered

- Extracting a shared React header component would remove structural duplication, but it expands the change beyond the requested visual correction.
- A fullscreen-only media query would depend on viewport thresholds and would be fragile under Windows display scaling.

The CSS-only shared geometry is the smallest stable change.

## Verification

- Run the web typecheck, focused web tests, and production web build.
- In Microsoft Edge, measure the settings content at the default `1280px`-class viewport and at a maximized/wide viewport. The two-card grid must remain `1120px` wide within a one-pixel tolerance when both viewports have sufficient room.
- In Edge, compare the global navigation position on a hub view and a workflow view. Both must start in the header column immediately after the brand.
- Confirm a narrow viewport still uses the existing responsive layout without horizontal overflow.
