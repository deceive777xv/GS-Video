import { describe, expect, it } from 'vitest'

import {
  solvePerspectiveGuides,
  type GuideGroup,
  type GuideGroups,
} from './perspective-guides'

function guidePair(vanishing: [number, number]): GuideGroup {
  const segment = (start: [number, number]): GuideGroup[0] => [
    { x: start[0], y: start[1] },
    { x: start[0] + .28 * (vanishing[0] - start[0]), y: start[1] + .28 * (vanishing[1] - start[1]) },
  ]
  return [segment([220, 260]), segment([420, 880])]
}

describe('solvePerspectiveGuides', () => {
  it('recovers FOV from two orthogonal horizontal-plane directions', () => {
    const width = 1920; const height = 1080
    const focal = .5 * height / Math.tan(Math.PI / 6)
    const groups: GuideGroups = [
      guidePair([width / 2 + focal, height / 2]),
      guidePair([width / 2 - focal, height / 2]),
    ]

    const solved = solvePerspectiveGuides({ width, height }, groups, 'both_horizontal_plane')

    expect(solved.valid).toBe(true)
    expect(solved.fov).toBeCloseTo(60, 6)
    expect(solved.gravity?.[1]).toBeCloseTo(-1, 6)
  })

  it('rejects parallel guide pairs instead of returning a guessed calibration', () => {
    const parallel: GuideGroup = [
      [{ x: 100, y: 50 }, { x: 300, y: 100 }],
      [{ x: 100, y: 150 }, { x: 300, y: 200 }],
    ]

    const solved = solvePerspectiveGuides(
      { width: 640, height: 360 },
      [parallel, parallel],
      'a_vertical_b_horizontal',
    )

    expect(solved.valid).toBe(false)
    expect(solved.message).toMatch(/平行/)
  })

  it('rejects nearly parallel guide pairs before they create unstable vanishing points', () => {
    const nearlyParallel: GuideGroup = [
      [{ x: 100, y: 100 }, { x: 500, y: 100 }],
      [{ x: 100, y: 200 }, { x: 500, y: 200.01 }],
    ]

    const solved = solvePerspectiveGuides(
      { width: 640, height: 360 },
      [nearlyParallel, nearlyParallel],
      'a_vertical_b_horizontal',
    )

    expect(solved.valid).toBe(false)
    expect(solved.message).toMatch(/平行/)
  })
})
