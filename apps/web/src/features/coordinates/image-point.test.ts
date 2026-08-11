import { describe, expect, it } from 'vitest'

import {
  getContainedImageRect,
  imagePointToViewport,
  toImagePoint,
  validateImagePoint,
} from './image-point'

describe('image point geometry', () => {
  it('maps clicks and markers through the same portrait letterbox rectangle', () => {
    const bounds = { left: 10, top: 20, width: 500, height: 300 }
    const image = { width: 200, height: 400 }
    expect(getContainedImageRect(bounds, image)).toMatchObject({
      left: 185, top: 20, width: 150, height: 300, scale: 0.75,
    })
    expect(toImagePoint({ clientX: 260, clientY: 170 }, bounds, image)).toEqual({ x: 100, y: 200 })
    expect(imagePointToViewport({ x: 100, y: 200 }, bounds, image)).toEqual({ x: 250.375, y: 150.375 })
  })

  it('rejects clicks in letterbox bars', () => {
    expect(toImagePoint(
      { clientX: 50, clientY: 150 },
      { left: 0, top: 0, width: 500, height: 300 },
      { width: 400, height: 400 },
    )).toBeNull()
  })

  it('maps landscape images through horizontal letterboxing and preserves edge pixels', () => {
    const bounds = { left: 5, top: 10, width: 300, height: 500 }
    const image = { width: 400, height: 200 }
    expect(getContainedImageRect(bounds, image)).toMatchObject({
      left: 5, top: 185, width: 300, height: 150, scale: 0.75,
    })
    expect(toImagePoint({ clientX: 304.9, clientY: 334.9 }, bounds, image)).toEqual({ x: 399, y: 199 })
    expect(imagePointToViewport({ x: 399, y: 199 }, bounds, image)).toEqual({ x: 299.625, y: 324.625 })
  })
})

describe('image point validation', () => {
  const image = { width: 960, height: 540 }

  it('accepts exact integer image bounds', () => {
    expect(validateImagePoint('0', '0', image).point).toEqual({ x: 0, y: 0 })
    expect(validateImagePoint('959', '539', image).point).toEqual({ x: 959, y: 539 })
  })

  it('rejects incomplete, fractional, and out-of-range coordinates', () => {
    expect(validateImagePoint('12', '', image).error).toMatch(/同时输入/)
    expect(validateImagePoint('12.5', '20', image).error).toMatch(/整数/)
    expect(validateImagePoint('960', '20', image).error).toContain('X 为 0–959')
    expect(validateImagePoint('20', '-1', image).error).toContain('Y 为 0–539')
  })
})
