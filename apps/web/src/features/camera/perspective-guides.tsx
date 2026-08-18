import {
  type KeyboardEvent,
  type PointerEvent as ReactPointerEvent,
  useRef,
} from 'react'

import type { SubjectMediaDto } from '../../api/types'
import { toImagePoint, type ImagePoint } from '../coordinates/image-point'

export type GuideSegment = [ImagePoint, ImagePoint]
export type GuideGroup = [GuideSegment, GuideSegment]
export type GuideGroups = [GuideGroup, GuideGroup]
export type ReferenceRelation = 'a_vertical_b_horizontal' | 'both_horizontal_plane'

export interface PerspectivePreview {
  valid: boolean
  fov: number
  horizon: [number, number, number] | null
  gravity: [number, number, number] | null
  message: string
}

const unit = (value: [number, number, number]): [number, number, number] | null => {
  const length = Math.hypot(...value)
  return Number.isFinite(length) && length > 1e-9
    ? [value[0] / length, value[1] / length, value[2] / length]
    : null
}

const cross = (a: [number, number, number], b: [number, number, number]): [number, number, number] => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
]

const line = (segment: GuideSegment): [number, number, number] | null => {
  const value = cross(
    [segment[0].x, segment[0].y, 1],
    [segment[1].x, segment[1].y, 1],
  )
  const length = Math.hypot(value[0], value[1])
  return Number.isFinite(length) && length > 1e-6
    ? [value[0] / length, value[1] / length, value[2] / length]
    : null
}

export function defaultGuideGroups(width: number, height: number): GuideGroups {
  const focal = height * Math.sqrt(3) / 2
  const verticalVp = { x: width / 2, y: height / 2 - focal }
  const horizontalVp = { x: width / 2 + focal, y: height / 2 + focal }
  const toward = (start: ImagePoint, target: ImagePoint): GuideSegment => [start, {
    x: Math.round(start.x + .3 * (target.x - start.x)),
    y: Math.round(start.y + .3 * (target.y - start.y)),
  }]
  return [
    [
      toward({ x: Math.round(width * .25), y: Math.round(height * .8) }, verticalVp),
      toward({ x: Math.round(width * .75), y: Math.round(height * .8) }, verticalVp),
    ],
    [
      toward({ x: Math.round(width * .15), y: Math.round(height * .25) }, horizontalVp),
      toward({ x: Math.round(width * .2), y: Math.round(height * .7) }, horizontalVp),
    ],
  ]
}

export function solvePerspectiveGuides(
  media: Pick<SubjectMediaDto, 'width' | 'height'>,
  groups: GuideGroups,
  relation: ReferenceRelation,
): PerspectivePreview {
  const vanishing = (group: GuideGroup): [number, number, number] | null => {
    const first = line(group[0]); const second = line(group[1])
    if (first === null || second === null) return null
    const point = cross(first, second)
    if (!point.every(Number.isFinite) || Math.abs(point[2]) <= 1e-4) return null
    return [point[0] / point[2], point[1] / point[2], 1]
  }
  const a = vanishing(groups[0]); const b = vanishing(groups[1])
  if (a === null || b === null) return { valid: false, fov: 60, horizon: null, gravity: null, message: '同组线段接近平行，请让两条线沿真实平行边缘产生可见汇聚。' }
  const cx = media.width / 2; const cy = media.height / 2
  const focalSquared = -((a[0] - cx) * (b[0] - cx) + (a[1] - cy) * (b[1] - cy))
  if (!Number.isFinite(focalSquared) || focalSquared <= 1e-6) return { valid: false, fov: 60, horizon: null, gravity: null, message: '两组方向不能得到正焦距；请检查它们在现实中是否互相垂直。' }
  const focal = Math.sqrt(focalSquared)
  const fov = 2 * Math.atan(media.height / (2 * focal)) * 180 / Math.PI
  if (!Number.isFinite(fov) || fov < 5 || fov > 150) return { valid: false, fov, horizon: null, gravity: null, message: '求得的 FOV 超出支持范围，请重新贴合参照线。' }
  const direction = (vp: [number, number, number]): [number, number, number] | null => unit([
    (vp[0] - cx) / focal,
    (vp[1] - cy) / focal,
    1,
  ])
  const da = direction(a); const db = direction(b)
  if (da === null || db === null) return { valid: false, fov, horizon: null, gravity: null, message: '参照方向退化。' }
  let gravity = relation === 'a_vertical_b_horizontal' ? da : unit(cross(da, db))
  if (gravity === null) return { valid: false, fov, horizon: null, gravity: null, message: '两组方向过于接近，无法确定重力轴。' }
  if (gravity[1] > 0) gravity = [-gravity[0], -gravity[1], -gravity[2]]
  const horizon: [number, number, number] = [gravity[0] / focal, gravity[1] / focal, gravity[2] - cx * gravity[0] / focal - cy * gravity[1] / focal]
  return { valid: true, fov, horizon, gravity, message: '几何条件可解；确认时后台会重新验证原始像素线段。' }
}

export function imagePointToFramePercent(point: ImagePoint, media: Pick<SubjectMediaDto, 'width' | 'height'>): { x: number; y: number } {
  const frameWidth = 16; const frameHeight = 9
  const scale = Math.min(frameWidth / media.width, frameHeight / media.height)
  const offsetX = (frameWidth - media.width * scale) / 2
  const offsetY = (frameHeight - media.height * scale) / 2
  return {
    x: (offsetX + (point.x + .5) * scale) / frameWidth * 100,
    y: (offsetY + (point.y + .5) * scale) / frameHeight * 100,
  }
}

export function PerspectiveGuides({ disabled, groups, media, onChange }: {
  disabled: boolean
  groups: GuideGroups
  media: SubjectMediaDto
  onChange(value: GuideGroups): void
}) {
  const drag = useRef<{ group: 0 | 1; segment: 0 | 1; endpoint: 0 | 1 } | null>(null)
  const update = (group: 0 | 1, segment: 0 | 1, endpoint: 0 | 1, point: ImagePoint): void => {
    const next = groups.map((guideGroup) => guideGroup.map((guideSegment) => guideSegment.map((value) => ({ ...value })))) as GuideGroups
    next[group][segment][endpoint] = point
    onChange(next)
  }
  const movePointer = (event: ReactPointerEvent<HTMLDivElement>): void => {
    if (drag.current === null) return
    const point = toImagePoint(event, event.currentTarget.getBoundingClientRect(), media)
    if (point !== null) update(drag.current.group, drag.current.segment, drag.current.endpoint, point)
  }
  const moveKey = (event: KeyboardEvent<HTMLButtonElement>, group: 0 | 1, segment: 0 | 1, endpoint: 0 | 1): void => {
    const delta = event.shiftKey ? 1 : 5
    const current = groups[group][segment][endpoint]
    let x = current.x; let y = current.y
    if (event.key === 'ArrowLeft') x -= delta
    else if (event.key === 'ArrowRight') x += delta
    else if (event.key === 'ArrowUp') y -= delta
    else if (event.key === 'ArrowDown') y += delta
    else return
    event.preventDefault()
    update(group, segment, endpoint, {
      x: Math.max(0, Math.min(media.width - 1, x)),
      y: Math.max(0, Math.min(media.height - 1, y)),
    })
  }
  return (
    <div className="perspective-guides" onPointerMove={movePointer} onPointerUp={() => { drag.current = null }} onPointerCancel={() => { drag.current = null }}>
      {groups.flatMap((group, groupIndex) => group.map((segment, segmentIndex) => {
        const start = imagePointToFramePercent(segment[0], media); const end = imagePointToFramePercent(segment[1], media)
        const width = Math.hypot(end.x - start.x, end.y - start.y)
        const angle = Math.atan2(end.y - start.y, end.x - start.x) * 180 / Math.PI
        const key = `${groupIndex}-${segmentIndex}`
        return <span className={`guide-segment guide-${groupIndex === 0 ? 'a' : 'b'}`} key={key} style={{ left: `${start.x}%`, top: `${start.y}%`, width: `${width}%`, transform: `rotate(${angle}deg)` }}>
          {([start, end] as const).map((_, endpointIndex) => <button
            aria-label={`${groupIndex === 0 ? 'A' : 'B'} 组第 ${segmentIndex + 1} 条线端点 ${endpointIndex + 1}`}
            className="guide-handle"
            disabled={disabled}
            key={endpointIndex}
            onKeyDown={(event) => moveKey(event, groupIndex as 0 | 1, segmentIndex as 0 | 1, endpointIndex as 0 | 1)}
            onPointerDown={(event) => {
              event.preventDefault(); event.stopPropagation(); event.currentTarget.focus(); event.currentTarget.setPointerCapture(event.pointerId)
              drag.current = { group: groupIndex as 0 | 1, segment: segmentIndex as 0 | 1, endpoint: endpointIndex as 0 | 1 }
            }}
            style={{ left: endpointIndex === 0 ? 0 : '100%' }}
            type="button"
          />)}
        </span>
      }))}
    </div>
  )
}
