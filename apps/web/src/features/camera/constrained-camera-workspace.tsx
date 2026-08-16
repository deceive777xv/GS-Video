import {
  type PointerEvent as ReactPointerEvent,
  useEffect,
  useRef,
  useState,
} from 'react'

import type { BackendClient } from '../../api/backend-client'
import type {
  Matrix4,
  MatrixCameraInput,
  PreviewFrameDto,
  ProjectDto,
  SourcePerspectiveCalibrationDto,
  SubjectMediaDto,
} from '../../api/types'
import { toImagePoint, type ImagePoint } from '../coordinates/image-point'

interface WorkspaceProps {
  backend: BackendClient
  busy: boolean
  project: ProjectDto
  onError(value: unknown): void
  onProjectChange(project: ProjectDto): void
  onRefresh(): Promise<ProjectDto>
}

interface MutationWorkspaceProps extends WorkspaceProps {
  mutationPending: boolean
  beginMutation(): boolean
  endMutation(): void
}

interface SourceCalibrationPanelProps extends MutationWorkspaceProps {
  confirmedFoot: ImagePoint | null
  draftDirty: boolean
  onDraftDirtyChange(value: boolean): void
}

interface SynthesisPanelProps extends MutationWorkspaceProps {
  calibrationDraftDirty: boolean
  confirmedFoot: ImagePoint | null
}

interface EulerPose {
  x: number
  y: number
  z: number
  yaw: number
  pitch: number
  roll: number
  fov: number
}

const DEFAULT_POSE: EulerPose = { x: 0, y: 0, z: -4, yaw: 0, pitch: 0, roll: 0, fov: 50 }

type Matrix3 = [[number, number, number], [number, number, number], [number, number, number]]

function multiply3(left: Matrix3, right: Matrix3): Matrix3 {
  const value = (row: 0 | 1 | 2, column: 0 | 1 | 2): number => (
    left[row][0] * right[0][column]
    + left[row][1] * right[1][column]
    + left[row][2] * right[2][column]
  )
  return [
    [value(0, 0), value(0, 1), value(0, 2)],
    [value(1, 0), value(1, 1), value(1, 2)],
    [value(2, 0), value(2, 1), value(2, 2)],
  ]
}

function poseMatrix(pose: EulerPose): Matrix4 {
  const yaw = pose.yaw * Math.PI / 180
  const pitch = pose.pitch * Math.PI / 180
  const roll = pose.roll * Math.PI / 180
  const cy = Math.cos(yaw); const sy = Math.sin(yaw)
  const cp = Math.cos(pitch); const sp = Math.sin(pitch)
  const cr = Math.cos(roll); const sr = Math.sin(roll)
  const rotation = multiply3(
    [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]],
    multiply3(
      [[1, 0, 0], [0, cp, -sp], [0, sp, cp]],
      [[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]],
    ),
  )
  return [
    [rotation[0][0], rotation[0][1], rotation[0][2], pose.x],
    [rotation[1][0], rotation[1][1], rotation[1][2], pose.y],
    [rotation[2][0], rotation[2][1], rotation[2][2], pose.z],
    [0, 0, 0, 1],
  ]
}

function matrixPose(camera: MatrixCameraInput | null): EulerPose {
  if (camera === null) return DEFAULT_POSE
  const matrix = camera.camera_to_world
  const pitch = Math.asin(Math.max(-1, Math.min(1, -matrix[1][2])))
  return {
    x: matrix[0][3], y: matrix[1][3], z: matrix[2][3],
    yaw: Math.atan2(matrix[0][2], matrix[2][2]) * 180 / Math.PI,
    pitch: pitch * 180 / Math.PI,
    roll: Math.atan2(matrix[1][0], matrix[1][1]) * 180 / Math.PI,
    fov: camera.fov_y_degrees,
  }
}

function cameraInput(pose: EulerPose): MatrixCameraInput {
  return { camera_to_world: poseMatrix(pose), fov_y_degrees: pose.fov }
}

function poseFingerprint(pose: EulerPose): string {
  return [pose.x, pose.y, pose.z, pose.yaw, pose.pitch, pose.roll, pose.fov]
    .map((value) => value.toFixed(6)).join('|')
}

function useBlobUrl(): [string | null, (blob: Blob | null) => void] {
  const [url, setUrl] = useState<string | null>(null)
  const current = useRef<string | null>(null)
  const replace = (blob: Blob | null): void => {
    if (current.current !== null) URL.revokeObjectURL(current.current)
    current.current = blob === null ? null : URL.createObjectURL(blob)
    setUrl(current.current)
  }
  useEffect(() => () => { if (current.current !== null) URL.revokeObjectURL(current.current) }, [])
  return [url, replace]
}

function Marker({ point, width, height, label }: { point: ImagePoint; width: number; height: number; label: string }) {
  return (
    <span
      className="ground-point-marker"
      style={{ left: `${(point.x + 0.5) / width * 100}%`, top: `${(point.y + 0.5) / height * 100}%` }}
    >{label}</span>
  )
}

type Vector3 = [number, number, number]

interface CalibrationControls {
  fov: number
  horizonLeft: number
  horizonRight: number
  verticalXBottom: number
  verticalXTop: number
}

function calibrationControls(calibration: SourcePerspectiveCalibrationDto | null): CalibrationControls {
  if (calibration === null) return {
    fov: 50, horizonLeft: 50, horizonRight: 50, verticalXBottom: 50, verticalXTop: 50,
  }
  const { image_width: width, image_height: height } = calibration
  const [a, b, c] = calibration.horizon_line
  const horizonY = (x: number): number => Math.abs(b) <= 1e-9 ? height / 2 : -(a * x + c) / b
  const focal = 0.5 * height / Math.tan(calibration.vertical_fov * Math.PI / 360)
  const [gx, gy, gz] = calibration.gravity_direction_camera
  const bottomX = width / 2
  const bottomY = height - 1
  let topX = bottomX
  if (Math.abs(gz) > 1e-9) {
    const vanishingX = (focal * gx + width / 2 * gz) / gz
    const vanishingY = (focal * gy + height / 2 * gz) / gz
    if (Math.abs(vanishingY - bottomY) > 1e-9) {
      topX = bottomX + (vanishingX - bottomX) * -bottomY / (vanishingY - bottomY)
    }
  } else if (Math.abs(gy) > 1e-9) {
    topX = bottomX - bottomY * gx / gy
  }
  return {
    fov: calibration.vertical_fov,
    horizonLeft: horizonY(0) / height * 100,
    horizonRight: horizonY(width) / height * 100,
    verticalXBottom: bottomX / width * 100,
    verticalXTop: topX / width * 100,
  }
}

function dot3(left: Vector3, right: Vector3): number {
  return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

function cross3(left: Vector3, right: Vector3): Vector3 {
  return [
    left[1] * right[2] - left[2] * right[1],
    left[2] * right[0] - left[0] * right[2],
    left[0] * right[1] - left[1] * right[0],
  ]
}

function unit3(value: Vector3): Vector3 | null {
  const length = Math.hypot(...value)
  return length <= 1e-9 ? null : [value[0] / length, value[1] / length, value[2] / length]
}

function rotate3(value: Vector3, axisValue: Vector3, angle: number): Vector3 {
  const axis = unit3(axisValue)
  if (axis === null) return value
  const cosine = Math.cos(angle); const sine = Math.sin(angle)
  const cross = cross3(axis, value); const dot = dot3(axis, value)
  return [
    value[0] * cosine + cross[0] * sine + axis[0] * dot * (1 - cosine),
    value[1] * cosine + cross[1] * sine + axis[1] * dot * (1 - cosine),
    value[2] * cosine + cross[2] * sine + axis[2] * dot * (1 - cosine),
  ]
}

function projectWorldPoint(point: Vector3, cameraToWorld: Matrix4, fov: number, width: number, height: number): ImagePoint | null {
  const delta: Vector3 = [
    point[0] - cameraToWorld[0][3],
    point[1] - cameraToWorld[1][3],
    point[2] - cameraToWorld[2][3],
  ]
  const camera: Vector3 = [
    cameraToWorld[0][0] * delta[0] + cameraToWorld[1][0] * delta[1] + cameraToWorld[2][0] * delta[2],
    cameraToWorld[0][1] * delta[0] + cameraToWorld[1][1] * delta[1] + cameraToWorld[2][1] * delta[2],
    cameraToWorld[0][2] * delta[0] + cameraToWorld[1][2] * delta[1] + cameraToWorld[2][2] * delta[2],
  ]
  if (camera[2] <= 1e-6) return null
  const focal = 0.5 * height / Math.tan(fov * Math.PI / 360)
  return { x: focal * camera[0] / camera[2] + width / 2, y: focal * camera[1] / camera[2] + height / 2 }
}

function PerspectiveGrid({ width, height, fov, horizonLeft, horizonRight }: {
  width: number; height: number; fov: number; horizonLeft: number; horizonRight: number
}) {
  const leftY = horizonLeft / 100 * height
  const rightY = horizonRight / 100 * height
  const horizon: Vector3 = [leftY - rightY, width, -width * leftY]
  const focal = 0.5 * height / Math.tan(fov * Math.PI / 360)
  let up = unit3([focal * horizon[0], focal * horizon[1], width / 2 * horizon[0] + height / 2 * horizon[1] + horizon[2]])
  if (up === null) return null
  if (up[1] > 0) up = [-up[0], -up[1], -up[2]]
  const optical: Vector3 = [0, 0, 1]
  const forward = unit3([
    optical[0] - dot3(optical, up) * up[0],
    optical[1] - dot3(optical, up) * up[1],
    optical[2] - dot3(optical, up) * up[2],
  ])
  if (forward === null) return null
  const right = unit3(cross3(forward, up))
  if (right === null) return null
  const project = (x: number, z: number): string | null => {
    const point: Vector3 = [
      right[0] * x + forward[0] * z - up[0],
      right[1] * x + forward[1] * z - up[1],
      right[2] * x + forward[2] * z - up[2],
    ]
    if (point[2] <= 1e-6) return null
    return `${focal * point[0] / point[2] + width / 2},${focal * point[1] / point[2] + height / 2}`
  }
  const depthLines = [1.25, 1.7, 2.3, 3.2, 4.5, 6.5, 9]
    .map((z) => [-6, 6].map((x) => project(x, z)).filter((point): point is string => point !== null))
  const longitudinal = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    .map((x) => [1.05, 1.3, 1.7, 2.3, 3.2, 4.5, 6.5, 9, 13].map((z) => project(x, z)).filter((point): point is string => point !== null))
  return <svg aria-label="由当前 FOV 与地平线派生的透视地面网格" className="perspective-ground-grid" preserveAspectRatio="none" viewBox={`0 0 ${width} ${height}`}>{[...depthLines, ...longitudinal].map((points, index) => points.length < 2 ? null : <polyline key={index} points={points.join(' ')} />)}</svg>
}

function ReviewThumbnail({ backend, busy, frame, selected, onError, onSelect }: {
  backend: BackendClient
  busy: boolean
  frame: number
  selected: boolean
  onError(value: unknown): void
  onSelect(): void
}) {
  const [url, replaceUrl] = useBlobUrl()
  useEffect(() => {
    let cancelled = false
    void backend.getSubjectMedia('proxy', frame)
      .then((descriptor) => backend.fetchSubjectMediaArtifact('proxy', descriptor.artifact_id, undefined, frame))
      .then((blob) => { if (!cancelled) replaceUrl(blob) })
      .catch((error) => { if (!cancelled) onError(error) })
    return () => { cancelled = true }
  }, [backend, frame])
  return (
    <button
      aria-label={`选择代表帧 ${frame}`}
      className={selected ? 'audit-thumbnail is-selected' : 'audit-thumbnail'}
      disabled={busy}
      onClick={onSelect}
      type="button"
    >
      {url === null ? <span>#{frame}</span> : <img alt="" src={url} />}
      <small>#{frame}</small>
    </button>
  )
}

interface ReviewRange {
  start_frame: number
  end_frame: number
  review_frames: number[]
}

function AuditReviewGroup({ backend, busy, label, ranges, selectedFrame, onError, onSelect }: {
  backend: BackendClient
  busy: boolean
  label: string
  ranges: ReviewRange[]
  selectedFrame: number
  onError(value: unknown): void
  onSelect(frame: number): void
}) {
  const [open, setOpen] = useState(false)
  const reviewFrames = [...new Set(ranges.flatMap((range) => range.review_frames))]
  const visibleFrames = reviewFrames.slice(0, 12)
  return (
    <details name="visibility-audit-review" onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary>{label}代表帧 · {ranges.length} 段</summary>
      {open ? <div className="audit-thumbnail-strip">
        {visibleFrames.map((frame) => <ReviewThumbnail backend={backend} busy={busy} frame={frame} key={frame} onError={onError} onSelect={() => onSelect(frame)} selected={selectedFrame === frame} />)}
        {reviewFrames.length > visibleFrames.length ? <small>仅显示前 {visibleFrames.length} 个代表帧，可用锚定帧编号查看其余帧。</small> : null}
      </div> : null}
    </details>
  )
}

function SourceCalibrationPanel({
  backend,
  busy,
  project,
  onError,
  onProjectChange,
  confirmedFoot,
  draftDirty,
  onDraftDirtyChange,
  mutationPending: pending,
  beginMutation,
  endMutation,
}: SourceCalibrationPanelProps) {
  const workflow = project.workflow
  const segmentKey = project.stages.segment?.cache_key ?? null
  const initialAnchor = workflow.source_perspective_calibration?.anchor_frame_index
    ?? workflow.subject_visibility_audit?.recommended_anchor_frames[0]
    ?? workflow.subject_prompt?.frame_index
    ?? 0
  const [anchorFrame, setAnchorFrame] = useState(initialAnchor)
  const [media, setMedia] = useState<SubjectMediaDto | null>(null)
  const [sourceUrl, replaceSourceUrl] = useBlobUrl()
  const [alphaUrl, replaceAlphaUrl] = useBlobUrl()
  const initialControls = calibrationControls(workflow.source_perspective_calibration)
  const [fov, setFov] = useState(initialControls.fov)
  const [horizonLeft, setHorizonLeft] = useState(initialControls.horizonLeft)
  const [horizonRight, setHorizonRight] = useState(initialControls.horizonRight)
  const [verticalXBottom, setVerticalXBottom] = useState(initialControls.verticalXBottom)
  const [verticalXTop, setVerticalXTop] = useState(initialControls.verticalXTop)
  const [footDraft, setFootDraft] = useState<ImagePoint | null>(null)
  const sourceFrameRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const calibration = workflow.source_perspective_calibration
    if (calibration === null) return
    const controls = calibrationControls(calibration)
    setAnchorFrame(calibration.anchor_frame_index)
    setFov(controls.fov)
    setHorizonLeft(controls.horizonLeft)
    setHorizonRight(controls.horizonRight)
    setVerticalXBottom(controls.verticalXBottom)
    setVerticalXTop(controls.verticalXTop)
    onDraftDirtyChange(false)
  }, [onDraftDirtyChange, workflow.source_perspective_calibration?.revision])

  useEffect(() => {
    let cancelled = false
    setMedia(null)
    replaceSourceUrl(null)
    replaceAlphaUrl(null)
    setFootDraft(null)
    const load = async (): Promise<void> => {
      try {
        const descriptor = await backend.getSubjectMedia('proxy', anchorFrame)
        const [source, alpha] = await Promise.all([
          backend.fetchSubjectMediaArtifact('proxy', descriptor.artifact_id, undefined, anchorFrame),
          backend.getSubjectMedia('alpha', anchorFrame).then((item) => (
            backend.fetchSubjectMediaArtifact('alpha', item.artifact_id, undefined, anchorFrame)
          )),
        ])
        if (cancelled) return
        setMedia(descriptor)
        replaceSourceUrl(source)
        replaceAlphaUrl(alpha)
        setFootDraft(null)
      } catch (error) {
        if (!cancelled) onError(error)
      }
    }
    void load()
    return () => { cancelled = true }
  }, [anchorFrame, backend, project.project_id, segmentKey])

  const scan = async (): Promise<void> => {
    if (segmentKey === null || !beginMutation()) return
    try {
      onProjectChange(await backend.scanSubjectVisibility(project.project_id, segmentKey))
    } catch (error) { onError(error) } finally { endMutation() }
  }

  const save = async (): Promise<void> => {
    if (media === null || media.frame_index !== anchorFrame || segmentKey === null || !beginMutation()) return
    try {
      const height = media.height
      const width = media.width
      const updated = await backend.calibrateSourcePerspective({
        expected_project_id: project.project_id,
        expected_segment_cache_key: segmentKey,
        anchor_frame_index: anchorFrame,
        image_width: width,
        image_height: height,
        vertical_fov: fov,
        horizon_start: [0, horizonLeft / 100 * height],
        horizon_end: [width, horizonRight / 100 * height],
        vertical_bottom: [verticalXBottom / 100 * width, height - 1],
        vertical_top: [verticalXTop / 100 * width, 0],
      })
      onDraftDirtyChange(false)
      onProjectChange(updated)
    } catch (error) { onError(error) } finally { endMutation() }
  }

  const confirmFoot = async (): Promise<void> => {
    const calibration = workflow.source_perspective_calibration
    if (draftDirty || footDraft === null || calibration === null || calibration.anchor_frame_index !== anchorFrame || !beginMutation()) return
    try {
      onProjectChange(await backend.confirmSourceContact({
        expected_project_id: project.project_id,
        source_calibration_revision: calibration.revision,
        foot_pixel: [footDraft.x, footDraft.y],
      }))
    } catch (error) { onError(error) } finally { endMutation() }
  }

  const chooseFoot = (event: ReactPointerEvent<HTMLDivElement>): void => {
    if (busy || pending || media === null || sourceFrameRef.current === null) return
    const point = toImagePoint(event, sourceFrameRef.current.getBoundingClientRect(), media)
    if (point !== null) setFootDraft(point)
  }

  const audit = workflow.subject_visibility_audit
  return (
    <section aria-busy={busy} className="calibration-card" aria-labelledby="source-calibration-title">
      <div className="calibration-card-heading">
        <div><span className="step-kicker">A · 源画面</span><h3 id="source-calibration-title">透视与可见性校准</h3></div>
        <button disabled={busy || pending || segmentKey === null} onClick={() => void scan()} type="button">扫描全部 Alpha</button>
      </div>
      <p className="technical-note">Alpha 只生成“完整下边界候选”，不会自动断言该处就是脚。半身、坐姿和遮挡素材可直接选纯透视模式。</p>
      {audit === null ? <p>尚未扫描全片可见性。</p> : (
        <><div className="audit-summary">
          <span>完整候选 {audit.fully_visible_ranges.length} 段</span>
          <span>底边裁切 {audit.bottom_cropped_ranges.length} 段</span>
          <span>不确定 {audit.uncertain_ranges.length} 段</span>
        </div><div className="audit-review" aria-label="可见性时间段代表帧复核">
          {([
            ['完整候选', audit.fully_visible_ranges],
            ['底边裁切', audit.bottom_cropped_ranges],
            ['不确定', audit.uncertain_ranges],
          ] as const).map(([label, ranges]) => ranges.length === 0 ? null : <AuditReviewGroup backend={backend} busy={busy || pending} key={label} label={label} onError={onError} onSelect={(frame) => { setAnchorFrame(frame); onDraftDirtyChange(true) }} ranges={ranges} selectedFrame={anchorFrame} />)}
        </div></>
      )}
      <div className="source-calibration-grid">
        <div className="source-reference-frame" onPointerDown={chooseFoot} ref={sourceFrameRef} style={media === null ? undefined : { aspectRatio: `${media.width} / ${media.height}` }}>
          {sourceUrl === null ? <div className="viewport-empty">载入源锚定帧…</div> : <img alt="源透视锚定帧" draggable={false} src={sourceUrl} />}
          {media === null ? null : <PerspectiveGrid fov={fov} height={media.height} horizonLeft={horizonLeft} horizonRight={horizonRight} width={media.width} />}
          <span className="reference-line horizon-line" style={{ left: 0, top: `${horizonLeft}%`, width: '100%', transform: `rotate(${Math.atan2(horizonRight - horizonLeft, 100) * 180 / Math.PI}deg)` }} />
          <span className="reference-line vertical-line" style={{ bottom: 0, left: `${verticalXBottom}%`, height: '100%', transform: `rotate(${Math.atan2(verticalXTop - verticalXBottom, 100) * -180 / Math.PI}deg)` }} />
          {footDraft !== null && media !== null ? <Marker point={footDraft} width={media.width} height={media.height} label="脚?" /> : null}
        </div>
        <div className="calibration-fields">
          <label>源锚定帧<input disabled={busy || pending} min="0" max={Math.max(0, (workflow.source_summary?.frame_count ?? 1) - 1)} onChange={(event) => { setAnchorFrame(Number(event.currentTarget.value)); onDraftDirtyChange(true) }} type="number" value={anchorFrame} /></label>
          {audit?.recommended_anchor_frames.length ? <div className="anchor-suggestions">推荐：{audit.recommended_anchor_frames.map((frame) => <button disabled={busy || pending} key={frame} onClick={() => { setAnchorFrame(frame); onDraftDirtyChange(true) }} type="button">#{frame}</button>)}</div> : null}
          <label>垂直 FOV<input disabled={busy || pending} min="20" max="100" onChange={(event) => { setFov(Number(event.currentTarget.value)); onDraftDirtyChange(true) }} type="range" value={fov} /><output>{fov.toFixed(0)}°</output></label>
          <label>地平线左端<input disabled={busy || pending} min="0" max="100" onChange={(event) => { setHorizonLeft(Number(event.currentTarget.value)); onDraftDirtyChange(true) }} type="range" value={horizonLeft} /></label>
          <label>地平线右端<input disabled={busy || pending} min="0" max="100" onChange={(event) => { setHorizonRight(Number(event.currentTarget.value)); onDraftDirtyChange(true) }} type="range" value={horizonRight} /></label>
          <label>竖直线底端<input disabled={busy || pending} min="-200" max="300" onChange={(event) => { setVerticalXBottom(Number(event.currentTarget.value)); onDraftDirtyChange(true) }} type="range" value={verticalXBottom} /></label>
          <label>竖直线顶端<input disabled={busy || pending} min="-200" max="300" onChange={(event) => { setVerticalXTop(Number(event.currentTarget.value)); onDraftDirtyChange(true) }} type="range" value={verticalXTop} /></label>
          <button disabled={busy || pending || media === null || media.frame_index !== anchorFrame || segmentKey === null} onClick={() => void save()} type="button">确认源透视校准</button>
          <button disabled={busy || pending || draftDirty || footDraft === null || workflow.source_perspective_calibration?.anchor_frame_index !== anchorFrame} onClick={() => void confirmFoot()} type="button">确认脚底候选</button>
          <span>{confirmedFoot === null ? '未确认脚底；仍可使用纯透视模式' : `脚底候选 (${confirmedFoot.x}, ${confirmedFoot.y})`}</span>
        </div>
      </div>
      <div hidden data-alpha-url={alphaUrl ?? ''} data-confirmed-foot={confirmedFoot === null ? '' : `${confirmedFoot.x},${confirmedFoot.y}`} id="source-calibration-authority" />
    </section>
  )
}

function ExplorationPanel({ backend, busy, project, onError, onProjectChange, onRefresh, mutationPending: pending, beginMutation, endMutation }: MutationWorkspaceProps) {
  const [pose, setPose] = useState(() => matrixPose(project.workflow.exploration_camera))
  const [liveUrl, replaceLiveUrl] = useBlobUrl()
  const [frame, setFrame] = useState<PreviewFrameDto | null>(null)
  const [frameUrl, replaceFrameUrl] = useBlobUrl()
  const [frameFingerprint, setFrameFingerprint] = useState<string | null>(null)
  const [points, setPoints] = useState<ImagePoint[]>([])
  const [candidate, setCandidate] = useState<ImagePoint | null>(null)
  const [flipNormal, setFlipNormal] = useState(false)
  const [orbitP0, setOrbitP0] = useState(false)
  const viewportRef = useRef<HTMLDivElement>(null)
  const generation = useRef(project.workflow.preview?.generation ?? 0)
  const liveRequest = useRef(0)
  const dragging = useRef<{ x: number; y: number } | null>(null)
  const fingerprint = poseFingerprint(pose)
  const frozen = frame !== null && frameFingerprint === fingerprint
  const confirmed = frozen
    && project.workflow.confirmed_camera_revision === frame.camera_revision
    && project.workflow.confirmed_preview_artifact_id === frame.artifact_id

  useEffect(() => {
    if (busy) return
    const controller = new AbortController()
    const timeout = setTimeout(() => {
      const requestId = ++liveRequest.current
      void backend.renderLivePreview({ expected_project_id: project.project_id, request_id: requestId, width: 960, height: 540, camera: cameraInput(pose) }, controller.signal)
        .then((blob) => { if (requestId === liveRequest.current) replaceLiveUrl(blob) })
        .catch((error) => { if (!controller.signal.aborted && requestId === liveRequest.current) onError(error) })
    }, 80)
    return () => { clearTimeout(timeout); controller.abort() }
  }, [backend, busy, fingerprint])

  useEffect(() => { setCandidate(null); setPoints([]); setFlipNormal(false) }, [fingerprint])

  const moveLocal = (right: number, down: number, forward: number): void => {
    setPose((current) => {
      const matrix = poseMatrix(current)
      return {
        ...current,
        x: current.x + matrix[0][0] * right + matrix[0][1] * down + matrix[0][2] * forward,
        y: current.y + matrix[1][0] * right + matrix[1][1] * down + matrix[1][2] * forward,
        z: current.z + matrix[2][0] * right + matrix[2][1] * down + matrix[2][2] * forward,
      }
    })
  }

  const orbitGround = (deltaX: number, deltaY: number): void => {
    const ground = project.workflow.local_ground_anchor
    if (ground === null) return
    setPose((current) => {
      const pivot = ground.p0_world
      const offset: Vector3 = [current.x - pivot[0], current.y - pivot[1], current.z - pivot[2]]
      const yawed = rotate3(offset, ground.plane_normal, -deltaX * Math.PI / 900)
      const matrix = poseMatrix(current)
      const pitched = rotate3(yawed, [matrix[0][0], matrix[1][0], matrix[2][0]], -deltaY * Math.PI / 900)
      const x = pivot[0] + pitched[0]; const y = pivot[1] + pitched[1]; const z = pivot[2] + pitched[2]
      const dx = pivot[0] - x; const dy = pivot[1] - y; const dz = pivot[2] - z
      const distance = Math.hypot(dx, dy, dz)
      return {
        ...current, x, y, z,
        yaw: Math.atan2(dx, dz) * 180 / Math.PI,
        pitch: Math.asin(Math.max(-1, Math.min(1, -dy / distance))) * 180 / Math.PI,
      }
    })
  }

  const freeze = async (): Promise<void> => {
    if (!beginMutation()) return
    try {
      generation.current += 1
      const result = await backend.renderPreview({ expected_project_id: project.project_id, generation: generation.current, width: 960, height: 540, camera: cameraInput(pose) })
      const blob = await backend.fetchPreviewArtifact(result.artifact_id)
      setFrame(result); setFrameFingerprint(fingerprint); replaceFrameUrl(blob)
      onProjectChange(await onRefresh())
    } catch (error) { onError(error) } finally { endMutation() }
  }

  const confirmFreeze = async (): Promise<void> => {
    if (frame === null || !beginMutation()) return
    try { onProjectChange(await backend.confirmCamera(project.project_id, frame.camera_revision)) } catch (error) { onError(error) } finally { endMutation() }
  }

  const choosePoint = (event: ReactPointerEvent<HTMLDivElement>): void => {
    if (!confirmed || frame === null || viewportRef.current === null || points.length >= 3) return
    const point = toImagePoint(event, viewportRef.current.getBoundingClientRect(), frame)
    if (point !== null) setCandidate(point)
  }

  const confirmCandidate = (): void => {
    if (candidate === null) return
    setPoints((current) => [...current, candidate]); setCandidate(null)
  }

  const saveGround = async (): Promise<void> => {
    if (frame === null || points.length !== 3 || !beginMutation()) return
    try {
      onProjectChange(await backend.calibrateLocalGround({
        expected_project_id: project.project_id,
        preview_artifact_id: frame.artifact_id,
        camera_revision: frame.camera_revision,
        pick_buffer_revision: frame.pick_buffer_revision,
        points: points.map((point) => [point.x, point.y]) as [[number, number], [number, number], [number, number]],
        flip_normal: flipNormal,
      }))
    } catch (error) { onError(error) } finally { endMutation() }
  }

  const updateNumber = (key: keyof EulerPose, value: number): void => {
    if (Number.isFinite(value)) setPose((current) => ({ ...current, [key]: value }))
  }
  const focusGround = (): void => {
    const ground = project.workflow.local_ground_anchor
    if (ground === null) return
    setPose((current) => {
      const dx = ground.p0_world[0] - current.x
      const dy = ground.p0_world[1] - current.y
      const dz = ground.p0_world[2] - current.z
      const distance = Math.hypot(dx, dy, dz)
      if (distance <= 1e-9) return current
      return {
        ...current,
        yaw: Math.atan2(dx, dz) * 180 / Math.PI,
        pitch: Math.asin(Math.max(-1, Math.min(1, -dy / distance))) * 180 / Math.PI,
      }
    })
  }
  const planePoints = points.length === 3
    ? points as [ImagePoint, ImagePoint, ImagePoint]
    : null
  const confirmedGround = project.workflow.local_ground_anchor
  const normalArrow = frame === null || confirmedGround === null
    || confirmedGround.preview_artifact_id !== frame.artifact_id
    ? null
    : (() => {
        const p0 = confirmedGround.p0_world
        const scale = Math.max(
          Math.hypot(confirmedGround.p1_world[0] - p0[0], confirmedGround.p1_world[1] - p0[1], confirmedGround.p1_world[2] - p0[2]),
          Math.hypot(confirmedGround.p2_world[0] - p0[0], confirmedGround.p2_world[1] - p0[1], confirmedGround.p2_world[2] - p0[2]),
        ) * 0.6
        const endpoint: Vector3 = [
          p0[0] + confirmedGround.plane_normal[0] * scale,
          p0[1] + confirmedGround.plane_normal[1] * scale,
          p0[2] + confirmedGround.plane_normal[2] * scale,
        ]
        const start = projectWorldPoint(p0, confirmedGround.frozen_camera_to_world, pose.fov, frame.width, frame.height)
        const end = projectWorldPoint(endpoint, confirmedGround.frozen_camera_to_world, pose.fov, frame.width, frame.height)
        return start === null || end === null ? null : { start, end }
      })()

  return (
    <section aria-busy={busy} className="calibration-card" aria-labelledby="exploration-title">
      <div className="calibration-card-heading"><div><span className="step-kicker">B · GS 场景</span><h3 id="exploration-title">6DoF 探索与三点局部地面</h3></div><span className={project.workflow.local_ground_anchor === null ? 'chip' : 'chip chip-ok'}>{project.workflow.local_ground_anchor === null ? '地面未标定' : `地面 r${project.workflow.local_ground_anchor.revision}`}</span></div>
      <div className="viewport-layout">
        <div
          aria-label="Gaussian 6DoF 探索视口"
          className="viewport-frame preview-surface exploration-viewport"
          onPointerDown={(event) => { if (!busy) { dragging.current = { x: event.clientX, y: event.clientY }; choosePoint(event) } }}
          onPointerMove={(event) => {
            if (busy || dragging.current === null || confirmed) return
            const dx = event.clientX - dragging.current.x; const dy = event.clientY - dragging.current.y
            dragging.current = { x: event.clientX, y: event.clientY }
            if (orbitP0) orbitGround(dx, dy)
            else setPose((current) => ({ ...current, yaw: current.yaw + dx * 0.2, pitch: Math.max(-89, Math.min(89, current.pitch - dy * 0.2)) }))
          }}
          onPointerUp={() => { dragging.current = null }}
          onWheel={(event) => { event.preventDefault(); if (!busy) moveLocal(0, 0, event.deltaY > 0 ? -0.25 : 0.25) }}
          ref={viewportRef}
        >
          {(frozen ? frameUrl : liveUrl) === null ? <div className="viewport-empty">准备 Gaussian 预览…</div> : <img alt={frozen ? '冻结的 Gaussian 三点标定帧' : 'Gaussian 实时探索预览'} draggable={false} src={(frozen ? frameUrl : liveUrl) ?? ''} />}
          {frame !== null ? points.map((point, index) => <Marker key={`${point.x}-${point.y}`} point={point} width={frame.width} height={frame.height} label={`P${index}`} />) : null}
          {frame !== null && candidate !== null ? <Marker point={candidate} width={frame.width} height={frame.height} label={`P${points.length}?`} /> : null}
          {frame !== null && planePoints !== null ? <svg aria-label="局部地面三角形与法线方向" className="ground-plane-overlay" preserveAspectRatio="none" viewBox={`0 0 ${frame.width} ${frame.height}`}><defs><marker id="ground-normal-arrow" markerHeight="7" markerWidth="7" orient="auto" refX="5" refY="3.5"><path d="M0,0 L7,3.5 L0,7 Z" /></marker></defs><polygon points={planePoints.map((point) => `${point.x + 0.5},${point.y + 0.5}`).join(' ')} />{normalArrow === null ? null : <line className="ground-normal-arrow" markerEnd="url(#ground-normal-arrow)" x1={normalArrow.start.x} x2={normalArrow.end.x} y1={normalArrow.start.y} y2={normalArrow.end.y} />}<text x={(planePoints[0].x + planePoints[1].x + planePoints[2].x) / 3} y={(planePoints[0].y + planePoints[1].y + planePoints[2].y) / 3}>{flipNormal ? 'N⊗' : 'N⊙'}</text></svg> : null}
        </div>
        <aside className="viewport-controls">
          <div className="camera-readout sixdof-readout">
            {(['x', 'y', 'z', 'yaw', 'pitch', 'roll'] as const).map((key) => <label key={key}>{key.toUpperCase()}<input aria-label={`探索相机 ${key}`} disabled={busy || confirmed} onChange={(event) => updateNumber(key, Number(event.currentTarget.value))} step="0.1" type="number" value={Number(pose[key].toFixed(3))} /></label>)}
          </div>
          <label>探索 FOV<input disabled={busy || confirmed} min="20" max="100" onChange={(event) => updateNumber('fov', Number(event.currentTarget.value))} type="range" value={pose.fov} /><output>{pose.fov.toFixed(0)}°</output></label>
          <div className="movement-pad"><button disabled={busy} onClick={() => moveLocal(-0.25, 0, 0)} type="button">左</button><button disabled={busy} onClick={() => moveLocal(0, 0, 0.25)} type="button">前</button><button disabled={busy} onClick={() => moveLocal(0.25, 0, 0)} type="button">右</button><button disabled={busy} onClick={() => moveLocal(0, -0.25, 0)} type="button">上</button><button disabled={busy} onClick={() => moveLocal(0, 0, -0.25)} type="button">后</button><button disabled={busy} onClick={() => moveLocal(0, 0.25, 0)} type="button">下</button></div>
          <button disabled={busy || confirmed || project.workflow.local_ground_anchor === null} onClick={focusGround} type="button">聚焦局部地面 P0</button>
          <label><input checked={orbitP0} disabled={busy || confirmed || project.workflow.local_ground_anchor === null} onChange={(event) => setOrbitP0(event.currentTarget.checked)} type="checkbox" />拖拽时围绕 P0 Orbit</label>
          <button disabled={busy || confirmed} onClick={() => setPose(DEFAULT_POSE)} type="button">重置探索相机</button>
          <button disabled={busy || pending || confirmed} onClick={() => void freeze()} type="button">冻结当前探索视角</button>
          <button disabled={busy || pending || !frozen || confirmed} onClick={() => void confirmFreeze()} type="button">确认冻结视角</button>
          <fieldset disabled={busy || pending}><legend>同一冻结帧的局部地面</legend><p>{points.length}/3 个点已确认</p><button disabled={!confirmed || candidate === null} onClick={confirmCandidate} type="button">确认 P{points.length} 候选</button><label><input checked={flipNormal} disabled={points.length !== 3} onChange={(event) => setFlipNormal(event.currentTarget.checked)} type="checkbox" />翻转法线（N⊙ / N⊗）</label><button disabled={points.length !== 3} onClick={() => void saveGround()} type="button">确认三点局部地面</button></fieldset>
        </aside>
      </div>
    </section>
  )
}

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image(); image.onload = () => resolve(image); image.onerror = reject; image.src = url
  })
}

function fittedPreviewSize(width: number, height: number): { width: number; height: number } {
  const scale = Math.min(1, 960 / width, 540 / height)
  return {
    width: Math.max(1, Math.round(width * scale)),
    height: Math.max(1, Math.round(height * scale)),
  }
}

function SynthesisPanel({ backend, busy, project, onError, onProjectChange, calibrationDraftDirty, confirmedFoot, mutationPending: pending, beginMutation, endMutation }: SynthesisPanelProps) {
  const workflow = project.workflow
  const calibration = workflow.source_perspective_calibration
  const ground = workflow.local_ground_anchor
  const existing = workflow.synthesis_placement
  const [mode, setMode] = useState<'contact' | 'perspective'>(existing?.mode ?? 'perspective')
  const [azimuth, setAzimuth] = useState(existing?.scene_azimuth ?? 0)
  const [scale, setScale] = useState(existing?.subject_to_scene_scale ?? 1)
  const [offsetX, setOffsetX] = useState(existing?.composition_offset_local[0] ?? 0)
  const [offsetY, setOffsetY] = useState(existing?.composition_offset_local[1] ?? 0)
  const [backgroundUrl, replaceBackgroundUrl] = useBlobUrl()
  const [sourceUrl, replaceSourceUrl] = useBlobUrl()
  const [alphaUrl, replaceAlphaUrl] = useBlobUrl()
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const requestId = useRef(10_000)
  const previewSize = calibration === null
    ? { width: 960, height: 540 }
    : fittedPreviewSize(calibration.image_width, calibration.image_height)

  useEffect(() => {
    if (confirmedFoot === null && mode === 'contact') setMode('perspective')
  }, [confirmedFoot, mode])

  useEffect(() => {
    replaceSourceUrl(null)
    replaceAlphaUrl(null)
    if (calibration === null) return
    let cancelled = false
    void Promise.all([
      backend.getSubjectMedia('proxy', calibration.anchor_frame_index),
      backend.getSubjectMedia('alpha', calibration.anchor_frame_index),
    ]).then(async ([proxy, alpha]) => Promise.all([
      backend.fetchSubjectMediaArtifact('proxy', proxy.artifact_id, undefined, calibration.anchor_frame_index),
      backend.fetchSubjectMediaArtifact('alpha', alpha.artifact_id, undefined, calibration.anchor_frame_index),
    ])).then(([proxy, alpha]) => { if (!cancelled) { replaceSourceUrl(proxy); replaceAlphaUrl(alpha) } }).catch((error) => { if (!cancelled) onError(error) })
    return () => { cancelled = true }
  }, [backend, calibration?.revision, project.project_id, project.stages.segment?.cache_key])

  useEffect(() => {
    requestId.current += 1
    replaceBackgroundUrl(null)
    if (busy || existing === null) return
    const controller = new AbortController()
    const current = ++requestId.current
    void backend.renderLivePreview({ expected_project_id: project.project_id, request_id: current, width: previewSize.width, height: previewSize.height, camera: { camera_to_world: existing.anchor_camera_to_world, fov_y_degrees: calibration?.vertical_fov ?? 50 } }, controller.signal)
      .then((blob) => { if (current === requestId.current) replaceBackgroundUrl(blob) })
      .catch((error) => { if (!controller.signal.aborted && current === requestId.current) onError(error) })
    return () => controller.abort()
  }, [backend, busy, existing?.solver_cache_key, previewSize.width, previewSize.height, project.project_id, project.scene_ply_asset_id, project.scene_ply])

  useEffect(() => {
    const initialCanvas = canvasRef.current
    if (initialCanvas !== null) initialCanvas.getContext('2d')?.clearRect(0, 0, initialCanvas.width, initialCanvas.height)
    if (backgroundUrl === null || sourceUrl === null || alphaUrl === null || canvasRef.current === null) return
    let cancelled = false
    void Promise.all([loadImage(backgroundUrl), loadImage(sourceUrl), loadImage(alphaUrl)]).then(([background, source, alpha]) => {
      if (cancelled || canvasRef.current === null) return
      const canvas = canvasRef.current; canvas.width = previewSize.width; canvas.height = previewSize.height
      const context = canvas.getContext('2d'); if (context === null) return
      context.drawImage(background, 0, 0, canvas.width, canvas.height)
      const foreground = document.createElement('canvas'); foreground.width = canvas.width; foreground.height = canvas.height
      const fg = foreground.getContext('2d'); if (fg === null) return
      fg.drawImage(source, 0, 0, canvas.width, canvas.height)
      const pixels = fg.getImageData(0, 0, canvas.width, canvas.height)
      const maskCanvas = document.createElement('canvas'); maskCanvas.width = canvas.width; maskCanvas.height = canvas.height
      const maskContext = maskCanvas.getContext('2d'); if (maskContext === null) return
      maskContext.drawImage(alpha, 0, 0, canvas.width, canvas.height)
      const mask = maskContext.getImageData(0, 0, canvas.width, canvas.height)
      for (let index = 0; index < pixels.data.length; index += 4) pixels.data[index + 3] = mask.data[index] ?? 0
      fg.putImageData(pixels, 0, 0); context.drawImage(foreground, 0, 0)
    }).catch((error) => { if (!cancelled) onError(error) })
    return () => { cancelled = true }
  }, [backgroundUrl, sourceUrl, alphaUrl, previewSize.width, previewSize.height])

  const solve = async (): Promise<void> => {
    if (calibrationDraftDirty || calibration === null || ground === null || (mode === 'contact' && confirmedFoot === null) || !beginMutation()) return
    try {
      onProjectChange(await backend.solveSynthesisPlacement({
        expected_project_id: project.project_id,
        source_calibration_revision: calibration.revision,
        ground_anchor_revision: ground.revision,
        mode,
        scene_azimuth: normalizedAzimuth,
        subject_to_scene_scale: scale,
        composition_offset_local: mode === 'perspective' ? [offsetX, offsetY] : [0, 0],
        foot_pixel: mode === 'contact' && confirmedFoot !== null ? [confirmedFoot.x, confirmedFoot.y] : null,
      }))
    } catch (error) { onError(error) } finally { endMutation() }
  }

  const confirm = async (): Promise<void> => {
    if (calibrationDraftDirty || existing === null || !beginMutation()) return
    try { onProjectChange(await backend.confirmSynthesisPlacement(project.project_id, existing.revision)) } catch (error) { onError(error) } finally { endMutation() }
  }
  const normalizedAzimuth = ((azimuth + 180) % 360 + 360) % 360 - 180
  const draftMatchesExisting = existing !== null
    && existing.mode === mode
    && Math.abs(existing.scene_azimuth - normalizedAzimuth) <= 1e-9
    && Math.abs(existing.subject_to_scene_scale - scale) <= 1e-9
    && (
      mode === 'contact'
        ? confirmedFoot !== null
          && existing.source_contact_revision === workflow.subject_contact_constraint?.revision
        : Math.abs(existing.composition_offset_local[0] - offsetX) <= 1e-9
          && Math.abs(existing.composition_offset_local[1] - offsetY) <= 1e-9
    )

  return (
    <section aria-busy={busy} className="calibration-card" aria-labelledby="synthesis-title">
      <div className="calibration-card-heading"><div><span className="step-kicker">C · 合成机位</span><h3 id="synthesis-title">受约束放置与视觉微调</h3></div><span className={workflow.confirmed_synthesis_placement_revision === existing?.revision ? 'chip chip-ok' : 'chip'}>{workflow.confirmed_synthesis_placement_revision === existing?.revision ? `已确认 r${existing?.revision}` : '待确认'}</span></div>
      {calibration === null || ground === null ? <p>请先完成源透视校准和三点局部地面。</p> : (
        <div className="synthesis-layout">
          <div className="viewport-frame synthesis-preview" style={{ aspectRatio: `${previewSize.width} / ${previewSize.height}` }}>{existing === null ? <div className="viewport-empty">调整参数后生成静态合成预览</div> : <canvas aria-label="源人物与受约束 GS 背景合成预览" ref={canvasRef} />}</div>
          <aside className="viewport-controls">
            <fieldset disabled={busy || pending || calibrationDraftDirty}><legend>约束模式</legend><label><input checked={mode === 'perspective'} onChange={() => setMode('perspective')} type="radio" />纯透视（不声明接触）</label><label><input checked={mode === 'contact'} disabled={confirmedFoot === null} onChange={() => setMode('contact')} type="radio" />脚底接触 P0</label></fieldset>
            <label>场景方位角<input disabled={busy || pending || calibrationDraftDirty} max="179.9" min="-180" onChange={(event) => setAzimuth(Number(event.currentTarget.value))} step="0.1" type="number" value={azimuth} /></label>
            <label>人物与场景比例<input disabled={busy || pending || calibrationDraftDirty} max="4" min="0.25" onChange={(event) => setScale(Number(event.currentTarget.value))} step="0.05" type="range" value={scale} /><output>{scale.toFixed(2)}×</output></label>
            {mode === 'perspective' ? <><label>构图横移<input disabled={busy || pending || calibrationDraftDirty} onChange={(event) => setOffsetX(Number(event.currentTarget.value))} step="0.1" type="number" value={offsetX} /></label><label>构图纵移<input disabled={busy || pending || calibrationDraftDirty} onChange={(event) => setOffsetY(Number(event.currentTarget.value))} step="0.1" type="number" value={offsetY} /></label></> : <div className="confirmed-contact-readout">已确认脚底 <strong>({confirmedFoot?.x}, {confirmedFoot?.y})</strong></div>}
            <button disabled={busy || pending || calibrationDraftDirty || (mode === 'contact' && confirmedFoot === null)} onClick={() => void solve()} type="button">更新受约束合成预览</button>
            <button disabled={busy || pending || calibrationDraftDirty || !draftMatchesExisting} onClick={() => void confirm()} type="button">确认合成机位</button>
            <p className="technical-note">相机自身 Yaw / Pitch / Roll 和 FOV 均由源透视、局部平面与场景方位角派生。</p>
          </aside>
        </div>
      )}
    </section>
  )
}

export function ConstrainedCameraWorkspace(props: WorkspaceProps) {
  const [calibrationDraftDirty, setCalibrationDraftDirty] = useState(false)
  const [mutationPending, setMutationPending] = useState(false)
  const mutationLock = useRef(false)
  const beginMutation = (): boolean => {
    if (mutationLock.current) return false
    mutationLock.current = true
    setMutationPending(true)
    return true
  }
  const endMutation = (): void => {
    mutationLock.current = false
    setMutationPending(false)
  }
  const mutationProps = { mutationPending, beginMutation, endMutation }
  const foot = props.project.workflow.subject_contact_constraint?.foot_pixel
  const confirmedFoot = foot === undefined ? null : { x: foot[0], y: foot[1] }
  return (
    <div className="constrained-camera-workspace">
      <SourceCalibrationPanel {...props} {...mutationProps} confirmedFoot={confirmedFoot} draftDirty={calibrationDraftDirty} onDraftDirtyChange={setCalibrationDraftDirty} />
      <ExplorationPanel {...props} {...mutationProps} />
      <SynthesisPanel {...props} {...mutationProps} calibrationDraftDirty={calibrationDraftDirty} confirmedFoot={confirmedFoot} />
    </div>
  )
}
