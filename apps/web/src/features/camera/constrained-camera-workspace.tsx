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
} from '../../api/types'
import { toImagePoint, type ImagePoint } from '../coordinates/image-point'

export type CameraPanel = 'source' | 'scene' | 'synthesis'

interface CameraWorkspaceProps {
  backend: BackendClient
  busy: boolean
  project: ProjectDto
  onError(value: unknown): void
  onProjectChange(project: ProjectDto): void
  onRefresh(): Promise<ProjectDto>
  panel: CameraPanel
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

type Matrix3 = [[number, number, number], [number, number, number], [number, number, number]]

const DEFAULT_POSE: EulerPose = {
  x: 0,
  y: 0,
  z: -4,
  yaw: 0,
  pitch: 0,
  roll: 0,
  fov: 50,
}

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
    x: matrix[0][3],
    y: matrix[1][3],
    z: matrix[2][3],
    yaw: Math.atan2(matrix[0][2], matrix[2][2]) * 180 / Math.PI,
    pitch: pitch * 180 / Math.PI,
    roll: Math.atan2(matrix[1][0], matrix[1][1]) * 180 / Math.PI,
    fov: camera.fov_y_degrees,
  }
}

function cameraInput(pose: EulerPose): MatrixCameraInput {
  return { camera_to_world: poseMatrix(pose), fov_y_degrees: pose.fov }
}

function previewAuthority(projectId: string, preview: PreviewFrameDto | null): string | null {
  if (preview === null) return null
  return JSON.stringify([
    projectId,
    preview.artifact_id,
    preview.generation,
    preview.width,
    preview.height,
    preview.camera_revision,
    preview.pick_buffer_revision,
  ])
}

function Marker({ point, frame, label }: {
  point: ImagePoint
  frame: PreviewFrameDto
  label: string
}) {
  const scale = Math.min(16 / frame.width, 9 / frame.height)
  const left = ((16 - frame.width * scale) / 2 + (point.x + 0.5) * scale) / 16 * 100
  const top = ((9 - frame.height * scale) / 2 + (point.y + 0.5) * scale) / 9 * 100
  return <span className="ground-point-marker" style={{ left: `${left}%`, top: `${top}%` }}>{label}</span>
}

function NumberField({ disabled, label, value, onChange, min, max, step = 0.1 }: {
  disabled: boolean
  label: string
  value: number
  onChange(value: number): void
  min?: number
  max?: number
  step?: number
}) {
  return (
    <label>{label}<input
      aria-label={label}
      disabled={disabled}
      max={max}
      min={min}
      onChange={(event) => {
        const next = Number(event.currentTarget.value)
        if (Number.isFinite(next)) onChange(next)
      }}
      step={step}
      type="number"
      value={value}
    /></label>
  )
}

export function ConstrainedCameraWorkspace({
  backend,
  busy,
  project,
  onError,
  onProjectChange,
  onRefresh,
  panel,
}: CameraWorkspaceProps) {
  void panel
  const [pose, setPose] = useState(() => matrixPose(project.workflow.exploration_camera))
  const [frame, setFrame] = useState<PreviewFrameDto | null>(null)
  const [frameUrl, setFrameUrl] = useState<string | null>(null)
  const [hints, setHints] = useState<ImagePoint[]>([])
  const [pending, setPending] = useState(false)
  const [scaleText, setScaleText] = useState(String(project.workflow.gs_scale))
  const [azimuthText, setAzimuthText] = useState(String(project.workflow.scene_azimuth))
  const viewportRef = useRef<HTMLDivElement>(null)
  const generation = useRef(project.workflow.preview?.generation ?? 0)
  const frameUrlRef = useRef<string | null>(null)
  const frameAuthorityRef = useRef<string | null>(null)
  const previewRequest = useRef(0)
  const onErrorRef = useRef(onError)
  onErrorRef.current = onError
  const authoritativePreview = project.workflow.preview
  const authoritativePreviewKey = previewAuthority(project.project_id, authoritativePreview)
  const targetGround = project.workflow.target_ground
  const candidateMatchesFrame = frame !== null
    && targetGround !== null
    && targetGround.preview_artifact_id === frame.artifact_id
    && targetGround.camera_revision === frame.camera_revision
    && targetGround.pick_buffer_revision === frame.pick_buffer_revision

  const replaceFrameUrl = (next: string | null): void => {
    if (frameUrlRef.current !== null) URL.revokeObjectURL(frameUrlRef.current)
    frameUrlRef.current = next
    setFrameUrl(next)
  }
  useEffect(() => {
    generation.current = authoritativePreview?.generation ?? 0
    const request = ++previewRequest.current
    if (authoritativePreview === null || authoritativePreviewKey === null) {
      frameAuthorityRef.current = null
      setFrame(null)
      replaceFrameUrl(null)
      return
    }
    if (
      frameAuthorityRef.current === authoritativePreviewKey
      && frameUrlRef.current !== null
    ) return

    frameAuthorityRef.current = null
    setFrame(null)
    replaceFrameUrl(null)
    const controller = new AbortController()
    void backend.fetchPreviewArtifact(
      authoritativePreview.artifact_id,
      controller.signal,
    ).then((blob) => {
      if (controller.signal.aborted || previewRequest.current !== request) return
      const nextUrl = URL.createObjectURL(blob)
      if (controller.signal.aborted || previewRequest.current !== request) {
        URL.revokeObjectURL(nextUrl)
        return
      }
      frameAuthorityRef.current = authoritativePreviewKey
      setFrame(authoritativePreview)
      replaceFrameUrl(nextUrl)
    }).catch((error: unknown) => {
      if (controller.signal.aborted || previewRequest.current !== request) return
      onErrorRef.current(error instanceof Error ? error : '无法载入当前 Gaussian 场景预览。')
    })
    return () => controller.abort()
  }, [authoritativePreviewKey, backend])
  useEffect(() => () => {
    previewRequest.current += 1
    frameAuthorityRef.current = null
    replaceFrameUrl(null)
  }, [])

  const updatePose = (key: keyof EulerPose, value: number): void => {
    setPose((current) => ({ ...current, [key]: value }))
    frameAuthorityRef.current = null
    setFrame(null)
    replaceFrameUrl(null)
    setHints([])
  }

  const renderSelectionView = async (): Promise<void> => {
    if (pending || busy) return
    setPending(true)
    try {
      generation.current += 1
      const rendered = await backend.renderPreview({
        expected_project_id: project.project_id,
        generation: generation.current,
        width: 960,
        height: 540,
        camera: cameraInput(pose),
      })
      const blob = await backend.fetchPreviewArtifact(rendered.artifact_id)
      frameAuthorityRef.current = previewAuthority(project.project_id, rendered)
      replaceFrameUrl(URL.createObjectURL(blob))
      setFrame(rendered)
      setHints([])
      onProjectChange(await onRefresh())
    } catch (error) {
      onError(error)
    } finally {
      setPending(false)
    }
  }

  const addHint = (event: ReactPointerEvent<HTMLDivElement>): void => {
    if (frame === null || viewportRef.current === null || hints.length >= 3 || pending || busy) return
    const point = toImagePoint(event, viewportRef.current.getBoundingClientRect(), frame)
    if (point === null || hints.some((item) => item.x === point.x && item.y === point.y)) return
    setHints((current) => [...current, point])
  }

  const fitGround = async (): Promise<void> => {
    if (frame === null || hints.length !== 3 || pending || busy) return
    setPending(true)
    try {
      const [p0, p1, p2] = hints
      if (p0 === undefined || p1 === undefined || p2 === undefined) return
      onProjectChange(await backend.fitTargetGround({
        expected_project_id: project.project_id,
        preview_artifact_id: frame.artifact_id,
        camera_revision: frame.camera_revision,
        pick_buffer_revision: frame.pick_buffer_revision,
        hints: [[p0.x, p0.y], [p1.x, p1.y], [p2.x, p2.y]],
      }))
    } catch (error) {
      onError(error)
    } finally {
      setPending(false)
    }
  }

  const confirmGround = async (): Promise<void> => {
    if (targetGround === null || targetGround.confirmed || pending || busy) return
    setPending(true)
    try {
      onProjectChange(await backend.confirmTargetGround(project.project_id, targetGround.revision))
    } catch (error) {
      onError(error)
    } finally {
      setPending(false)
    }
  }

  const saveAlignment = async (): Promise<void> => {
    const gsScale = Number(scaleText)
    const sceneAzimuth = Number(azimuthText)
    if (!Number.isFinite(gsScale) || gsScale < 0.001 || gsScale > 1000
      || !Number.isFinite(sceneAzimuth) || sceneAzimuth < -180 || sceneAzimuth >= 180) return
    setPending(true)
    try {
      onProjectChange(await backend.updateProject({
        expected_project_id: project.project_id,
        gs_scale: gsScale,
        scene_azimuth: sceneAzimuth,
      }))
    } catch (error) {
      onError(error)
    } finally {
      setPending(false)
    }
  }

  const shownHints = candidateMatchesFrame && targetGround !== null
    ? targetGround.hint_pixels.map(([x, y]) => ({ x, y }))
    : hints

  return (
    <section aria-busy={busy || pending} className="calibration-card" aria-labelledby="target-ground-title">
      <div className="calibration-card-heading">
        <div><span className="step-kicker">GS · 自动地面对齐</span><h3 id="target-ground-title">探索场景并给出三个地面提示</h3></div>
        <span className={targetGround?.confirmed === true ? 'chip chip-ok' : 'chip'}>
          {targetGround === null ? '尚无候选' : targetGround.confirmed ? `已确认 r${targetGround.revision}` : `候选 r${targetGround.revision}`}
        </span>
      </div>
      <div className="viewport-layout">
        <div
          aria-label="Gaussian 地面提示视口"
          className="viewport-frame preview-surface exploration-viewport"
          onPointerDown={addHint}
          ref={viewportRef}
        >
          {frameUrl === null ? <div className="viewport-empty">调整探索相机后，生成可点选视图</div> : <img alt="Gaussian 地面选择视图" draggable={false} src={frameUrl} />}
          {frame === null ? null : shownHints.map((point, index) => <Marker frame={frame} key={`${point.x}:${point.y}:${index}`} label={`H${index}`} point={point} />)}
          {frame !== null && shownHints.length === 3 ? <svg aria-label="自动地面候选范围" className="ground-plane-overlay" preserveAspectRatio="none" viewBox={`0 0 ${frame.width} ${frame.height}`}><polygon points={shownHints.map((point) => `${point.x + 0.5},${point.y + 0.5}`).join(' ')} /></svg> : null}
        </div>
        <aside className="viewport-controls">
          <div className="camera-readout sixdof-readout">
            {(['x', 'y', 'z', 'yaw', 'pitch', 'roll'] as const).map((key) => <NumberField disabled={busy || pending} key={key} label={`探索相机 ${key.toUpperCase()}`} onChange={(value) => updatePose(key, value)} value={pose[key]} />)}
          </div>
          <NumberField disabled={busy || pending} label="探索相机 FOV" max={120} min={10} onChange={(value) => updatePose('fov', value)} value={pose.fov} />
          <button disabled={busy || pending} onClick={() => { setPose(DEFAULT_POSE); frameAuthorityRef.current = null; setFrame(null); replaceFrameUrl(null); setHints([]) }} type="button">重置探索相机</button>
          <button disabled={busy || pending} onClick={() => void renderSelectionView()} type="button">更新地面选择视图</button>
        </aside>
      </div>
      <div className="camera-secondary-panel">
        <p>{shownHints.length}/3 个地面提示。P0 对应源锚帧相机在源地面上的垂直投影。重新生成视图会清除未拟合提示。</p>
        <div className="secondary-actions">
          <button disabled={busy || pending || hints.length === 0} onClick={() => setHints([])} type="button">重选三点</button>
          <button disabled={busy || pending || frame === null || hints.length !== 3} onClick={() => void fitGround()} type="button">自动寻找真正地面</button>
          <button disabled={busy || pending || targetGround === null || targetGround.confirmed} onClick={() => void confirmGround()} type="button">确认地面</button>
        </div>
        {targetGround === null ? null : <p className="technical-note">置信度 {(targetGround.confidence * 100).toFixed(1)}%，三邻域支持 {targetGround.support_counts.join(' / ')}，RMS {targetGround.rms_residual.toPrecision(3)}。</p>}
      </div>
      <div className="camera-secondary-panel">
        <h4>轨迹映射</h4>
        <div className="secondary-actions">
          <label>GS 比例<input aria-label="GS 比例" disabled={busy || pending} min="0.001" max="1000" onChange={(event) => setScaleText(event.currentTarget.value)} step="0.01" type="number" value={scaleText} /></label>
          <label>场景方位角<input aria-label="场景方位角" disabled={busy || pending} min="-180" max="179.999" onChange={(event) => setAzimuthText(event.currentTarget.value)} step="1" type="number" value={azimuthText} /></label>
          <button disabled={busy || pending || targetGround?.confirmed !== true} onClick={() => void saveAlignment()} type="button">保存轨迹对齐</button>
        </div>
      </div>
    </section>
  )
}
