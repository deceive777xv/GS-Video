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

function poseFingerprint(pose: EulerPose): string {
  return JSON.stringify(cameraInput(pose))
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
  const [liveUrl, setLiveUrl] = useState<string | null>(null)
  const [frameFingerprint, setFrameFingerprint] = useState<string | null>(null)
  const [settling, setSettling] = useState(false)
  const [hints, setHints] = useState<ImagePoint[]>([])
  const [pending, setPending] = useState(false)
  const viewportRef = useRef<HTMLDivElement>(null)
  const generation = useRef(project.workflow.preview?.generation ?? 0)
  const frameUrlRef = useRef<string | null>(null)
  const liveUrlRef = useRef<string | null>(null)
  const frameAuthorityRef = useRef<string | null>(null)
  const previewRequest = useRef(0)
  const automaticRequest = useRef(0)
  const dragging = useRef<{ pointerId: number; x: number; y: number } | null>(null)
  const onErrorRef = useRef(onError)
  const onProjectChangeRef = useRef(onProjectChange)
  const onRefreshRef = useRef(onRefresh)
  onErrorRef.current = onError
  onProjectChangeRef.current = onProjectChange
  onRefreshRef.current = onRefresh
  const authoritativePreview = project.workflow.preview
  const authoritativePreviewKey = previewAuthority(project.project_id, authoritativePreview)
  const targetGround = project.workflow.target_ground
  const fingerprint = poseFingerprint(pose)
  const frameMatchesPose = frame !== null && frameFingerprint === fingerprint
  const candidateMatchesFrame = frame !== null
    && frameMatchesPose
    && targetGround !== null
    && targetGround.preview_artifact_id === frame.artifact_id
    && targetGround.camera_revision === frame.camera_revision
    && targetGround.pick_buffer_revision === frame.pick_buffer_revision

  const replaceFrameUrl = (next: string | null): void => {
    if (frameUrlRef.current !== null) URL.revokeObjectURL(frameUrlRef.current)
    frameUrlRef.current = next
    setFrameUrl(next)
  }
  const replaceLiveUrl = (next: string | null): void => {
    if (liveUrlRef.current !== null) URL.revokeObjectURL(liveUrlRef.current)
    liveUrlRef.current = next
    setLiveUrl(next)
  }
  useEffect(() => {
    generation.current = authoritativePreview?.generation ?? 0
    const request = ++previewRequest.current
    if (authoritativePreview === null || authoritativePreviewKey === null) {
      frameAuthorityRef.current = null
      setFrame(null)
      setFrameFingerprint(null)
      replaceFrameUrl(null)
      return
    }
    if (
      frameAuthorityRef.current === authoritativePreviewKey
      && frameUrlRef.current !== null
    ) return

    frameAuthorityRef.current = null
    setFrame(null)
    setFrameFingerprint(null)
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
      setFrameFingerprint(poseFingerprint(matrixPose(project.workflow.exploration_camera)))
      replaceFrameUrl(nextUrl)
    }).catch((error: unknown) => {
      if (controller.signal.aborted || previewRequest.current !== request) return
      onErrorRef.current(error instanceof Error ? error : '无法载入当前 Gaussian 场景预览。')
    })
    return () => controller.abort()
  }, [authoritativePreviewKey, backend, project.workflow.exploration_camera])
  useEffect(() => () => {
    automaticRequest.current += 1
    previewRequest.current += 1
    frameAuthorityRef.current = null
    replaceLiveUrl(null)
    replaceFrameUrl(null)
  }, [])

  const changePose = (updater: (current: EulerPose) => EulerPose): void => {
    setPose(updater)
    setHints([])
  }

  const updatePose = (key: keyof EulerPose, value: number): void => {
    changePose((current) => ({ ...current, [key]: value }))
  }

  const moveLocal = (right: number, down: number, forward: number): void => {
    changePose((current) => {
      const matrix = poseMatrix(current)
      return {
        ...current,
        x: current.x + matrix[0][0] * right + matrix[0][1] * down + matrix[0][2] * forward,
        y: current.y + matrix[1][0] * right + matrix[1][1] * down + matrix[1][2] * forward,
        z: current.z + matrix[2][0] * right + matrix[2][1] * down + matrix[2][2] * forward,
      }
    })
  }

  useEffect(() => {
    if (busy || pending || frameMatchesPose || (authoritativePreview !== null && frame === null)) return
    const request = ++automaticRequest.current
    generation.current += 1
    const requestedGeneration = generation.current
    const requestedCamera = cameraInput(pose)
    const liveController = new AbortController()
    const settledController = new AbortController()
    setSettling(true)
    const liveTimer = window.setTimeout(() => {
      void backend.renderLivePreview({
        expected_project_id: project.project_id,
        request_id: request,
        width: 960,
        height: 540,
        camera: requestedCamera,
      }, liveController.signal).then((blob) => {
        if (liveController.signal.aborted || automaticRequest.current !== request) return
        replaceLiveUrl(URL.createObjectURL(blob))
      }).catch((error: unknown) => {
        if (liveController.signal.aborted || automaticRequest.current !== request) return
        onErrorRef.current(error)
      })
    }, 40)
    const settledTimer = window.setTimeout(() => {
      void backend.renderPreview({
        expected_project_id: project.project_id,
        generation: requestedGeneration,
        width: 960,
        height: 540,
        camera: requestedCamera,
      }, settledController.signal).then(async (rendered) => {
        const blob = await backend.fetchPreviewArtifact(rendered.artifact_id, settledController.signal)
        if (settledController.signal.aborted || automaticRequest.current !== request) return
        const nextUrl = URL.createObjectURL(blob)
        if (settledController.signal.aborted || automaticRequest.current !== request) {
          URL.revokeObjectURL(nextUrl)
          return
        }
        frameAuthorityRef.current = previewAuthority(project.project_id, rendered)
        setFrame(rendered)
        setFrameFingerprint(fingerprint)
        replaceFrameUrl(nextUrl)
        replaceLiveUrl(null)
        setHints([])
        const refreshed = await onRefreshRef.current()
        if (settledController.signal.aborted || automaticRequest.current !== request) return
        onProjectChangeRef.current(refreshed)
      }).catch((error: unknown) => {
        if (settledController.signal.aborted || automaticRequest.current !== request) return
        onErrorRef.current(error)
      }).finally(() => {
        if (automaticRequest.current === request) setSettling(false)
      })
    }, 140)
    return () => {
      window.clearTimeout(liveTimer)
      window.clearTimeout(settledTimer)
      liveController.abort()
      settledController.abort()
    }
  }, [authoritativePreview, backend, busy, fingerprint, frame, frameMatchesPose, pending, pose, project.project_id])

  useEffect(() => {
    const viewport = viewportRef.current
    if (viewport === null) return
    const onWheel = (event: WheelEvent): void => {
      event.preventDefault()
      event.stopPropagation()
      if (!busy && !pending) moveLocal(0, 0, event.deltaY > 0 ? -0.25 : 0.25)
    }
    viewport.addEventListener('wheel', onWheel, { passive: false })
    return () => viewport.removeEventListener('wheel', onWheel)
  })

  const addHint = (event: ReactPointerEvent<HTMLDivElement>): void => {
    if (event.button !== 0 || !frameMatchesPose || frame === null || viewportRef.current === null || hints.length >= 3 || pending || busy) return
    const point = toImagePoint(event, viewportRef.current.getBoundingClientRect(), frame)
    if (point === null || hints.some((item) => item.x === point.x && item.y === point.y)) return
    setHints((current) => [...current, point])
  }

  const fitGround = async (): Promise<void> => {
    if (!frameMatchesPose || frame === null || hints.length !== 3 || pending || busy) return
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
          onPointerDown={(event) => {
            if (event.button === 1 && !busy && !pending) {
              event.preventDefault()
              event.currentTarget.setPointerCapture?.(event.pointerId)
              dragging.current = { pointerId: event.pointerId, x: event.clientX, y: event.clientY }
              return
            }
            addHint(event)
          }}
          onPointerMove={(event) => {
            const drag = dragging.current
            if (drag === null || drag.pointerId !== event.pointerId || busy || pending) return
            const dx = event.clientX - drag.x
            const dy = event.clientY - drag.y
            dragging.current = { ...drag, x: event.clientX, y: event.clientY }
            if (event.shiftKey) moveLocal(-dx * 0.01, -dy * 0.01, 0)
            else changePose((current) => ({
              ...current,
              yaw: current.yaw + dx * 0.2,
              pitch: Math.max(-89, Math.min(89, current.pitch - dy * 0.2)),
            }))
          }}
          onPointerUp={(event) => {
            if (dragging.current?.pointerId === event.pointerId) dragging.current = null
          }}
          onPointerCancel={() => { dragging.current = null }}
          ref={viewportRef}
        >
          {(liveUrl ?? frameUrl) === null ? <div className="viewport-empty">正在准备 Gaussian 实时预览…</div> : <img alt="Gaussian 地面选择视图" draggable={false} src={(liveUrl ?? frameUrl) ?? ''} />}
          {!frameMatchesPose || frame === null ? null : shownHints.map((point, index) => <Marker frame={frame} key={`${point.x}:${point.y}:${index}`} label={`H${index}`} point={point} />)}
          {frameMatchesPose && frame !== null && shownHints.length === 3 ? <svg aria-label="自动地面候选范围" className="ground-plane-overlay" preserveAspectRatio="none" viewBox={`0 0 ${frame.width} ${frame.height}`}><polygon points={shownHints.map((point) => `${point.x + 0.5},${point.y + 0.5}`).join(' ')} /></svg> : null}
          {settling ? <span className="viewport-status">交互预览 · 停止后自动启用地面拾取</span> : null}
        </div>
        <aside className="viewport-controls">
          <div className="camera-readout sixdof-readout">
            {(['x', 'y', 'z', 'yaw', 'pitch', 'roll'] as const).map((key) => <NumberField disabled={busy || pending} key={key} label={`探索相机 ${key.toUpperCase()}`} onChange={(value) => updatePose(key, value)} value={pose[key]} />)}
          </div>
          <NumberField disabled={busy || pending} label="探索相机 FOV" max={120} min={10} onChange={(value) => updatePose('fov', value)} value={pose.fov} />
          <button disabled={busy || pending} onClick={() => changePose(() => DEFAULT_POSE)} type="button">重置探索相机</button>
          <p className="technical-note">中键拖拽环视，Shift + 中键平移，滚轮前后移动；停止操作后会自动生成可拾取帧。</p>
        </aside>
      </div>
      <div className="camera-secondary-panel">
        <p>{shownHints.length}/3 个地面提示。P0 对应源锚帧相机在源地面上的垂直投影。重新生成视图会清除未拟合提示。</p>
        <div className="secondary-actions">
          <button disabled={busy || pending || hints.length === 0} onClick={() => setHints([])} type="button">重选三点</button>
          <button disabled={busy || pending || !frameMatchesPose || frame === null || hints.length !== 3} onClick={() => void fitGround()} type="button">自动寻找真正地面</button>
          <button disabled={busy || pending || targetGround === null || targetGround.confirmed} onClick={() => void confirmGround()} type="button">确认地面</button>
        </div>
        {targetGround === null ? null : <p className="technical-note">置信度 {(targetGround.confidence * 100).toFixed(1)}%，三邻域支持 {targetGround.support_counts.join(' / ')}，RMS {targetGround.rms_residual.toPrecision(3)}。</p>}
      </div>
    </section>
  )
}
