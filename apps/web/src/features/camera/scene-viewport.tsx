import {
  type PointerEvent as ReactPointerEvent,
  useEffect,
  useRef,
  useState,
} from 'react'

import type { BackendClient } from '../../api/backend-client'
import type {
  CameraInput,
  FootPointDto,
  PreviewDto,
  PreviewFrameDto,
  ProjectDto,
} from '../../api/types'
import { nextPreviewGeneration } from './preview-generation'

export interface ImageSize { width: number; height: number }
export interface Point { x: number; y: number }
export interface PointEvent { clientX: number; clientY: number }
export interface PointBounds { left: number; top: number; width: number; height: number }

export function toCameraInput(camera: CameraInput): CameraInput {
  return {
    target: [...camera.target],
    distance: camera.distance,
    yaw: camera.yaw,
    pitch: camera.pitch,
    fov_y_degrees: camera.fov_y_degrees,
  }
}

export function cameraFingerprint(camera: CameraInput): string {
  return [
    ...camera.target,
    camera.distance,
    camera.yaw,
    camera.pitch,
    camera.fov_y_degrees,
  ].map((value) => Object.is(value, -0) ? '0' : String(value)).join('|')
}

function formatAngleInput(value: number): string {
  return String(Number(value.toFixed(3)))
}

function normalizeYaw(value: number): number {
  const normalized = ((value + 180) % 360 + 360) % 360 - 180
  return Object.is(normalized, -0) ? 0 : normalized
}

export function toImagePoint(
  event: PointEvent,
  bounds: PointBounds,
  image: ImageSize,
): Point | null {
  if (bounds.width <= 0 || bounds.height <= 0 || image.width <= 0 || image.height <= 0) return null
  const scale = Math.min(bounds.width / image.width, bounds.height / image.height)
  const left = bounds.left + (bounds.width - image.width * scale) / 2
  const top = bounds.top + (bounds.height - image.height * scale) / 2
  const x = (event.clientX - left) / scale
  const y = (event.clientY - top) / scale
  if (x < 0 || y < 0 || x >= image.width || y >= image.height) return null
  return { x: Math.floor(x), y: Math.floor(y) }
}

interface SceneViewportProps {
  backend: BackendClient
  camera: CameraInput
  initialPreview?: PreviewDto | null
  canConfirm?: boolean
  onError(value: unknown): void
  onPreview(frame: PreviewFrameDto, camera: CameraInput): void
  onProjectChange?(project: ProjectDto): void
  onFootPoint?(footPoint: FootPointDto): void
}

export function SceneViewport({
  backend,
  camera: initialCamera,
  initialPreview = null,
  canConfirm = true,
  onError,
  onPreview,
  onProjectChange,
  onFootPoint,
}: SceneViewportProps) {
  const [camera, setCamera] = useState(() => toCameraInput(initialCamera))
  const [frame, setFrame] = useState<PreviewFrameDto | null>(initialPreview === null ? null : {
    artifact_id: initialPreview.artifact_id,
    generation: initialPreview.generation,
    width: initialPreview.width,
    height: initialPreview.height,
    camera_revision: initialPreview.camera_revision,
    pick_buffer_revision: initialPreview.pick_buffer_revision,
  })
  const [frameCameraFingerprint, setFrameCameraFingerprint] = useState<string | null>(
    initialPreview === null ? null : cameraFingerprint(initialCamera),
  )
  const [frameUrl, setFrameUrl] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [yawInput, setYawInput] = useState(() => formatAngleInput(initialCamera.yaw))
  const [pitchInput, setPitchInput] = useState(() => formatAngleInput(initialCamera.pitch))
  const [footX, setFootX] = useState('')
  const [footY, setFootY] = useState('')
  const generation = useRef(initialPreview?.generation ?? 0)
  const requestAuthority = useRef(0)
  const drag = useRef<{
    originX: number
    originY: number
    lastX: number
    lastY: number
    moved: boolean
  } | null>(null)
  const suppressNextClick = useRef(false)
  const frameUrlRef = useRef<string | null>(null)
  const viewportRef = useRef<HTMLDivElement>(null)
  const firstRender = useRef(initialPreview !== null)
  const onErrorRef = useRef(onError)
  const onPreviewRef = useRef(onPreview)

  useEffect(() => { onErrorRef.current = onError }, [onError])
  useEffect(() => { onPreviewRef.current = onPreview }, [onPreview])
  useEffect(() => { setYawInput(formatAngleInput(camera.yaw)) }, [camera.yaw])
  useEffect(() => { setPitchInput(formatAngleInput(camera.pitch)) }, [camera.pitch])
  useEffect(() => {
    const viewport = viewportRef.current
    if (viewport === null) return
    const updateDistance = (event: WheelEvent): void => {
      event.preventDefault()
      const factor = event.deltaY > 0 ? 1.08 : 0.92
      setCamera((current) => ({
        ...current,
        distance: Math.min(100, Math.max(0.1, current.distance * factor)),
      }))
    }
    viewport.addEventListener('wheel', updateDistance, { passive: false })
    return () => viewport.removeEventListener('wheel', updateDistance)
  }, [])

  const replaceFrameUrl = (next: string | null): void => {
    if (frameUrlRef.current !== null && frameUrlRef.current !== next) {
      URL.revokeObjectURL(frameUrlRef.current)
    }
    frameUrlRef.current = next
    setFrameUrl(next)
  }

  useEffect(() => () => replaceFrameUrl(null), [])

  useEffect(() => {
    setCamera((current) => (
      cameraFingerprint(current) === cameraFingerprint(initialCamera)
        ? current
        : toCameraInput(initialCamera)
    ))
    requestAuthority.current += 1
    setFrame(null)
    setFrameCameraFingerprint(null)
    replaceFrameUrl(null)
    if (initialPreview === null) {
      firstRender.current = false
      return
    }
    generation.current = Math.max(generation.current, initialPreview.generation)
    firstRender.current = true
  }, [
    initialCamera.distance,
    initialCamera.fov_y_degrees,
    initialCamera.pitch,
    initialCamera.target[0],
    initialCamera.target[1],
    initialCamera.target[2],
    initialCamera.yaw,
    initialPreview?.artifact_id,
    initialPreview?.generation,
  ])

  useEffect(() => {
    if (initialPreview === null) return
    const controller = new AbortController()
    const authority = ++requestAuthority.current
    const nextFrame = {
      artifact_id: initialPreview.artifact_id,
      generation: initialPreview.generation,
      width: initialPreview.width,
      height: initialPreview.height,
      camera_revision: initialPreview.camera_revision,
      pick_buffer_revision: initialPreview.pick_buffer_revision,
    }
    const nextFingerprint = cameraFingerprint(initialCamera)
    void backend.fetchPreviewArtifact(initialPreview.artifact_id, controller.signal)
      .then((blob) => {
        if (controller.signal.aborted || authority !== requestAuthority.current) return
        firstRender.current = false
        replaceFrameUrl(URL.createObjectURL(blob))
        setFrame(nextFrame)
        setFrameCameraFingerprint(nextFingerprint)
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted) onErrorRef.current(error instanceof Error ? error : '无法载入场景预览。')
      })
    return () => controller.abort()
  }, [
    backend,
    initialCamera.distance,
    initialCamera.fov_y_degrees,
    initialCamera.pitch,
    initialCamera.target[0],
    initialCamera.target[1],
    initialCamera.target[2],
    initialCamera.yaw,
    initialPreview?.artifact_id,
    initialPreview?.camera_revision,
    initialPreview?.generation,
    initialPreview?.height,
    initialPreview?.pick_buffer_revision,
    initialPreview?.width,
  ])

  useEffect(() => {
    if (firstRender.current) {
      firstRender.current = false
      return
    }
    const controller = new AbortController()
    const authority = ++requestAuthority.current
    const timeout = setTimeout(() => {
      const nextGeneration = nextPreviewGeneration(generation.current)
      generation.current = nextGeneration
      setLoading(true)
      void backend.renderPreview({
        generation: nextGeneration,
        width: 960,
        height: 540,
        camera: toCameraInput(camera),
      }, controller.signal).then(async (nextFrame) => {
        const blob = await backend.fetchPreviewArtifact(nextFrame.artifact_id, controller.signal)
        if (controller.signal.aborted || authority !== requestAuthority.current) return
        replaceFrameUrl(URL.createObjectURL(blob))
        setFrame(nextFrame)
        setFrameCameraFingerprint(cameraFingerprint(camera))
        onPreviewRef.current(nextFrame, camera)
      }).catch((error: unknown) => {
        if (!controller.signal.aborted && authority === requestAuthority.current) {
          onErrorRef.current(error instanceof Error ? error : '场景预览生成失败。')
        }
      }).finally(() => {
        if (authority === requestAuthority.current) setLoading(false)
      })
    }, 150)
    return () => {
      clearTimeout(timeout)
      controller.abort()
    }
  }, [backend, camera])

  const restoreAngleInput = (axis: 'yaw' | 'pitch'): void => {
    if (axis === 'yaw') setYawInput(formatAngleInput(camera.yaw))
    else setPitchInput(formatAngleInput(camera.pitch))
  }

  const commitAngleInput = (axis: 'yaw' | 'pitch'): void => {
    const input = axis === 'yaw' ? yawInput : pitchInput
    const parsed = input.trim() === '' ? Number.NaN : Number(input)
    if (!Number.isFinite(parsed)) {
      restoreAngleInput(axis)
      return
    }
    const value = axis === 'yaw'
      ? normalizeYaw(parsed)
      : Math.min(89, Math.max(-89, parsed))
    if (axis === 'yaw') setYawInput(formatAngleInput(value))
    else setPitchInput(formatAngleInput(value))
    setCamera((current) => current[axis] === value ? current : { ...current, [axis]: value })
  }

  const pointerDown = (event: ReactPointerEvent<HTMLDivElement>): void => {
    drag.current = {
      originX: event.clientX,
      originY: event.clientY,
      lastX: event.clientX,
      lastY: event.clientY,
      moved: false,
    }
    suppressNextClick.current = false
    event.currentTarget.setPointerCapture?.(event.pointerId)
  }
  const pointerMove = (event: ReactPointerEvent<HTMLDivElement>): void => {
    if (drag.current === null) return
    const dx = event.clientX - drag.current.lastX
    const dy = event.clientY - drag.current.lastY
    drag.current = {
      ...drag.current,
      lastX: event.clientX,
      lastY: event.clientY,
      moved: drag.current.moved || Math.hypot(
        event.clientX - drag.current.originX,
        event.clientY - drag.current.originY,
      ) >= 3,
    }
    setCamera((current) => ({
      ...current,
      yaw: current.yaw + dx * 0.25,
      pitch: Math.min(89, Math.max(-89, current.pitch + dy * 0.25)),
    }))
  }
  const pointerUp = (): void => {
    suppressNextClick.current = drag.current?.moved ?? false
    drag.current = null
  }

  const chooseFootPoint = (event: ReactPointerEvent<HTMLDivElement>): void => {
    if (suppressNextClick.current) {
      suppressNextClick.current = false
      return
    }
    if (drag.current !== null || frame === null || !frameMatchesCamera || viewportRef.current === null) return
    const mapped = toImagePoint(event, viewportRef.current.getBoundingClientRect(), frame)
    if (mapped === null) {
      onError('点击位于预览内容之外，请在图像范围内选择落脚点。')
      return
    }
    setFootX(String(mapped.x))
    setFootY(String(mapped.y))
    void submitFootPoint(mapped)
  }

  const submitFootPoint = async (override?: Point): Promise<void> => {
    if (frame === null || !frameMatchesCamera) {
      onError('请先等待最新场景预览。')
      return
    }
    const point = override ?? { x: Number(footX), y: Number(footY) }
    if (!Number.isInteger(point.x) || !Number.isInteger(point.y)
      || point.x < 0 || point.y < 0 || point.x >= frame.width || point.y >= frame.height) {
      onError('落脚点坐标必须位于当前预览图像内。')
      return
    }
    try {
      const foot = await backend.pickFootPoint({
        x: point.x,
        y: point.y,
        preview_artifact_id: frame.artifact_id,
        camera_revision: frame.camera_revision,
        pick_buffer_revision: frame.pick_buffer_revision,
      })
      onFootPoint?.(foot)
    } catch (error) {
      onError(error instanceof Error ? error : '落脚点深度无效或预览已经过期。')
    }
  }

  const confirmCamera = async (): Promise<void> => {
    if (frame === null || !frameMatchesCamera) {
      onError('请先等待最新场景预览。')
      return
    }
    try {
      onProjectChange?.(await backend.confirmCamera(frame.camera_revision))
    } catch (error) {
      onError(error instanceof Error ? error : '初始机位确认失败。')
    }
  }

  const frameMatchesCamera = frame !== null
    && frameCameraFingerprint === cameraFingerprint(camera)

  return (
    <div className="viewport-layout">
      <div
        aria-label="Gaussian 场景视口"
        className="viewport-frame"
        onClick={chooseFootPoint}
        onPointerDown={pointerDown}
        onPointerMove={pointerMove}
        onPointerUp={pointerUp}
        onPointerCancel={pointerUp}
        ref={viewportRef}
        tabIndex={0}
      >
        {frameUrl === null ? (
          <div className="viewport-empty"><span>正在准备 Gaussian 场景帧</span></div>
        ) : (
          <img alt="最新 Gaussian 场景后端预览" draggable={false} src={frameUrl} />
        )}
        {loading ? <span className="viewport-loading">更新视图…</span> : null}
      </div>
      <aside className="viewport-controls" aria-label="机位控制">
        <div className="camera-readout">
          <label>
            Yaw 角度
            <input
              aria-label="Yaw 角度"
              inputMode="decimal"
              onBlur={() => commitAngleInput('yaw')}
              onChange={(event) => setYawInput(event.currentTarget.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter') {
                  event.preventDefault()
                  commitAngleInput('yaw')
                } else if (event.key === 'Escape') {
                  restoreAngleInput('yaw')
                }
              }}
              step="0.1"
              type="number"
              value={yawInput}
            />
          </label>
          <label>
            Pitch 角度
            <input
              aria-label="Pitch 角度"
              inputMode="decimal"
              max="89"
              min="-89"
              onBlur={() => commitAngleInput('pitch')}
              onChange={(event) => setPitchInput(event.currentTarget.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter') {
                  event.preventDefault()
                  commitAngleInput('pitch')
                } else if (event.key === 'Escape') {
                  restoreAngleInput('pitch')
                }
              }}
              step="0.1"
              type="number"
              value={pitchInput}
            />
          </label>
          <span>距离 {camera.distance.toFixed(2)}</span>
        </div>
        <label>
          垂直视场角
          <input
            aria-label="垂直视场角"
            max="100"
            min="20"
            onChange={(event) => {
              const value = Number(event.currentTarget.value)
              if (!Number.isFinite(value)) return
              const fov = Math.min(100, Math.max(20, value))
              setCamera((current) => ({ ...current, fov_y_degrees: fov }))
            }}
            onKeyDown={(event) => {
              if (!['ArrowLeft', 'ArrowDown', 'ArrowRight', 'ArrowUp', 'Home', 'End'].includes(event.key)) return
              event.preventDefault()
              setCamera((current) => {
                const delta = ['ArrowRight', 'ArrowUp'].includes(event.key) ? 1 : -1
                const next = event.key === 'Home' ? 20 : event.key === 'End' ? 100 : current.fov_y_degrees + delta
                return { ...current, fov_y_degrees: Math.min(100, Math.max(20, next)) }
              })
            }}
            step="1"
            type="range"
            value={camera.fov_y_degrees}
          />
          <output>{camera.fov_y_degrees.toFixed(0)}°</output>
        </label>
        <button disabled={!canConfirm || !frameMatchesCamera || loading} onClick={() => void confirmCamera()} type="button">
          确认初始机位
        </button>
        <fieldset>
          <legend>无鼠标落脚点输入</legend>
          <label>落脚点 X 坐标<input aria-label="落脚点 X 坐标" inputMode="numeric" onChange={(event) => setFootX(event.currentTarget.value)} value={footX} /></label>
          <label>落脚点 Y 坐标<input aria-label="落脚点 Y 坐标" inputMode="numeric" onChange={(event) => setFootY(event.currentTarget.value)} value={footY} /></label>
          <button disabled={!frameMatchesCamera} onClick={() => void submitFootPoint()} type="button">确认场景落脚点</button>
        </fieldset>
      </aside>
    </div>
  )
}
