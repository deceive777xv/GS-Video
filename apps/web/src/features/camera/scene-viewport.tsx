import {
  type PointerEvent as ReactPointerEvent,
  useEffect,
  useRef,
  useState,
} from 'react'

import type { BackendClient } from '../../api/backend-client'
import { BackendClientError } from '../../api/http-backend-client'
import type {
  CameraInput,
  FootPointDto,
  PreviewDto,
  PreviewFrameDto,
  ProjectDto,
} from '../../api/types'
import {
  ImagePointFields,
  ImagePointMarker,
  useImagePointDraft,
} from '../coordinates/image-point-editor'
import { toImagePoint } from '../coordinates/image-point'
import { nextPreviewGeneration } from './preview-generation'

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

interface SceneViewportProps {
  backend: BackendClient
  camera: CameraInput
  expectedProjectId?: string
  initialPreview?: PreviewDto | null
  confirmedCameraRevision?: number | null
  confirmedPreviewArtifactId?: string | null
  initialFootPoint?: FootPointDto | null
  canConfirm?: boolean
  onError(value: unknown): void
  onPreview(frame: PreviewFrameDto, camera: CameraInput): void
  onProjectChange?(project: ProjectDto): void
  onFootPoint?(footPoint: FootPointDto): void
  onAuthorityStale?(): Promise<void> | void
}

export function SceneViewport({
  backend,
  camera: initialCamera,
  expectedProjectId = 'unbound-project',
  initialPreview = null,
  confirmedCameraRevision = null,
  confirmedPreviewArtifactId = null,
  initialFootPoint = null,
  canConfirm = true,
  onError,
  onPreview,
  onProjectChange,
  onFootPoint,
  onAuthorityStale,
}: SceneViewportProps) {
  const [camera, setCamera] = useState(() => toCameraInput(initialCamera))
  const [frame, setFrame] = useState<PreviewFrameDto | null>(null)
  const [frameCameraFingerprint, setFrameCameraFingerprint] = useState<string | null>(null)
  const [frameUrl, setFrameUrl] = useState<string | null>(null)
  const [displayedCameraFingerprint, setDisplayedCameraFingerprint] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [authoritativeTrigger, setAuthoritativeTrigger] = useState(0)
  const [yawInput, setYawInput] = useState(() => formatAngleInput(initialCamera.yaw))
  const [pitchInput, setPitchInput] = useState(() => formatAngleInput(initialCamera.pitch))
  const [pickingFootPoint, setPickingFootPoint] = useState(false)
  const [rejectedPickBufferAuthority, setRejectedPickBufferAuthority] = useState<string | null>(null)
  const [rejectedConfirmationAuthority, setRejectedConfirmationAuthority] = useState<string | null>(null)
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
  const frameImageRef = useRef<HTMLImageElement>(null)
  const pickingFootPointRef = useRef(false)
  const footPickRequestId = useRef(0)
  const firstRender = useRef(initialPreview !== null)
  const onErrorRef = useRef(onError)
  const onPreviewRef = useRef(onPreview)
  const initialFingerprint = cameraFingerprint(initialCamera)
  const liveSequence = useRef(1)
  const latestLiveTarget = useRef({
    sequence: 1,
    fingerprint: initialFingerprint,
    camera: toCameraInput(initialCamera),
  })
  const liveAttemptedSequence = useRef(0)
  const liveRequestId = useRef(0)
  const liveDisplayRequestId = useRef(0)
  const liveInFlight = useRef(false)
  const livePump = useRef<() => void>(() => undefined)
  const livePumpEpoch = useRef(0)
  const liveDisabled = useRef(false)
  const forceAuthoritativeSequence = useRef<number | null>(null)
  const pendingLiveClose = useRef<{
    backend: BackendClient
    timeout: ReturnType<typeof setTimeout>
  } | null>(null)
  const authoritativeDisplaySequence = useRef(0)
  const frameMatchesCamera = frame !== null
    && frameCameraFingerprint === cameraFingerprint(camera)
  const frameTuple = frame === null
    ? null
    : `${frame.artifact_id}:${frame.camera_revision}:${frame.pick_buffer_revision}`
  const currentCameraFingerprint = cameraFingerprint(camera)
  const frameReady = frame !== null
    && frameUrl !== null
    && frameMatchesCamera
    && displayedCameraFingerprint === currentCameraFingerprint
  const confirmationAuthority = `${frameTuple ?? 'none'}:${confirmedCameraRevision ?? 'none'}:${confirmedPreviewArtifactId ?? 'none'}`
  const frameUsable = frameReady && rejectedPickBufferAuthority !== frameTuple
  const frameConfirmed = frameUsable
    && confirmedCameraRevision === frame.camera_revision
    && confirmedPreviewArtifactId === frame.artifact_id
    && rejectedConfirmationAuthority !== confirmationAuthority
  const frameAuthority = frameTuple === null
    ? 'no-authoritative-frame'
    : [
        frameTuple,
        frameReady ? 'ready' : `stale:${cameraFingerprint(camera)}`,
        frameConfirmed ? 'confirmed' : 'unconfirmed',
        rejectedPickBufferAuthority === frameTuple ? 'pick-buffer-rejected' : 'pick-buffer-usable',
        rejectedConfirmationAuthority === confirmationAuthority ? 'confirmation-rejected' : 'confirmation-usable',
      ].join(':')
  const restoredFootPoint = frame !== null
    && frameConfirmed
    && initialFootPoint?.preview_artifact_id === frame.artifact_id
    && initialFootPoint.camera_revision === frame.camera_revision
    && initialFootPoint.pick_buffer_revision === frame.pick_buffer_revision
    ? { x: initialFootPoint.image[0], y: initialFootPoint.image[1] }
    : null
  const footDraft = useImagePointDraft(frame, frameAuthority, restoredFootPoint)
  const footInteractionAuthority = `${frameAuthority}:${footDraft.xText}:${footDraft.yText}`
  const latestFootInteractionAuthority = useRef(footInteractionAuthority)
  latestFootInteractionAuthority.current = footInteractionAuthority

  useEffect(() => { onErrorRef.current = onError }, [onError])
  useEffect(() => { onPreviewRef.current = onPreview }, [onPreview])
  useEffect(() => { setYawInput(formatAngleInput(camera.yaw)) }, [camera.yaw])
  useEffect(() => { setPitchInput(formatAngleInput(camera.pitch)) }, [camera.pitch])
  useEffect(() => {
    const viewport = viewportRef.current
    if (viewport === null) return
    const updateDistance = (event: WheelEvent): void => {
      event.preventDefault()
      if (pickingFootPointRef.current) return
      const factor = event.deltaY > 0 ? 1.08 : 0.92
      setCamera((current) => ({
        ...current,
        distance: Math.min(100, Math.max(0.1, current.distance * factor)),
      }))
    }
    viewport.addEventListener('wheel', updateDistance, { passive: false })
    return () => viewport.removeEventListener('wheel', updateDistance)
  }, [])

  const replaceFrameUrl = (next: string | null, fingerprint: string | null = null): void => {
    if (frameUrlRef.current !== null && frameUrlRef.current !== next) {
      URL.revokeObjectURL(frameUrlRef.current)
    }
    frameUrlRef.current = next
    setFrameUrl(next)
    setDisplayedCameraFingerprint(fingerprint)
  }

  useEffect(() => () => replaceFrameUrl(null), [])

  useEffect(() => {
    const pending = pendingLiveClose.current
    if (pending?.backend === backend) {
      clearTimeout(pending.timeout)
      pendingLiveClose.current = null
    }
    return () => {
      const closeLivePreview = backend.closeLivePreview?.bind(backend)
      if (closeLivePreview === undefined) return
      let timeout: ReturnType<typeof setTimeout>
      timeout = setTimeout(() => {
        if (pendingLiveClose.current?.timeout === timeout) {
          pendingLiveClose.current = null
        }
        void closeLivePreview().catch(() => undefined)
      }, 0)
      pendingLiveClose.current = { backend, timeout }
    }
  }, [backend])

  useEffect(() => {
    const controller = new AbortController()
    const epoch = ++livePumpEpoch.current
    liveAttemptedSequence.current = 0
    liveInFlight.current = false
    liveDisabled.current = false
    const renderLivePreview = backend.renderLivePreview?.bind(backend)
    if (renderLivePreview === undefined) return () => controller.abort()

    const pump = (): void => {
      if (controller.signal.aborted
        || epoch !== livePumpEpoch.current
        || liveDisabled.current
        || liveInFlight.current) return
      const target = latestLiveTarget.current
      if (target.sequence <= liveAttemptedSequence.current) return
      liveAttemptedSequence.current = target.sequence
      liveInFlight.current = true
      const requestId = Math.max(Date.now(), liveRequestId.current + 1)
      liveRequestId.current = requestId
      void renderLivePreview({
        expected_project_id: expectedProjectId,
        request_id: requestId,
        width: 960,
        height: 540,
        camera: toCameraInput(target.camera),
      }, controller.signal).then((blob) => {
        if (controller.signal.aborted
          || requestId <= liveDisplayRequestId.current
          || target.sequence !== latestLiveTarget.current.sequence
          || target.fingerprint !== latestLiveTarget.current.fingerprint
          || target.sequence <= authoritativeDisplaySequence.current) return
        liveDisplayRequestId.current = requestId
        replaceFrameUrl(URL.createObjectURL(blob), target.fingerprint)
      }).catch(() => {
        if (controller.signal.aborted || epoch !== livePumpEpoch.current) return
        if (target.sequence !== latestLiveTarget.current.sequence
          || target.fingerprint !== latestLiveTarget.current.fingerprint) return
        liveDisabled.current = true
        forceAuthoritativeSequence.current = latestLiveTarget.current.sequence
        onErrorRef.current('实时预览暂不可用，已切换到高质量预览。')
        setAuthoritativeTrigger((current) => current + 1)
      }).finally(() => {
        if (epoch !== livePumpEpoch.current) return
        liveInFlight.current = false
        pump()
      })
    }
    livePump.current = pump
    pump()
    return () => {
      controller.abort()
      if (epoch === livePumpEpoch.current) {
        if (livePump.current === pump) livePump.current = () => undefined
        liveInFlight.current = false
      }
    }
  }, [backend, expectedProjectId])

  useEffect(() => {
    const fingerprint = cameraFingerprint(camera)
    if (fingerprint !== latestLiveTarget.current.fingerprint) {
      liveSequence.current += 1
      latestLiveTarget.current = {
        sequence: liveSequence.current,
        fingerprint,
        camera: toCameraInput(camera),
      }
    }
    livePump.current()
  }, [camera])

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
        authoritativeDisplaySequence.current = Math.max(
          authoritativeDisplaySequence.current,
          latestLiveTarget.current.sequence,
        )
        replaceFrameUrl(URL.createObjectURL(blob), nextFingerprint)
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
    const authoritativeSequence = latestLiveTarget.current.sequence
    const renderImmediately = forceAuthoritativeSequence.current === authoritativeSequence
    if (renderImmediately) forceAuthoritativeSequence.current = null
    const timeout = setTimeout(() => {
      const nextGeneration = nextPreviewGeneration(generation.current)
      generation.current = nextGeneration
      setLoading(true)
      void backend.renderPreview({
        expected_project_id: expectedProjectId,
        generation: nextGeneration,
        width: 960,
        height: 540,
        camera: toCameraInput(camera),
      }, controller.signal).then(async (nextFrame) => {
        const blob = await backend.fetchPreviewArtifact(nextFrame.artifact_id, controller.signal)
        if (controller.signal.aborted || authority !== requestAuthority.current) return
        authoritativeDisplaySequence.current = Math.max(
          authoritativeDisplaySequence.current,
          authoritativeSequence,
        )
        replaceFrameUrl(URL.createObjectURL(blob), cameraFingerprint(camera))
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
    }, renderImmediately ? 0 : 120)
    return () => {
      clearTimeout(timeout)
      controller.abort()
    }
  }, [authoritativeTrigger, backend, camera, expectedProjectId])

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
    if (pickingFootPointRef.current) return
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
    if (pickingFootPointRef.current || drag.current === null) return
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
    if (pickingFootPointRef.current || drag.current !== null || frame === null || !frameUsable || frameImageRef.current === null) return
    const mapped = toImagePoint(event, frameImageRef.current.getBoundingClientRect(), frame)
    if (mapped === null) {
      onError('点击位于预览内容之外，请在图像范围内选择落脚点。')
      return
    }
    footDraft.select(mapped)
  }

  const submitFootPoint = async (): Promise<void> => {
    if (pickingFootPointRef.current) return
    if (frame === null || !frameUsable) {
      onError('请先等待最新场景预览。')
      return
    }
    if (!frameConfirmed) {
      onError('请先确认当前初始机位。')
      return
    }
    const point = footDraft.point
    if (point === null) {
      onError('落脚点坐标必须位于当前预览图像内。')
      return
    }
    const requestId = ++footPickRequestId.current
    const requestAuthority = footInteractionAuthority
    const expectedFrameAuthority = frameTuple
    pickingFootPointRef.current = true
    setPickingFootPoint(true)
    try {
      const foot = await backend.pickFootPoint({
        x: point.x,
        y: point.y,
        preview_artifact_id: frame.artifact_id,
        camera_revision: frame.camera_revision,
        pick_buffer_revision: frame.pick_buffer_revision,
      })
      if (requestId !== footPickRequestId.current
        || latestFootInteractionAuthority.current !== requestAuthority) return
      if (foot.preview_artifact_id !== frame.artifact_id
        || foot.camera_revision !== frame.camera_revision
        || foot.pick_buffer_revision !== frame.pick_buffer_revision) {
        setRejectedPickBufferAuthority(expectedFrameAuthority)
        footDraft.accept(null)
        await onAuthorityStale?.()
        onError('落脚点响应不属于当前预览，已刷新场景状态。')
        return
      }
      footDraft.accept({ x: foot.image[0], y: foot.image[1] })
      onFootPoint?.(foot)
    } catch (error) {
      const authorityIsCurrent = requestId === footPickRequestId.current
        && latestFootInteractionAuthority.current === requestAuthority
      if (authorityIsCurrent
        && error instanceof BackendClientError
        && (error.code === 'stale_pick_buffer' || error.code === 'camera_not_confirmed')) {
        if (error.code === 'stale_pick_buffer') {
          setRejectedPickBufferAuthority(expectedFrameAuthority)
        } else {
          setRejectedConfirmationAuthority(confirmationAuthority)
        }
        footDraft.accept(null)
        await onAuthorityStale?.()
      }
      if (authorityIsCurrent) {
        onError(error instanceof Error ? error : '落脚点深度无效或预览已经过期。')
      }
    } finally {
      if (requestId === footPickRequestId.current) {
        pickingFootPointRef.current = false
        setPickingFootPoint(false)
      }
    }
  }

  const confirmCamera = async (): Promise<void> => {
    if (frame === null || !frameUsable) {
      onError('请先等待最新场景预览。')
      return
    }
    try {
      const project = await backend.confirmCamera(expectedProjectId, frame.camera_revision)
      setRejectedConfirmationAuthority(null)
      onProjectChange?.(project)
    } catch (error) {
      onError(error instanceof Error ? error : '初始机位确认失败。')
    }
  }

  return (
    <div className="viewport-layout">
      <div
        aria-label="Gaussian 场景视口"
        className="viewport-frame preview-surface"
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
          <img alt="最新 Gaussian 场景后端预览" draggable={false} ref={frameImageRef} src={frameUrl} />
        )}
        <ImagePointMarker containerRef={viewportRef} image={frame} mediaRef={frameImageRef} pending={footDraft.dirty} point={footDraft.point} />
        {loading ? <span className="viewport-loading">更新视图…</span> : null}
      </div>
      <aside className="viewport-controls" aria-label="机位控制">
        <div className="camera-readout">
          <label>
            Yaw 角度
            <input
              aria-label="Yaw 角度"
              disabled={pickingFootPoint}
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
              disabled={pickingFootPoint}
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
            disabled={pickingFootPoint}
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
        <button disabled={!canConfirm || !frameUsable || loading || pickingFootPoint} onClick={() => void confirmCamera()} type="button">
          确认初始机位
        </button>
        <fieldset>
          <legend>场景落脚点</legend>
          <ImagePointFields
            confirmedLabel="场景落脚点已验证"
            disabled={loading || pickingFootPoint || !frameUsable}
            draft={footDraft}
            emptyLabel="尚未选择场景落脚点"
            image={frame}
            pendingLabel="候选场景落脚点待确认"
            xLabel="落脚点 X 坐标"
            yLabel="落脚点 Y 坐标"
          />
          <button disabled={!frameConfirmed || footDraft.point === null || loading || pickingFootPoint} onClick={() => void submitFootPoint()} type="button">
            {pickingFootPoint ? '验证中…' : '确认场景落脚点'}
          </button>
        </fieldset>
      </aside>
    </div>
  )
}
