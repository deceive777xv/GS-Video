import { useEffect, useRef, useState } from 'react'

import type { BackendClient } from '../../api/backend-client'
import type { ProjectDto, TaskDto, TaskEvent } from '../../api/types'

interface PreviewPageProps {
  backend: BackendClient
  busy: boolean
  latestEvent: TaskEvent | null
  project: ProjectDto
  activeTask: TaskDto | null
  onError(value: unknown): void
  onProjectChange(project: ProjectDto): void
  onStartStage(stage: 'composite'): Promise<unknown>
  onBackToCamera(): void
  onReselectSubject(): void
}

export function PreviewPage({
  backend, busy, project, activeTask, latestEvent, onError, onProjectChange, onStartStage,
  onBackToCamera, onReselectSubject,
}: PreviewPageProps) {
  const [frameUrl, setFrameUrl] = useState<string | null>(null)
  const [compositeVideo, setCompositeVideo] = useState<{
    authority: string
    artifactId: string
    url: string
  } | null>(null)
  const [running, setRunning] = useState(false)
  const source = project.workflow.source_summary
  const authoritativeCrop = project.workflow.output_crop ?? (
    source == null ? null : { x: 0, y: 0, width: source.width, height: source.height }
  )
  const [cropText, setCropText] = useState(() => ({
    x: String(authoritativeCrop?.x ?? 0),
    y: String(authoritativeCrop?.y ?? 0),
    width: String(authoritativeCrop?.width ?? 1920),
    height: String(authoritativeCrop?.height ?? 1080),
  }))
  const frameObjectUrl = useRef<string | null>(null)
  const compositeObjectUrl = useRef<string | null>(null)
  const compositeRequest = useRef(0)
  const descriptorAuthority = useRef<string | null>(null)
  const preview = project.workflow.preview
  const composite = project.stages.composite
  const groundConfirmed = project.workflow.target_ground?.confirmed === true
  const compositeAuthority = composite?.status === 'succeeded'
    && composite.cache_key !== null
    && composite.cache_key !== undefined
    ? composite.cache_key
    : null
  const cropDraft = {
    x: Number(cropText.x), y: Number(cropText.y),
    width: Number(cropText.width), height: Number(cropText.height),
  }
  const cropValid = Number.isInteger(cropDraft.x) && Number.isInteger(cropDraft.y)
    && Number.isInteger(cropDraft.width) && cropDraft.width >= 2 && cropDraft.width <= 3840
    && cropDraft.width % 2 === 0
    && Number.isInteger(cropDraft.height) && cropDraft.height >= 2 && cropDraft.height <= 2160
    && cropDraft.height % 2 === 0
  const cropSaved = authoritativeCrop !== null && cropValid
    && cropDraft.x === authoritativeCrop.x && cropDraft.y === authoritativeCrop.y
    && cropDraft.width === authoritativeCrop.width && cropDraft.height === authoritativeCrop.height

  useEffect(() => {
    if (authoritativeCrop === null) return
    setCropText({
      x: String(authoritativeCrop.x), y: String(authoritativeCrop.y),
      width: String(authoritativeCrop.width), height: String(authoritativeCrop.height),
    })
  }, [authoritativeCrop?.x, authoritativeCrop?.y, authoritativeCrop?.width, authoritativeCrop?.height])

  useEffect(() => {
    if (frameObjectUrl.current !== null) {
      URL.revokeObjectURL(frameObjectUrl.current)
      frameObjectUrl.current = null
    }
    setFrameUrl(null)
    if (preview === null || compositeAuthority !== null) return
    const controller = new AbortController()
    void backend.fetchPreviewArtifact(preview.artifact_id, controller.signal).then((blob) => {
      if (controller.signal.aborted) return
      if (frameObjectUrl.current !== null) URL.revokeObjectURL(frameObjectUrl.current)
      frameObjectUrl.current = URL.createObjectURL(blob)
      setFrameUrl(frameObjectUrl.current)
    }).catch((error: unknown) => {
      if (!controller.signal.aborted) onError(error instanceof Error ? error : '无法载入已验证的场景帧。')
    })
    return () => controller.abort()
  }, [backend, compositeAuthority, onError, preview?.artifact_id])

  useEffect(() => () => {
    if (frameObjectUrl.current !== null) URL.revokeObjectURL(frameObjectUrl.current)
  }, [])

  useEffect(() => {
    const request = ++compositeRequest.current
    descriptorAuthority.current = null
    if (compositeObjectUrl.current !== null) {
      URL.revokeObjectURL(compositeObjectUrl.current)
      compositeObjectUrl.current = null
    }
    setCompositeVideo(null)
    if (compositeAuthority === null) return

    const controller = new AbortController()
    void backend.getCompositePreview(controller.signal).then(async (descriptor) => {
      if (controller.signal.aborted || compositeRequest.current !== request) return
      descriptorAuthority.current = descriptor.artifact_id
      const blob = await backend.fetchCompositePreviewArtifact(
        descriptor.artifact_id,
        controller.signal,
      )
      if (
        controller.signal.aborted
        || compositeRequest.current !== request
        || descriptorAuthority.current !== descriptor.artifact_id
      ) return
      const nextUrl = URL.createObjectURL(blob)
      if (
        controller.signal.aborted
        || compositeRequest.current !== request
        || descriptorAuthority.current !== descriptor.artifact_id
      ) {
        URL.revokeObjectURL(nextUrl)
        return
      }
      if (compositeObjectUrl.current !== null) {
        URL.revokeObjectURL(compositeObjectUrl.current)
      }
      compositeObjectUrl.current = nextUrl
      setCompositeVideo({
        authority: compositeAuthority,
        artifactId: descriptor.artifact_id,
        url: nextUrl,
      })
    }).catch((error: unknown) => {
      if (controller.signal.aborted || compositeRequest.current !== request) return
      descriptorAuthority.current = null
      if (compositeObjectUrl.current !== null) {
        URL.revokeObjectURL(compositeObjectUrl.current)
        compositeObjectUrl.current = null
      }
      setCompositeVideo(null)
      onError(error instanceof Error ? error : '无法载入后端验证的合成预览。')
    })

    return () => {
      controller.abort()
      if (compositeRequest.current === request) {
        descriptorAuthority.current = null
      }
      if (compositeObjectUrl.current !== null) {
        URL.revokeObjectURL(compositeObjectUrl.current)
        compositeObjectUrl.current = null
      }
    }
  }, [backend, compositeAuthority, onError])

  const generate = async (): Promise<void> => {
    if (busy || !cropSaved) return
    setRunning(true)
    try { await onStartStage('composite') }
    catch (error) { onError(error instanceof Error ? error : '预览合成任务失败。') }
    finally { setRunning(false) }
  }

  const saveCrop = async (): Promise<void> => {
    if (!cropValid || busy || running) return
    setRunning(true)
    try {
      onProjectChange(await backend.updateProject({
        expected_project_id: project.project_id,
        output_crop: cropDraft,
      }))
    } catch (error) {
      onError(error instanceof Error ? error : '无法保存固定输出裁剪。')
    } finally {
      setRunning(false)
    }
  }

  const displayedComposite = compositeVideo?.authority === compositeAuthority
    ? compositeVideo
    : null
  const compositeRunning = activeTask?.target_stage === 'composite'
    && ['queued', 'running'].includes(activeTask.status)
  const stageRows = ['map_trajectory', 'render', 'composite'] as const
  const eventError = latestEvent?.type === 'task_event'
    && latestEvent.task_id === activeTask?.id
    ? latestEvent.error
    : null
  const eventCode = typeof eventError?.code === 'string' ? eventError.code : null
  const eventCategory = typeof eventError?.category === 'string' ? eventError.category : null
  const recoveryTarget = activeTask?.target_stage ?? 'composite'
  const recoveryStage = project.stages[recoveryTarget]
  const recoveryCode = eventCode ?? activeTask?.error ?? recoveryStage?.error_code ?? ''
  const recoveryKey = `${recoveryTarget}:${eventCategory ?? ''}:${recoveryCode}`.toLowerCase()
  const subjectRecovery = recoveryTarget === 'segment'
    || /(subject|segment|mask|alpha)/.test(recoveryKey)
  const cameraRecovery = !subjectRecovery && (
    recoveryTarget === 'solve_camera'
    || recoveryTarget === 'map_trajectory'
    || /(camera|authority|preview_stale|pick)/.test(recoveryKey)
  )
  const resourceRecovery = !subjectRecovery && !cameraRecovery && (
    recoveryTarget === 'composite'
    || /(composite|encode|preview)/.test(recoveryKey)
  )
  const retryable = activeTask?.status === 'failed'
    && eventError?.retryable === true
    && activeTask.target_stage === 'composite'
  const recoveryFailed = activeTask?.status === 'failed'
    || recoveryStage?.status === 'failed'
  return (
    <section aria-labelledby="preview-title" className="page-grid">
      <div className="page-heading">
        <p className="eyebrow">04 · PREVIEW</p>
        <h2 id="preview-title">检查构图与运动</h2>
        <p>人物像素固定在解算相机画面中；这里检查自动地面对齐后的完整低分辨率合成。</p>
      </div>
      <div className="preview-layout">
        <article className="preview-card">
          <div className="preview-media preview-surface">
            {compositeAuthority !== null ? (
              displayedComposite === null
                ? <div className="viewport-empty">正在验证合成预览…</div>
                : (
                    <video
                      aria-label="低分辨率合成预览"
                      controls
                      playsInline
                      preload="metadata"
                      src={displayedComposite.url}
                    />
                  )
            ) : (
              frameUrl === null
                ? <div className="viewport-empty">暂无相机参考帧</div>
                : <img alt="相机参考帧（非合成视频）" src={frameUrl} />
            )}
          </div>
          <div className="preview-caption">
            <span>{compositeAuthority === null ? '相机参考 · 非合成视频' : '后端验证 · 低分辨率合成'}</span>
            <span>ViPE 逐帧内参</span>
            <span>目标地面 r{project.workflow.target_ground?.revision ?? '—'}</span>
          </div>
        </article>
        <aside className="control-card">
          <h3>全片合成</h3>
          <p>GS 比例 {project.workflow.gs_scale.toFixed(3)}× · 方位角 {project.workflow.scene_azimuth.toFixed(1)}°</p>
          <fieldset disabled={busy || running}>
            <legend>固定输出裁剪</legend>
            {(['x', 'y', 'width', 'height'] as const).map((key) => <label key={key}>{key.toUpperCase()}<input aria-label={`输出裁剪 ${key.toUpperCase()}`} onChange={(event) => setCropText((current) => ({ ...current, [key]: event.currentTarget.value }))} step={key === 'width' || key === 'height' ? 2 : 1} type="number" value={cropText[key]} /></label>)}
          </fieldset>
          <p className="technical-note">X/Y 使用源画面像素坐标，可为负数；裁剪框允许超出源视频范围，外部区域由 GS 背景填充。宽高须为偶数，最大 3840×2160。</p>
          <button disabled={busy || running || !cropValid || cropSaved} onClick={() => void saveCrop()} type="button">保存固定裁剪</button>
          <button disabled={busy || running || compositeRunning || !groundConfirmed || !cropSaved} onClick={() => void generate()} type="button">{busy || running || compositeRunning ? '生成中…' : '生成预览'}</button>
          <div className="stage-list" aria-label="预览阶段缓存状态">
            {stageRows.map((name) => <div key={name}><span>{name}</span><strong>{project.stages[name]?.status ?? 'pending'}</strong></div>)}
          </div>
        </aside>
      </div>
      <div className="honest-state">
        <strong>{compositeAuthority === null ? '等待真实合成任务' : '合成阶段已由后端确认'}</strong>
        <p>{compositeAuthority === null
          ? '当前静态 Gaussian 画面仅用于机位参考，不是合成结果。'
          : '播放器只使用当前成功合成缓存对应的后端验证视频。'}</p>
      </div>
      {recoveryFailed ? (
        <div className="recovery-actions">
          {subjectRecovery ? <button onClick={onReselectSubject} type="button">重新选择人物</button> : null}
          {resourceRecovery ? <button onClick={() => void backend.updateProject({ preview_height: 360 }).then(onProjectChange).catch((error: unknown) => onError(error instanceof Error ? error : '无法降低预览分辨率。'))} type="button">降低预览分辨率</button> : null}
          {cameraRecovery ? <button onClick={onBackToCamera} type="button">返回机位</button> : null}
          {retryable ? <button disabled={busy} onClick={() => void generate()} type="button">重试</button> : null}
        </div>
      ) : null}
    </section>
  )
}
