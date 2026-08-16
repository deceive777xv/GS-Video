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
  const [motionScale, setMotionScale] = useState(String(project.workflow.motion_scale))
  const [running, setRunning] = useState(false)
  const frameObjectUrl = useRef<string | null>(null)
  const compositeObjectUrl = useRef<string | null>(null)
  const compositeRequest = useRef(0)
  const descriptorAuthority = useRef<string | null>(null)
  const preview = project.workflow.preview
  const composite = project.stages.composite
  const synthesisPlacement = project.workflow.synthesis_placement ?? null
  const placementConfirmed = synthesisPlacement !== null
    && project.workflow.confirmed_synthesis_placement_revision
      === synthesisPlacement.revision
  const compositeAuthority = composite?.status === 'succeeded'
    && composite.cache_key !== null
    && composite.cache_key !== undefined
    ? composite.cache_key
    : null

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

  const patchMotion = async (): Promise<void> => {
    const value = Number(motionScale)
    if (!Number.isFinite(value) || value <= 0 || value > 4) {
      onError('运动幅度必须大于 0 且不超过 4。')
      return
    }
    try { onProjectChange(await backend.updateProject({ motion_scale: value })) }
    catch (error) { onError(error instanceof Error ? error : '运动幅度更新失败。') }
  }

  const generate = async (): Promise<void> => {
    if (busy) return
    setRunning(true)
    try { await onStartStage('composite') }
    catch (error) { onError(error instanceof Error ? error : '预览合成任务失败。') }
    finally { setRunning(false) }
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
        <p>调整只会让后端定向失效轨迹、渲染、合成和导出阶段，不会重新分割人物。</p>
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
            <span>源垂直 FOV {project.workflow.source_perspective_calibration?.vertical_fov.toFixed(0) ?? '—'}°</span>
            <span>受约束机位 r{project.workflow.synthesis_placement?.revision ?? '—'}</span>
          </div>
        </article>
        <aside className="control-card">
          <h3>运动迁移</h3>
          <label>运动幅度
            <input aria-label="运动幅度" max="4" min="0.1" onChange={(event) => setMotionScale(event.currentTarget.value)} step="0.05" type="range" value={motionScale} />
            <output>{Number(motionScale).toFixed(2)}×</output>
          </label>
          <button className="button-secondary" onClick={() => void patchMotion()} type="button">应用运动幅度</button>
          <button disabled={busy || running || compositeRunning || !placementConfirmed} onClick={() => void generate()} type="button">{busy || running || compositeRunning ? '生成中…' : '生成预览'}</button>
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
