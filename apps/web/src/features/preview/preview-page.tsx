import { useEffect, useRef, useState } from 'react'

import type { BackendClient } from '../../api/backend-client'
import type { ProjectDto, TaskDto } from '../../api/types'

interface PreviewPageProps {
  backend: BackendClient
  project: ProjectDto
  activeTask: TaskDto | null
  onError(value: unknown): void
  onProjectChange(project: ProjectDto): void
  onStartStage(stage: 'composite'): Promise<unknown>
  onBackToCamera(): void
  onReselectSubject(): void
}

export function PreviewPage({
  backend, project, activeTask, onError, onProjectChange, onStartStage,
  onBackToCamera, onReselectSubject,
}: PreviewPageProps) {
  const [frameUrl, setFrameUrl] = useState<string | null>(null)
  const [motionScale, setMotionScale] = useState(String(project.workflow.motion_scale))
  const [running, setRunning] = useState(false)
  const url = useRef<string | null>(null)
  const preview = project.workflow.preview

  useEffect(() => {
    if (preview === null) return
    const controller = new AbortController()
    void backend.fetchPreviewArtifact(preview.artifact_id, controller.signal).then((blob) => {
      if (controller.signal.aborted) return
      if (url.current !== null) URL.revokeObjectURL(url.current)
      url.current = URL.createObjectURL(blob)
      setFrameUrl(url.current)
    }).catch((error: unknown) => {
      if (!controller.signal.aborted) onError(error instanceof Error ? error : '无法载入已验证的场景帧。')
    })
    return () => controller.abort()
  }, [backend, onError, preview?.artifact_id])

  useEffect(() => () => {
    if (url.current !== null) URL.revokeObjectURL(url.current)
  }, [])

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
    setRunning(true)
    try { await onStartStage('composite') }
    catch (error) { onError(error instanceof Error ? error : '预览合成任务失败。') }
    finally { setRunning(false) }
  }

  const composite = project.stages.composite
  const compositeRunning = activeTask?.target_stage === 'composite'
    && ['queued', 'running'].includes(activeTask.status)
  const stageRows = ['map_trajectory', 'render', 'composite'] as const
  return (
    <section aria-labelledby="preview-title" className="page-grid">
      <div className="page-heading">
        <p className="eyebrow">04 · PREVIEW</p>
        <h2 id="preview-title">检查构图与运动</h2>
        <p>调整只会让后端定向失效轨迹、渲染、合成和导出阶段，不会重新分割人物。</p>
      </div>
      <div className="preview-layout">
        <article className="preview-card">
          {frameUrl === null ? <div className="viewport-empty">暂无场景预览帧</div> : <img alt="最近验证的合成参考帧" src={frameUrl} />}
          <div className="preview-caption">
            <span>垂直 FOV {project.workflow.target_camera?.fov_y_degrees.toFixed(0) ?? '—'}°</span>
            <span>焦点距离 {project.workflow.target_camera?.distance.toFixed(2) ?? '—'}</span>
          </div>
        </article>
        <aside className="control-card">
          <h3>运动迁移</h3>
          <label>运动幅度
            <input aria-label="运动幅度" max="4" min="0.1" onChange={(event) => setMotionScale(event.currentTarget.value)} step="0.05" type="range" value={motionScale} />
            <output>{Number(motionScale).toFixed(2)}×</output>
          </label>
          <button className="button-secondary" onClick={() => void patchMotion()} type="button">应用运动幅度</button>
          <button disabled={running || compositeRunning || project.workflow.foot_point === null} onClick={() => void generate()} type="button">{running || compositeRunning ? '生成中…' : '生成预览'}</button>
          <div className="stage-list" aria-label="预览阶段缓存状态">
            {stageRows.map((name) => <div key={name}><span>{name}</span><strong>{project.stages[name]?.status ?? 'pending'}</strong></div>)}
          </div>
        </aside>
      </div>
      <div className="honest-state">
        <strong>{composite?.status === 'succeeded' ? '合成阶段已由后端确认' : '等待真实合成任务'}</strong>
        <p>当前 API 公开的是最近验证的静态场景帧；低分辨率合成视频尚未暴露。完整预览仍需要 Task 15 前完成具体 WorkflowServices 组装。</p>
      </div>
      {composite?.status === 'failed' ? (
        <div className="recovery-actions">
          <button onClick={onReselectSubject} type="button">重新选择人物</button>
          <button onClick={() => void backend.updateProject({ preview_height: 360 }).then(onProjectChange).catch((error: unknown) => onError(error instanceof Error ? error : '无法降低预览分辨率。'))} type="button">降低预览分辨率</button>
          <button onClick={onBackToCamera} type="button">返回机位</button>
          <button onClick={() => void generate()} type="button">重试</button>
        </div>
      ) : null}
    </section>
  )
}
