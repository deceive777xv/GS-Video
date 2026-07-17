import { useEffect, useRef, useState } from 'react'

import type { BackendClient } from '../../api/backend-client'
import type { ProjectDto, TaskDto, VerifiedExportDto } from '../../api/types'
import type { PlatformBridge } from '../../platform/platform-bridge'

interface ExportPageProps {
  backend: BackendClient
  platform: PlatformBridge
  project: ProjectDto
  activeTask: TaskDto | null
  onError(value: unknown): void
  onStartStage(stage: 'export'): Promise<TaskDto>
}

export function ExportPage({ backend, platform, project, activeTask, onError, onStartStage }: ExportPageProps) {
  const [exporting, setExporting] = useState(false)
  const [verified, setVerified] = useState<VerifiedExportDto | null>(null)
  const finalizedTask = useRef<string | null>(null)
  const localExportIntent = useRef(false)

  const saveVerified = async (descriptor: VerifiedExportDto): Promise<void> => {
    if (platform.kind === 'browser') {
      await platform.saveExport(descriptor.filename, {
        kind: 'browser-download',
        blob: await backend.fetchExportArtifact(descriptor.artifact_id),
      })
    } else {
      await platform.saveExport(descriptor.filename, {
        kind: 'local-export',
        path: `verified-export:${descriptor.artifact_id}`,
        saveTo: (destination) => backend.copyVerifiedExport(descriptor.artifact_id, destination),
      })
    }
  }

  const loadVerified = async (): Promise<VerifiedExportDto> => {
    const descriptor = await backend.getVerifiedExport()
    setVerified(descriptor)
    return descriptor
  }

  useEffect(() => {
    if (project.stages.export?.status !== 'succeeded'
      || project.workflow.export_result?.verified !== true) return
    let active = true
    void backend.getVerifiedExport().then((descriptor) => {
      if (active) setVerified(descriptor)
    }).catch((error: unknown) => {
      if (active) onError(error instanceof Error ? error : '无法恢复已验证导出元数据。')
    })
    return () => { active = false }
  }, [backend, onError, project.stages.export?.status, project.workflow.export_result?.artifact_id, project.workflow.export_result?.verified])

  useEffect(() => {
    if (activeTask?.target_stage !== 'export' || activeTask.status !== 'succeeded'
      || finalizedTask.current === activeTask.id) return
    finalizedTask.current = activeTask.id
    const saveAfterVerification = localExportIntent.current
    localExportIntent.current = false
    setExporting(true)
    void loadVerified().then(async (descriptor) => {
      if (saveAfterVerification) await saveVerified(descriptor)
    }).catch((error: unknown) => {
      finalizedTask.current = null
      onError(error instanceof Error ? error : '后端未能提供通过 ffprobe 验证的导出。')
    }).finally(() => setExporting(false))
  }, [activeTask, backend, onError, platform])

  const start = async (): Promise<void> => {
    setExporting(true)
    localExportIntent.current = true
    try {
      const task = await onStartStage('export')
      if (task.status === 'succeeded' && finalizedTask.current !== task.id) {
        finalizedTask.current = task.id
        localExportIntent.current = false
        const descriptor = await loadVerified()
        await saveVerified(descriptor)
      }
    } catch (error) {
      localExportIntent.current = false
      onError(error instanceof Error ? error : '导出任务失败。')
    } finally { setExporting(false) }
  }

  const authoritativeReady = project.stages.composite?.status === 'succeeded'
  const exportRunning = activeTask?.target_stage === 'export'
    && ['queued', 'running'].includes(activeTask.status)
  return (
    <section aria-labelledby="export-title" className="page-grid export-page">
      <div className="page-heading">
        <p className="eyebrow">05 · EXPORT</p>
        <h2 id="export-title">导出成片</h2>
        <p>只有合成阶段成功后才能启动导出；保存前再次读取后端的 ffprobe 验证结果。</p>
      </div>
      <article className="export-card">
        <div className="export-orbit" aria-hidden="true"><span /></div>
        <div>
          <span className="asset-kicker">VERIFIED MP4</span>
          <h3>{verified?.filename ?? '准备最终输出'}</h3>
          {verified === null ? <p>文件路径不会进入浏览器 DTO；保存位置由平台对话框决定。</p> : (
            <dl className="asset-stats">
              <div><dt>帧数</dt><dd>{verified.frame_count}</dd></div>
              <div><dt>时长</dt><dd>{verified.duration_seconds.toFixed(2)} 秒</dd></div>
              <div><dt>音轨</dt><dd>{verified.has_audio ? '已验证' : '无'}</dd></div>
            </dl>
          )}
          <div className="export-actions">
            <button disabled={!authoritativeReady || exporting || exportRunning} onClick={() => void start()} type="button">{exporting || exportRunning ? '验证导出中…' : '导出视频'}</button>
            {verified !== null ? <button className="button-secondary" disabled={exporting} onClick={() => void saveVerified(verified).catch((error: unknown) => onError(error instanceof Error ? error : '保存导出失败。'))} type="button">保存已验证视频</button> : null}
          </div>
        </div>
      </article>
    </section>
  )
}
