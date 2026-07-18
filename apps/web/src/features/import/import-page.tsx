import { useEffect, useRef, useState } from 'react'

import type { BackendClient } from '../../api/backend-client'
import type { AssetKind, EnvironmentDto, ProjectDto } from '../../api/types'
import type { PlatformBridge } from '../../platform/platform-bridge'
import { sha256Hex } from './incremental-sha256'
import {
  clearUploadResume,
  matchesSelectedFile,
  matchesServerStatus,
  readUploadResume,
  writeUploadResume,
  type UploadResumeRecord,
} from './upload-resume'

function isMissingUpload(error: unknown): boolean {
  return typeof error === 'object'
    && error !== null
    && 'status' in error
    && error.status === 404
}

interface ImportPageProps {
  backend: BackendClient
  busy: boolean
  environment: EnvironmentDto
  platform: PlatformBridge
  project: ProjectDto
  onError(value: unknown): void
  onProjectChange(project: ProjectDto): void
  onStartStage(stage: 'ingest'): Promise<unknown>
}

export function ImportPage({
  backend,
  busy,
  environment,
  platform,
  project,
  onError,
  onProjectChange,
  onStartStage,
}: ImportPageProps) {
  const [status, setStatus] = useState<string | null>(null)
  const [uploadId, setUploadId] = useState<string | null>(null)
  const [importing, setImporting] = useState(false)
  const [cancelling, setCancelling] = useState(false)
  const uploadAbort = useRef<AbortController | null>(null)
  const importAdmission = useRef<AbortController | null>(null)
  const cancelAdmission = useRef(false)
  const resumable = useRef<UploadResumeRecord | null>(null)
  const disposed = useRef(false)

  useEffect(() => {
    disposed.current = false
    return () => {
      disposed.current = true
      uploadAbort.current?.abort()
    }
  }, [])

  const refresh = async (signal?: AbortSignal): Promise<ProjectDto> => {
    const next = await backend.getProject()
    if (!disposed.current && signal?.aborted !== true) onProjectChange(next)
    return next
  }

  const importAsset = async (kind: AssetKind): Promise<void> => {
    if (busy || importAdmission.current !== null || cancelAdmission.current) return
    const controller = new AbortController()
    importAdmission.current = controller
    uploadAbort.current = controller
    setImporting(true)
    setStatus(kind === 'source_video' ? '正在选择源视频…' : '正在选择 Gaussian 场景…')
    try {
      const picked = await platform.pickInputFile({
        kind,
        extensions: kind === 'source_video' ? ['mp4', 'mov', 'mkv'] : ['ply'],
        description: kind === 'source_video' ? '源视频' : 'Gaussian 场景',
      })
      if (disposed.current || controller.signal.aborted) return
      if (picked === null) {
        setStatus(null)
        return
      }
      if (picked.kind === 'browser-file') {
        const file = picked.file
        setStatus(`正在校验 ${file.name}…`)
        const sha256 = await sha256Hex(file, undefined, controller.signal)
        const previous = resumable.current?.kind === kind
          ? resumable.current
          : readUploadResume(project.project_id, kind)
        let record: UploadResumeRecord | null = null
        let uploaded = new Set<number>()
        if (previous !== null) {
          if (!matchesSelectedFile(previous, kind, file, sha256)) {
            try {
              await backend.cancelUpload(previous.id)
            } catch (error) {
              if (!isMissingUpload(error)) throw error
            }
            clearUploadResume(project.project_id, kind)
          } else {
            try {
              const existing = await backend.getUpload(previous.id)
              if (matchesServerStatus(previous, existing)) {
                record = previous
                uploaded = new Set(existing.uploaded_chunks)
              } else {
                try {
                  await backend.cancelUpload(previous.id)
                } catch (error) {
                  if (!isMissingUpload(error)) throw error
                }
                clearUploadResume(project.project_id, kind)
              }
            } catch (error) {
              if (!isMissingUpload(error)) throw error
              clearUploadResume(project.project_id, kind)
            }
          }
        }
        if (record === null) {
          const session = await backend.createUpload({
              kind,
              filename: file.name,
              mime_type: file.type || 'application/octet-stream',
              total_size: file.size,
              sha256,
            })
          record = {
            version: 1,
            projectId: project.project_id,
            kind,
            filename: file.name,
            mimeType: file.type || 'application/octet-stream',
            size: file.size,
            sha256,
            id: session.id,
            chunkSize: session.chunk_size,
          }
          writeUploadResume(record)
        }
        resumable.current = record
        if (disposed.current || controller.signal.aborted) return
        setUploadId(record.id)
        const chunkCount = Math.ceil(file.size / record.chunkSize)
        for (let index = 0; index < chunkCount; index += 1) {
          if (uploaded.has(index)) continue
          const start = index * record.chunkSize
          await backend.putUploadChunk(
            record.id,
            index,
            file.slice(start, Math.min(file.size, start + record.chunkSize)),
            controller.signal,
          )
          setStatus(`上传 ${file.name} · ${index + 1} / ${chunkCount} 块`)
        }
        await backend.completeUpload(record.id)
        if (disposed.current || controller.signal.aborted) return
        clearUploadResume(project.project_id, kind)
        resumable.current = null
        setUploadId(null)
        uploadAbort.current = null
      }
      if (disposed.current || controller.signal.aborted) return
      const next = await refresh(controller.signal)
      if (disposed.current || controller.signal.aborted) return
      setStatus('素材已由本地服务校验。')
      if (next.workflow.source_summary !== null && next.workflow.scene_summary !== null) {
        await onStartStage('ingest')
      }
    } catch (error) {
      if (disposed.current || controller.signal.aborted) return
      onError(error instanceof Error ? error : '素材导入失败。')
      setStatus(null)
    } finally {
      if (importAdmission.current === controller) {
        importAdmission.current = null
        uploadAbort.current = null
        if (!disposed.current) setImporting(false)
      }
    }
  }

  const cancelUpload = async (): Promise<void> => {
    if (cancelAdmission.current) return
    cancelAdmission.current = true
    setCancelling(true)
    const current = resumable.current
    const controller = uploadAbort.current
    const currentUploadId = current?.id ?? uploadId
    controller?.abort()
    try {
      if (currentUploadId !== null) {
        try {
          await backend.cancelUpload(currentUploadId)
        } catch (error) {
          if (!isMissingUpload(error)) throw error
        }
      }
      if (current !== null) clearUploadResume(project.project_id, current.kind)
      if (resumable.current?.id === current?.id) resumable.current = null
      if (uploadAbort.current === controller) uploadAbort.current = null
      setUploadId((value) => value === currentUploadId ? null : value)
      setStatus('上传已取消，可以重新选择文件并重新上传。')
    } catch (error) {
      onError(error instanceof Error ? error : '无法取消当前上传。')
    } finally {
      cancelAdmission.current = false
      if (!disposed.current) setCancelling(false)
    }
  }

  const video = project.workflow.source_summary
  const scene = project.workflow.scene_summary
  return (
    <section aria-labelledby="import-title" className="page-grid">
      <div className="page-heading">
        <p className="eyebrow">01 · INPUTS</p>
        <h2 id="import-title">导入素材</h2>
        <p>源视频提供人物、声音与镜头运动；Gaussian PLY 只提供目标环境。</p>
      </div>
      <div className="import-grid">
        <article className={`asset-drop ${video === null ? '' : 'asset-ready'}`}>
          <span className="asset-kicker">SOURCE VIDEO</span>
          <h3>{video?.filename ?? '单人短视频'}</h3>
          {video === null ? <p>10–30 秒，最高 1080p，单镜头。</p> : (
            <dl className="asset-stats">
              <div><dt>画面</dt><dd>{video.width} × {video.height}</dd></div>
              <div><dt>时长</dt><dd>{video.duration_seconds.toFixed(1)} 秒</dd></div>
              <div><dt>音轨</dt><dd>{video.has_audio ? '保留' : '无'}</dd></div>
            </dl>
          )}
          <button disabled={busy || importing || cancelling} onClick={() => void importAsset('source_video')} type="button">选择源视频</button>
        </article>
        <article className={`asset-drop ${scene === null ? '' : 'asset-ready'}`}>
          <span className="asset-kicker">GAUSSIAN SCENE</span>
          <h3>{scene?.filename ?? '静态 Gaussian PLY'}</h3>
          {scene === null ? <p>需要位置、尺度、旋转、不透明度与颜色属性。</p> : (
            <dl className="asset-stats">
              <div><dt>Gaussian</dt><dd>{scene.gaussian_count.toLocaleString()}</dd></div>
              <div><dt>预计显存</dt><dd>{scene.estimated_vram_mb.toLocaleString()} MB / 8192 MB</dd></div>
              <div><dt>预算</dt><dd>{scene.estimated_vram_mb <= 8192 ? '可尝试' : '建议降采样'}</dd></div>
            </dl>
          )}
          <button disabled={busy || importing || cancelling} onClick={() => void importAsset('scene_ply')} type="button">选择 Gaussian 场景</button>
        </article>
      </div>
      <aside className={`environment-card ${environment.ready ? 'is-ready' : 'has-issues'}`}>
        <div><span className="status-dot" /><strong>{environment.ready ? '本机环境可用' : '环境需要修复'}</strong></div>
        <p>检测显存 {environment.vram_mb.toLocaleString()} MB；处理按阶段串行，实时性不是目标。</p>
        {environment.issues.length > 0 ? (
          <ul>{environment.issues.map((issue) => <li key={issue.code}><code>{issue.code}</code> {issue.message}</li>)}</ul>
        ) : null}
      </aside>
      {status !== null ? <p aria-live="polite" className="inline-status">{status}</p> : null}
      {uploadId !== null ? <button className="button-secondary" onClick={() => void cancelUpload()} type="button">取消当前上传</button> : null}
    </section>
  )
}
