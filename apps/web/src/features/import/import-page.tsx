import { useEffect, useRef, useState } from 'react'

import type { BackendClient } from '../../api/backend-client'
import type { AssetKind, EnvironmentDto, ProjectDto } from '../../api/types'
import type { PlatformBridge } from '../../platform/platform-bridge'

export async function sha256Hex(blob: Blob): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', await blob.arrayBuffer())
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('')
}

interface ImportPageProps {
  backend: BackendClient
  environment: EnvironmentDto
  platform: PlatformBridge
  project: ProjectDto
  onError(value: unknown): void
  onProjectChange(project: ProjectDto): void
  onStartStage(stage: 'ingest'): Promise<unknown>
}

export function ImportPage({
  backend,
  environment,
  platform,
  project,
  onError,
  onProjectChange,
  onStartStage,
}: ImportPageProps) {
  const [status, setStatus] = useState<string | null>(null)
  const [uploadId, setUploadId] = useState<string | null>(null)
  const uploadAbort = useRef<AbortController | null>(null)
  const activeUpload = useRef<string | null>(null)
  const resumable = useRef<{
    kind: AssetKind
    filename: string
    size: number
    sha256: string
    id: string
    chunkSize: number
  } | null>(null)

  useEffect(() => () => {
    uploadAbort.current?.abort()
    if (activeUpload.current !== null) void backend.cancelUpload(activeUpload.current)
  }, [backend])

  const refresh = async (): Promise<ProjectDto> => {
    const next = await backend.getProject()
    onProjectChange(next)
    return next
  }

  const importAsset = async (kind: AssetKind): Promise<void> => {
    setStatus(kind === 'source_video' ? '正在选择源视频…' : '正在选择 Gaussian 场景…')
    try {
      const picked = await platform.pickInputFile({
        kind,
        extensions: kind === 'source_video' ? ['mp4', 'mov', 'mkv'] : ['ply'],
        description: kind === 'source_video' ? '源视频' : 'Gaussian 场景',
      })
      if (picked === null) {
        setStatus(null)
        return
      }
      if (picked.kind === 'browser-file') {
        const file = picked.file
        const controller = new AbortController()
        uploadAbort.current = controller
        setStatus(`正在校验 ${file.name}…`)
        const sha256 = await sha256Hex(file)
        const previous = resumable.current
        const session = previous !== null
          && previous.kind === kind
          && previous.filename === file.name
          && previous.size === file.size
          && previous.sha256 === sha256
          ? { id: previous.id, chunk_size: previous.chunkSize }
          : await backend.createUpload({
              kind,
              filename: file.name,
              mime_type: file.type || 'application/octet-stream',
              total_size: file.size,
              sha256,
            })
        resumable.current = {
          kind, filename: file.name, size: file.size, sha256,
          id: session.id, chunkSize: session.chunk_size,
        }
        activeUpload.current = session.id
        setUploadId(session.id)
        let uploaded = new Set<number>()
        try {
          const existing = await backend.getUpload(session.id)
          uploaded = new Set(existing?.uploaded_chunks ?? [])
        } catch {
          // A freshly created upload may not need an extra status round-trip.
        }
        const chunkCount = Math.ceil(file.size / session.chunk_size)
        for (let index = 0; index < chunkCount; index += 1) {
          if (uploaded.has(index)) continue
          const start = index * session.chunk_size
          await backend.putUploadChunk(
            session.id,
            index,
            file.slice(start, Math.min(file.size, start + session.chunk_size)),
            controller.signal,
          )
          setStatus(`上传 ${file.name} · ${index + 1} / ${chunkCount} 块`)
        }
        await backend.completeUpload(session.id)
        resumable.current = null
        activeUpload.current = null
        setUploadId(null)
        uploadAbort.current = null
      }
      const next = await refresh()
      setStatus('素材已由本地服务校验。')
      if (next.workflow.source_summary !== null && next.workflow.scene_summary !== null) {
        await onStartStage('ingest')
      }
    } catch (error) {
      uploadAbort.current = null
      onError(error instanceof Error ? error : '素材导入失败。')
      setStatus(null)
    }
  }

  const cancelUpload = async (): Promise<void> => {
    uploadAbort.current?.abort()
    if (uploadId !== null) await backend.cancelUpload(uploadId)
    uploadAbort.current = null
    activeUpload.current = null
    resumable.current = null
    setUploadId(null)
    setStatus('上传已取消，可以重新选择文件并重新上传。')
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
          <button onClick={() => void importAsset('source_video')} type="button">选择源视频</button>
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
          <button onClick={() => void importAsset('scene_ply')} type="button">选择 Gaussian 场景</button>
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
