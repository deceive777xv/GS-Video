import { useEffect, useState } from 'react'

import type { BackendClient } from '../../api/backend-client'
import type { AssetKind, AssetListItemDto, LibraryAssetKind, ProjectDto } from '../../api/types'
import { sha256Hex } from '../import/incremental-sha256'
import type { PlatformBridge } from '../../platform/platform-bridge'

interface AssetLibraryPageProps {
  backend: BackendClient
  busy: boolean
  kind: LibraryAssetKind
  platform: PlatformBridge
  project: ProjectDto | null
  returnProjectId: string | null
  onError(error: unknown): void
  onProjectChange(project: ProjectDto, context: AssetSelectionContext): void | Promise<void>
}

export interface AssetSelectionContext {
  expectedProjectId: string
  returnProjectId: string | null
}

function apiKind(kind: LibraryAssetKind): AssetKind {
  return kind === 'video' ? 'source_video' : 'scene_ply'
}

function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KB`
  if (value < 1024 ** 3) return `${(value / 1024 ** 2).toFixed(1)} MB`
  return `${(value / 1024 ** 3).toFixed(2)} GB`
}

export function AssetLibraryPage({ backend, busy, kind, platform, project, returnProjectId, onError, onProjectChange }: AssetLibraryPageProps) {
  const [items, setItems] = useState<AssetListItemDto[]>([])
  const [loading, setLoading] = useState(true)
  const [importing, setImporting] = useState(false)
  const [selectingId, setSelectingId] = useState<string | null>(null)

  const refresh = async (): Promise<void> => {
    setItems(await backend.listAssets(kind))
  }

  useEffect(() => {
    let disposed = false
    setLoading(true)
    void backend.listAssets(kind).then((next) => { if (!disposed) setItems(next) })
      .catch((error: unknown) => { if (!disposed) onError(error) })
      .finally(() => { if (!disposed) setLoading(false) })
    return () => { disposed = true }
  }, [backend, kind, onError])

  const importAsset = async (): Promise<void> => {
    if (importing) return
    setImporting(true)
    let uploadId: string | null = null
    try {
      const picked = await platform.pickInputFile({
        kind: apiKind(kind),
        extensions: kind === 'video' ? ['mp4', 'mov', 'mkv'] : ['ply'],
        description: kind === 'video' ? '视频素材' : 'PLY 素材',
        assignToCurrent: false,
      })
      if (picked === null) return
      if (picked.kind === 'browser-file') {
        const file = picked.file
        const sha256 = await sha256Hex(file)
        const session = await backend.createUpload({
          kind: apiKind(kind),
          filename: file.name,
          mime_type: file.type || 'application/octet-stream',
          total_size: file.size,
          sha256,
          assign_to_current: false,
        })
        uploadId = session.id
        const chunks = Math.ceil(file.size / session.chunk_size)
        for (let index = 0; index < chunks; index += 1) {
          const start = index * session.chunk_size
          await backend.putUploadChunk(session.id, index, file.slice(start, Math.min(file.size, start + session.chunk_size)))
        }
        await backend.completeUpload(session.id)
        uploadId = null
      }
      await refresh()
    } catch (error) {
      if (uploadId !== null) await backend.cancelUpload(uploadId).catch(() => undefined)
      onError(error)
    } finally {
      setImporting(false)
    }
  }

  const selectedId = kind === 'video' ? project?.source_video_asset_id : project?.scene_ply_asset_id
  const returnQuery = returnProjectId === null
    ? ''
    : `?returnProject=${encodeURIComponent(returnProjectId)}`

  const selectAsset = async (assetId: string): Promise<void> => {
    const expectedProjectId = project?.project_id
    if (selectingId !== null || expectedProjectId === undefined) return
    const context = { expectedProjectId, returnProjectId }
    setSelectingId(assetId)
    try {
      await onProjectChange(
        await backend.selectProjectAsset(apiKind(kind), assetId, expectedProjectId),
        context,
      )
    } catch (error) {
      onError(error)
    } finally {
      setSelectingId(null)
    }
  }

  return (
    <main className="hub-main asset-library-main" id="main-content" tabIndex={-1}>
      <section className="section-heading asset-library-heading">
        <div><p className="eyebrow">ASSET LIBRARY</p><h1>素材库</h1><p>视频与 PLY 分开管理，同一份素材可供多个项目复用。</p></div>
        <button disabled={importing} onClick={() => void importAsset()} type="button">{importing ? '正在导入…' : `导入${kind === 'video' ? '视频' : ' PLY'}`}</button>
      </section>
      <nav className="asset-tabs" aria-label="素材类型">
        <a className={kind === 'video' ? 'is-current' : ''} href={`#/assets/video${returnQuery}`}>视频</a>
        <a className={kind === 'ply' ? 'is-current' : ''} href={`#/assets/ply${returnQuery}`}>PLY</a>
      </nav>
      {project === null ? <p className="technical-note">请先打开或新建项目，再把素材用于当前项目。</p> : busy ? <p className="technical-note">当前项目任务运行中，完成或取消后才能更换素材。</p> : null}
      {loading ? <div className="empty-state">正在读取素材索引…</div> : items.length === 0 ? (
        <div className="empty-state"><strong>暂无{kind === 'video' ? '视频' : ' PLY'}素材</strong><p>导入后，文件会保存到应用共享素材目录，不会复制进项目目录。</p></div>
      ) : (
        <div className="asset-table-wrap"><table className="asset-table"><thead><tr><th>文件</th><th>大小</th><th>导入时间</th><th>引用项目</th><th>操作</th></tr></thead><tbody>
          {items.map(({ asset, references }) => (
            <tr key={asset.asset_id}>
              <td><strong>{asset.original_filename}</strong><small>{asset.sha256.slice(0, 12)}…</small></td>
              <td>{formatBytes(asset.size)}</td><td>{new Date(asset.imported_at).toLocaleString()}</td>
              <td>{references.length === 0 ? '未引用' : references.map((item) => item.name).join('、')}</td>
              <td><div className="table-actions"><button className="button-secondary" disabled={project === null || busy || selectingId !== null || selectedId === asset.asset_id} onClick={() => void selectAsset(asset.asset_id)} type="button">{selectedId === asset.asset_id ? '当前使用' : '用于当前项目'}</button><button className="button-quiet-danger" disabled={references.length > 0} title={references.length > 0 ? '被项目引用的素材不能删除' : undefined} onClick={() => void backend.deleteAsset(asset.asset_id).then(refresh).catch(onError)} type="button">删除</button></div></td>
            </tr>
          ))}
        </tbody></table></div>
      )}
    </main>
  )
}
