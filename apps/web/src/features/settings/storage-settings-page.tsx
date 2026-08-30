import { useEffect, useRef, useState } from 'react'

import type { BackendClient } from '../../api/backend-client'
import type {
  StorageLayoutDto,
  StorageLayoutStatusDto,
  StorageLayoutUpdate,
} from '../../api/types'
import type { PlatformBridge } from '../../platform/platform-bridge'

interface StorageSettingsPageProps {
  backend: BackendClient
  busy: boolean
  initial: StorageLayoutStatusDto
  platform: PlatformBridge
  onChange(value: StorageLayoutStatusDto): void
  onError(value: unknown): void
}

function formatBytes(value: number): string {
  if (value < 1024 ** 3) return `${(value / 1024 ** 2).toFixed(1)} MB`
  return `${(value / 1024 ** 3).toFixed(1)} GB`
}

export function StorageSettingsPage({
  backend,
  busy,
  initial,
  platform,
  onChange,
  onError,
}: StorageSettingsPageProps) {
  const [layout, setLayout] = useState<StorageLayoutDto | null>(null)
  const [projectRoot, setProjectRoot] = useState(initial.project_library_root)
  const [cacheRoot, setCacheRoot] = useState(initial.cache_root)
  const [projectAction, setProjectAction] = useState<StorageLayoutUpdate['project_action']>('migrate')
  const [cacheAction, setCacheAction] = useState<StorageLayoutUpdate['cache_action']>('start_fresh')
  const [saving, setSaving] = useState(false)
  const [cleaning, setCleaning] = useState(false)
  const [cleanupMessage, setCleanupMessage] = useState<string | null>(null)
  const [measurementFailed, setMeasurementFailed] = useState(false)
  const measurement = useRef<{
    key: string
    request: Promise<StorageLayoutDto>
  } | null>(null)

  useEffect(() => {
    const key = `${initial.project_library_id}:${initial.cache_id}:${String(initial.restart_required)}`
    if (measurement.current?.key !== key) {
      measurement.current = { key, request: backend.getStorageLayout() }
    }
    const request = measurement.current.request
    let active = true
    setLayout(null)
    setMeasurementFailed(false)
    setProjectRoot(initial.project_library_root)
    setCacheRoot(initial.cache_root)
    void request.then((next) => {
      if (active) setLayout(next)
    }).catch((error: unknown) => {
      if (!active) return
      setMeasurementFailed(true)
      onError(error)
    })
    return () => {
      active = false
    }
  }, [backend, initial.cache_id, initial.cache_root, initial.project_library_id, initial.project_library_root, initial.restart_required, onError])

  const retryMeasurement = (): void => {
    const key = `${initial.project_library_id}:${initial.cache_id}:${String(initial.restart_required)}`
    const request = backend.getStorageLayout()
    measurement.current = { key, request }
    setMeasurementFailed(false)
    void request.then(setLayout).catch((error: unknown) => {
      setMeasurementFailed(true)
      onError(error)
    })
  }

  const choose = async (kind: 'project' | 'cache'): Promise<void> => {
    if (platform.pickDirectory === undefined) return
    const selected = await platform.pickDirectory()
    if (selected === null) return
    if (kind === 'project') setProjectRoot(selected)
    else setCacheRoot(selected)
  }

  const save = async (): Promise<void> => {
    setSaving(true)
    try {
      const next = await backend.updateStorageLayout({
        project_library_root: projectRoot,
        cache_root: cacheRoot,
        project_action: projectAction,
        cache_action: cacheAction,
      })
      setLayout(next)
      onChange(next)
    } catch (error) {
      onError(error)
    } finally {
      setSaving(false)
    }
  }

  const cleanup = async (mode: 'safe' | 'deep'): Promise<void> => {
    setCleaning(true)
    setCleanupMessage(null)
    try {
      const plan = await backend.planStorageCacheCleanup(mode)
      const warning = mode === 'deep'
        ? `深度清理将释放 ${formatBytes(plan.reclaimable_bytes)}，删除 ${plan.removable_entries} 项缓存，并让相关阶段重新计算。继续吗？`
        : `安全清理将释放 ${formatBytes(plan.reclaimable_bytes)}，删除 ${plan.removable_entries} 项未引用缓存。继续吗？`
      if (!window.confirm(warning)) return
      const result = await backend.cleanupStorageCache(mode, plan.plan_token)
      setLayout(result.storage)
      onChange(result.storage)
      setCleanupMessage(`已释放 ${formatBytes(result.freed_bytes)}，清理 ${result.removed_entries} 个缓存目录。`)
    } catch (error) {
      onError(error)
    } finally {
      setCleaning(false)
    }
  }

  const desktop = platform.kind === 'tauri' && platform.pickDirectory !== undefined
  const status = layout ?? initial
  const disabled = busy || saving || cleaning || !status.editable || !desktop
  const capacity = (value: number | undefined): string =>
    value === undefined ? '正在统计…' : formatBytes(value)

  return (
    <main className="hub-main storage-settings-main" id="main-content" tabIndex={-1}>
      <section className="storage-settings-heading">
        <p className="eyebrow">STORAGE</p>
        <h1>项目库与缓存目录</h1>
        <p>项目库保存项目配置和共享原始素材；缓存目录保存可重建的帧、遮罩、渲染与导出产物。</p>
      </section>

      <section className="storage-settings-grid">
        <article className="storage-root-card">
          <div><span>长期保存</span><h2>项目库</h2></div>
          <label htmlFor="project-library-root">目录</label>
          <div className="storage-path-row">
            <input id="project-library-root" readOnly value={projectRoot} />
            <button className="button-secondary" disabled={disabled} onClick={() => void choose('project')} type="button">选择目录</button>
          </div>
          <fieldset disabled={disabled}>
            <legend>切换方式</legend>
            <label><input checked={projectAction === 'migrate'} name="project-action" onChange={() => setProjectAction('migrate')} type="radio" />迁移当前项目库</label>
            <label><input checked={projectAction === 'open_existing'} name="project-action" onChange={() => setProjectAction('open_existing')} type="radio" />打开已有项目库</label>
          </fieldset>
          <dl className="storage-capacity"><div><dt>已占用</dt><dd>{capacity(layout?.project_library_bytes)}</dd></div><div><dt>磁盘剩余</dt><dd>{capacity(layout?.project_library_free_bytes)}</dd></div></dl>
        </article>

        <article className="storage-root-card">
          <div><span>可清理 / 可重建</span><h2>缓存目录</h2></div>
          <label htmlFor="cache-root">目录</label>
          <div className="storage-path-row">
            <input id="cache-root" readOnly value={cacheRoot} />
            <button className="button-secondary" disabled={disabled} onClick={() => void choose('cache')} type="button">选择目录</button>
          </div>
          <fieldset disabled={disabled}>
            <legend>切换方式</legend>
            <label><input checked={cacheAction === 'start_fresh'} name="cache-action" onChange={() => setCacheAction('start_fresh')} type="radio" />使用全新缓存</label>
            <label><input checked={cacheAction === 'migrate'} name="cache-action" onChange={() => setCacheAction('migrate')} type="radio" />迁移当前缓存</label>
          </fieldset>
          <dl className="storage-capacity"><div><dt>缓存占用</dt><dd>{capacity(layout?.cache_bytes)}</dd></div><div><dt>磁盘剩余</dt><dd>{capacity(layout?.cache_free_bytes)}</dd></div></dl>
          <div className="storage-cleanup-actions">
            <button className="button-secondary" disabled={busy || cleaning || status.restart_required || layout === null} onClick={() => void cleanup('safe')} type="button">安全清理</button>
            <button className="button-danger" disabled={busy || cleaning || status.restart_required || layout === null} onClick={() => void cleanup('deep')} type="button">深度清理</button>
          </div>
          {cleanupMessage === null ? null : <p className="storage-cleanup-result" role="status">{cleanupMessage}</p>}
          {measurementFailed ? <p className="storage-cleanup-result" role="alert">容量统计失败。<button className="button-secondary" onClick={retryMeasurement} type="button">重试容量统计</button></p> : null}
        </article>
      </section>

      {!desktop ? <p className="storage-settings-notice">浏览器模式只能查看当前目录；请在桌面应用中使用原生目录选择器修改。</p> : null}
      {status.restart_required ? (
        <div className="storage-restart-card" role="status">
          <div><strong>新目录已经提交</strong><p>重启后应用只会装配新的项目库与缓存目录。</p></div>
          <button disabled={platform.restartApp === undefined} onClick={() => void platform.restartApp?.()} type="button">重启应用</button>
        </div>
      ) : (
        <div className="storage-settings-actions">
          <small>仅支持本机固定磁盘上的普通目录；项目库与缓存目录不能互相包含。</small>
          <button disabled={disabled || (projectRoot === status.project_library_root && cacheRoot === status.cache_root)} onClick={() => void save()} type="button">应用目录设置</button>
        </div>
      )}
    </main>
  )
}
