import { useEffect, useMemo, useRef, useState } from 'react'

import type { BackendClient } from '../../api/backend-client'
import type { ExportEncodingSettings, ProjectDto, TaskDto, VerifiedExportDto } from '../../api/types'
import type { PlatformBridge } from '../../platform/platform-bridge'

interface ExportPageProps {
  backend: BackendClient
  busy: boolean
  platform: PlatformBridge
  project: ProjectDto
  activeTask: TaskDto | null
  onError(value: unknown): void
  onProjectChange(project: ProjectDto): void
  onStartStage(stage: 'export'): Promise<TaskDto>
}

export function ExportPage({ backend, busy, platform, project, activeTask, onError, onProjectChange, onStartStage }: ExportPageProps) {
  const [exporting, setExporting] = useState(false)
  const [settings, setSettings] = useState<ExportEncodingSettings>(project.workflow.export_settings)
  const [verified, setVerified] = useState<VerifiedExportDto | null>(null)
  const [browserArtifact, setBrowserArtifact] = useState<{
    artifactId: string
    blob: Blob
  } | null>(null)
  const finalizedTask = useRef<string | null>(null)
  const localExportIntent = useRef(false)
  const verificationAuthority = useRef(0)
  const authoritativeArtifactId = project.stages.export?.status === 'succeeded'
    && project.workflow.export_result?.verified === true
    ? project.workflow.export_result.artifact_id
    : null
  const settingsDirty = JSON.stringify(settings) !== JSON.stringify(project.workflow.export_settings)
  const recommendedBitrate = useMemo(() => {
    const source = project.workflow.source_summary
    if (source === null) return 12
    const [numerator = '0', denominator = '1'] = source.fps.split('/')
    const fps = Number(numerator) / Math.max(1, Number(denominator))
    const codecFactor = settings.codec === 'h264' ? 0.075 : 0.05
    return Math.min(200, Math.max(0.5, source.width * source.height * fps * codecFactor / 1_000_000))
  }, [project.workflow.source_summary, settings.codec])

  useEffect(() => setSettings(project.workflow.export_settings), [project.project_id])

  const saveVerified = async (descriptor: VerifiedExportDto): Promise<void> => {
    if (platform.kind === 'browser') {
      if (authoritativeArtifactId !== descriptor.artifact_id
        || browserArtifact?.artifactId !== descriptor.artifact_id) {
        throw new Error('已验证视频仍在准备，请稍后再保存。')
      }
      return platform.saveExport(descriptor.filename, {
        kind: 'browser-download',
        blob: browserArtifact.blob,
      })
    } else {
      await platform.saveExport(descriptor.filename, {
        kind: 'local-export',
        path: `verified-export:${descriptor.artifact_id}`,
        saveTo: (destination) => backend.copyVerifiedExport(descriptor.artifact_id, destination),
      })
    }
  }

  const loadVerified = async (
    authority: number,
    expectedArtifactId: string,
  ): Promise<VerifiedExportDto | null> => {
    const descriptor = await backend.getVerifiedExport()
    if (verificationAuthority.current !== authority) return null
    if (!descriptor.verified || descriptor.artifact_id !== expectedArtifactId) {
      throw new Error('后端返回的导出已失效，请重新验证。')
    }
    if (platform.kind === 'browser') {
      const blob = await backend.fetchExportArtifact(descriptor.artifact_id)
      if (verificationAuthority.current !== authority) return null
      setBrowserArtifact({ artifactId: descriptor.artifact_id, blob })
    }
    if (verificationAuthority.current !== authority) return null
    setVerified(descriptor)
    return descriptor
  }

  useEffect(() => {
    const authority = ++verificationAuthority.current
    setVerified(null)
    setBrowserArtifact(null)
    if (authoritativeArtifactId === null) return
    void loadVerified(authority, authoritativeArtifactId).catch((error: unknown) => {
      if (verificationAuthority.current === authority) {
        onError(error instanceof Error ? error : '无法恢复已验证导出元数据。')
      }
    })
    return () => {
      if (verificationAuthority.current === authority) verificationAuthority.current += 1
    }
  }, [authoritativeArtifactId, backend, onError, platform.kind])

  useEffect(() => {
    if (activeTask?.target_stage !== 'export' || activeTask.status !== 'succeeded'
      || finalizedTask.current === activeTask.id) return
    finalizedTask.current = activeTask.id
    const saveAfterVerification = localExportIntent.current && platform.kind === 'tauri'
    localExportIntent.current = false
    setExporting(true)
    void backend.getVerifiedExport().then(async (descriptor) => {
      if (!descriptor.verified) throw new Error('后端未验证导出。')
      onProjectChange(await backend.getProject())
      if (saveAfterVerification) await saveVerified(descriptor)
    }).catch((error: unknown) => {
      finalizedTask.current = null
      onError(error instanceof Error ? error : '后端未能提供通过 ffprobe 验证的导出。')
    }).finally(() => setExporting(false))
  }, [activeTask, backend, onError, onProjectChange, platform])

  const start = async (): Promise<void> => {
    if (busy) return
    setExporting(true)
    localExportIntent.current = platform.kind === 'tauri'
    try {
      if (settingsDirty) {
        onProjectChange(await backend.updateProject({
          expected_project_id: project.project_id,
          export_settings: settings,
        }))
      }
      const task = await onStartStage('export')
      if (task.status === 'succeeded' && finalizedTask.current !== task.id) {
        finalizedTask.current = task.id
        localExportIntent.current = false
        if (platform.kind === 'tauri') {
          const descriptor = await backend.getVerifiedExport()
          if (!descriptor.verified) throw new Error('后端未验证导出。')
          onProjectChange(await backend.getProject())
          await saveVerified(descriptor)
        } else {
          const descriptor = await backend.getVerifiedExport()
          if (!descriptor.verified) throw new Error('后端未验证导出。')
          onProjectChange(await backend.getProject())
        }
      }
    } catch (error) {
      localExportIntent.current = false
      onError(error instanceof Error ? error : '导出任务失败。')
    } finally { setExporting(false) }
  }

  const authoritativeReady = project.stages.post_process?.status === 'succeeded'
  const exportRunning = activeTask?.target_stage === 'export'
    && ['queued', 'running'].includes(activeTask.status)
  const displayVerified = verified?.artifact_id === authoritativeArtifactId ? verified : null
  const canSave = displayVerified !== null && (
    platform.kind === 'tauri' || browserArtifact?.artifactId === displayVerified.artifact_id
  )
  return (
    <section aria-labelledby="export-title" className="page-grid export-page">
      <div className="page-heading">
        <p className="eyebrow">06 · EXPORT</p>
        <h2 id="export-title">导出成片</h2>
        <p>导出只读取权威处理后帧；修改编码不会重算合成或效果链。</p>
      </div>
      <article className="export-settings-card">
        <h3>编码配置</h3>
        <div className="export-settings-grid">
          <label>视频编码<select value={settings.codec} onChange={(event) => setSettings({ ...settings, codec: event.target.value as ExportEncodingSettings['codec'] })}><option value="h264">H.264 · 兼容优先</option><option value="h265">H.265 / HEVC · 更小文件</option></select></label>
          <label>码率模式<select value={settings.rate_control} onChange={(event) => setSettings({ ...settings, rate_control: event.target.value as ExportEncodingSettings['rate_control'] })}><option value="constant_quality">恒定质量</option><option value="two_pass_vbr">双遍 VBR</option></select></label>
          <label>压缩档位<select value={settings.compression_preset} onChange={(event) => setSettings({ ...settings, compression_preset: event.target.value as ExportEncodingSettings['compression_preset'] })}><option value="fast">快速</option><option value="balanced">均衡</option><option value="high_compression">高压缩</option></select></label>
          {settings.rate_control === 'constant_quality' ? <label>质量 <output>{settings.quality}</output><input max={100} min={1} onChange={(event) => setSettings({ ...settings, quality: Number(event.target.value) })} type="range" value={settings.quality} /></label> : <label>目标码率 <output>{settings.target_bitrate_mbps.toFixed(1)} Mbps</output><input max={200} min={0.5} onChange={(event) => setSettings({ ...settings, target_bitrate_mbps: Number(event.target.value) })} step={0.5} type="range" value={settings.target_bitrate_mbps} /><small>建议约 {recommendedBitrate.toFixed(1)} Mbps</small></label>}
        </div>
        <p>MP4 · 8-bit yuv420p limited-range · BT.709 · AAC 192 kbps</p>
      </article>
      <article className="export-card">
        <div className="export-orbit" aria-hidden="true"><span /></div>
        <div>
          <span className="asset-kicker">VERIFIED MP4</span>
          <h3>{displayVerified?.filename ?? '准备最终输出'}</h3>
          {displayVerified === null ? <p>文件路径不会进入浏览器 DTO；保存位置由平台对话框决定。</p> : (
            <dl className="asset-stats">
              <div><dt>帧数</dt><dd>{displayVerified.frame_count}</dd></div>
              <div><dt>时长</dt><dd>{displayVerified.duration_seconds.toFixed(2)} 秒</dd></div>
              <div><dt>音轨</dt><dd>{displayVerified.has_audio ? '已验证' : '无'}</dd></div>
            </dl>
          )}
          <div className="export-actions">
            <button disabled={!authoritativeReady || busy || exporting || exportRunning} onClick={() => void start()} type="button">{busy || exporting || exportRunning ? '验证导出中…' : '导出视频'}</button>
            {displayVerified !== null ? <button className="button-secondary" disabled={exporting || !canSave} onClick={() => void saveVerified(displayVerified).catch((error: unknown) => onError(error instanceof Error ? error : '保存导出失败。'))} type="button">保存已验证视频</button> : null}
          </div>
        </div>
      </article>
    </section>
  )
}
