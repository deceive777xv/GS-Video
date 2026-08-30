import { useEffect, useRef, useState } from 'react'

import type { BackendClient } from '../../api/backend-client'
import type { MatteRefinementSettings, ProjectDto, TaskDto, TaskEvent } from '../../api/types'

interface PreviewPageProps {
  backend: BackendClient
  busy: boolean
  latestEvent: TaskEvent | null
  project: ProjectDto
  activeTask: TaskDto | null
  onError(value: unknown): void
  onProjectChange(project: ProjectDto): void
  onStartStage(stage: 'post_process'): Promise<unknown>
  onBackToCamera(): void
  onReselectSubject(): void
}

const DEFAULT_MATTE: MatteRefinementSettings = {
  enabled: true,
  edge_offset: -1,
  feather_radius: 1,
  decontaminate_strength: 0,
  decontaminate_radius: 3,
}

export function PreviewPage({
  backend, busy, project, activeTask, latestEvent, onError, onProjectChange, onStartStage,
  onBackToCamera, onReselectSubject,
}: PreviewPageProps) {
  const [frameUrl, setFrameUrl] = useState<string | null>(null)
  const [draftUrl, setDraftUrl] = useState<string | null>(null)
  const [draftPending, setDraftPending] = useState(false)
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
  const [scaleText, setScaleText] = useState(String(project.workflow.gs_scale))
  const [azimuthText, setAzimuthText] = useState(String(project.workflow.scene_azimuth))
  const authoritativeMatte = project.workflow.matte_refinement ?? DEFAULT_MATTE
  const [matte, setMatte] = useState<MatteRefinementSettings>(() => ({
    ...authoritativeMatte,
  }))
  const frameObjectUrl = useRef<string | null>(null)
  const draftObjectUrl = useRef<string | null>(null)
  const draftRequest = useRef(0)
  const compositeObjectUrl = useRef<string | null>(null)
  const compositeRequest = useRef(0)
  const descriptorAuthority = useRef<string | null>(null)
  const preview = project.workflow.preview
  const postProcess = project.stages.post_process
  const groundConfirmed = project.workflow.target_ground?.confirmed === true
  const previewAuthority = postProcess?.status === 'succeeded'
    && postProcess.cache_key !== null
    && postProcess.cache_key !== undefined
    ? postProcess.cache_key
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
  const scaleDraft = Number(scaleText)
  const azimuthDraft = Number(azimuthText)
  const alignmentValid = Number.isFinite(scaleDraft) && scaleDraft >= 0.001 && scaleDraft <= 1000
    && Number.isFinite(azimuthDraft) && azimuthDraft >= -180 && azimuthDraft <= 180
  const alignmentSaved = alignmentValid
    && scaleDraft === project.workflow.gs_scale
    && azimuthDraft === project.workflow.scene_azimuth
  const matteSaved = JSON.stringify(matte) === JSON.stringify(authoritativeMatte)
  const colorReady = project.workflow.source_color_interpretation !== null
  const draftValid = cropValid && alignmentValid && groundConfirmed
    && project.stages.solve_camera?.status === 'succeeded'
    && project.stages.segment?.status === 'succeeded'
  const draftDirty = draftValid && (!cropSaved || !alignmentSaved || !matteSaved)
  const showDraft = previewAuthority === null || draftDirty
  const draftKey = draftValid ? JSON.stringify([
    project.project_id,
    project.stages.solve_camera?.cache_key,
    project.stages.segment?.cache_key,
    project.workflow.target_ground?.revision,
    project.workflow.subject_prompt?.frame_index,
    scaleDraft,
    azimuthDraft,
    cropDraft.x,
    cropDraft.y,
    cropDraft.width,
    cropDraft.height,
    matte,
  ]) : null

  useEffect(() => {
    if (authoritativeCrop === null) return
    setCropText({
      x: String(authoritativeCrop.x), y: String(authoritativeCrop.y),
      width: String(authoritativeCrop.width), height: String(authoritativeCrop.height),
    })
  }, [authoritativeCrop?.x, authoritativeCrop?.y, authoritativeCrop?.width, authoritativeCrop?.height])

  useEffect(() => {
    setScaleText(String(project.workflow.gs_scale))
    setAzimuthText(String(project.workflow.scene_azimuth))
  }, [project.workflow.gs_scale, project.workflow.scene_azimuth])

  useEffect(() => {
    setMatte({ ...authoritativeMatte })
  }, [
    authoritativeMatte.enabled,
    authoritativeMatte.edge_offset,
    authoritativeMatte.feather_radius,
    authoritativeMatte.decontaminate_strength,
    authoritativeMatte.decontaminate_radius,
  ])

  useEffect(() => {
    if (frameObjectUrl.current !== null) {
      URL.revokeObjectURL(frameObjectUrl.current)
      frameObjectUrl.current = null
    }
    setFrameUrl(null)
    if (preview === null || previewAuthority !== null) return
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
  }, [backend, onError, preview?.artifact_id, previewAuthority])

  useEffect(() => () => {
    if (frameObjectUrl.current !== null) URL.revokeObjectURL(frameObjectUrl.current)
    if (draftObjectUrl.current !== null) URL.revokeObjectURL(draftObjectUrl.current)
  }, [])

  useEffect(() => {
    const request = ++draftRequest.current
    if (!showDraft || draftKey === null) {
      setDraftPending(false)
      return
    }
    const controller = new AbortController()
    setDraftPending(true)
    const timer = window.setTimeout(() => {
      void backend.renderDraftCompositePreview({
        expected_project_id: project.project_id,
        request_id: request,
        maximum_width: 960,
        maximum_height: 540,
        gs_scale: scaleDraft,
        scene_azimuth: azimuthDraft,
        output_crop: cropDraft,
        matte_refinement: matte,
      }, controller.signal).then((blob) => {
        if (controller.signal.aborted || draftRequest.current !== request) return
        const nextUrl = URL.createObjectURL(blob)
        if (controller.signal.aborted || draftRequest.current !== request) {
          URL.revokeObjectURL(nextUrl)
          return
        }
        if (draftObjectUrl.current !== null) URL.revokeObjectURL(draftObjectUrl.current)
        draftObjectUrl.current = nextUrl
        setDraftUrl(nextUrl)
      }).catch((error: unknown) => {
        if (controller.signal.aborted || draftRequest.current !== request) return
        onError(error instanceof Error ? error : '无法生成虚拟相机代表帧预览。')
      }).finally(() => {
        if (!controller.signal.aborted && draftRequest.current === request) {
          setDraftPending(false)
        }
      })
    }, 120)
    return () => {
      window.clearTimeout(timer)
      controller.abort()
    }
  }, [backend, draftKey, showDraft])

  useEffect(() => {
    const request = ++compositeRequest.current
    descriptorAuthority.current = null
    if (compositeObjectUrl.current !== null) {
      URL.revokeObjectURL(compositeObjectUrl.current)
      compositeObjectUrl.current = null
    }
    setCompositeVideo(null)
    if (previewAuthority === null) return

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
        authority: previewAuthority,
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
  }, [backend, onError, previewAuthority])

  const generate = async (): Promise<void> => {
    if (busy || !cropValid || !alignmentValid || !groundConfirmed || !colorReady) return
    setRunning(true)
    try {
      if (!cropSaved || !alignmentSaved || !matteSaved) {
        onProjectChange(await backend.updateProject({
          expected_project_id: project.project_id,
          gs_scale: scaleDraft,
          scene_azimuth: azimuthDraft,
          output_crop: cropDraft,
          matte_refinement: matte,
        }))
      }
      await onStartStage('post_process')
    }
    catch (error) { onError(error instanceof Error ? error : '预览合成任务失败。') }
    finally { setRunning(false) }
  }

  const saveDraft = async (): Promise<void> => {
    if (!cropValid || !alignmentValid || busy || running) return
    setRunning(true)
    try {
      onProjectChange(await backend.updateProject({
        expected_project_id: project.project_id,
        gs_scale: scaleDraft,
        scene_azimuth: azimuthDraft,
        output_crop: cropDraft,
        matte_refinement: matte,
      }))
    } catch (error) {
      onError(error instanceof Error ? error : '无法保存轨迹映射与固定输出裁剪。')
    } finally {
      setRunning(false)
    }
  }

  const displayedComposite = compositeVideo?.authority === previewAuthority
    ? compositeVideo
    : null
  const previewRunning = activeTask?.target_stage === 'post_process'
    && ['queued', 'running'].includes(activeTask.status)
  const stageRows = ['map_trajectory', 'render', 'composite', 'post_process'] as const
  const eventError = latestEvent?.type === 'task_event'
    && latestEvent.task_id === activeTask?.id
    ? latestEvent.error
    : null
  const eventCode = typeof eventError?.code === 'string' ? eventError.code : null
  const eventCategory = typeof eventError?.category === 'string' ? eventError.category : null
  const recoveryTarget = activeTask?.target_stage ?? 'post_process'
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
    || recoveryTarget === 'post_process'
    || /(composite|encode|preview)/.test(recoveryKey)
  )
  const retryable = activeTask?.status === 'failed'
    && eventError?.retryable === true
    && activeTask.target_stage === 'post_process'
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
            {previewAuthority !== null && !showDraft ? (
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
            ) : draftUrl !== null ? (
              <img alt="虚拟相机代表帧合成预览" src={draftUrl} />
            ) : frameUrl === null ? (
              <div className="viewport-empty">{draftPending ? '正在生成虚拟相机代表帧…' : '暂无可用的合成代表帧'}</div>
            ) : <>
                <img alt="相机参考帧（非合成视频）" src={frameUrl} />
                <span aria-label="固定输出裁剪预览" className="fixed-output-crop-preview" />
                {draftPending ? <span className="viewport-status">正在刷新真实合成代表帧…</span> : null}
              </>}
          </div>
          <div className="preview-caption">
            <span>{previewAuthority !== null && !showDraft ? '后端验证 · 低分辨率合成' : draftUrl !== null ? '当前参数 · 代表帧真实合成' : draftPending ? '相机参考 · 等待代表帧' : '相机参考 · 非合成视频'}</span>
            <span>ViPE 逐帧内参</span>
            <span>目标地面 r{project.workflow.target_ground?.revision ?? '—'}</span>
          </div>
          <fieldset className="preview-alignment-controls" disabled={busy || running}>
            <legend>轨迹映射</legend>
            <div className="preview-alignment-grid">
              <label>
                <span>GS 比例</span>
                <input aria-label="GS 比例" max="1000" min="0.001" onChange={(event) => setScaleText(event.currentTarget.value)} step="0.01" type="number" value={scaleText} />
              </label>
              <label>
                <span className="preview-field-heading">场景方位角 <small>负值逆时针 · 正值顺时针</small></span>
                <div className="range-number">
                  <input
                    aria-label="场景方位角滑杆"
                    max="180"
                    min="-180"
                    onChange={(event) => setAzimuthText(event.currentTarget.value)}
                    step="1"
                    type="range"
                    value={Number.isFinite(azimuthDraft) ? Math.min(180, Math.max(-180, azimuthDraft)) : 0}
                  />
                  <input aria-label="场景方位角" max="180" min="-180" onChange={(event) => setAzimuthText(event.currentTarget.value)} step="1" type="number" value={azimuthText} />
                </div>
              </label>
            </div>
          </fieldset>
        </article>
        <aside className="viewport-controls preview-controls">
          <h3>全片合成</h3>
          <fieldset disabled={busy || running}>
            <legend>固定输出裁剪</legend>
            <div className="camera-readout preview-crop-grid">
              {(['x', 'y', 'width', 'height'] as const).map((key) => <label key={key}>{key.toUpperCase()}<input aria-label={`输出裁剪 ${key.toUpperCase()}`} onChange={(event) => {
                const value = event.currentTarget.value
                setCropText((current) => ({ ...current, [key]: value }))
              }} step={key === 'width' || key === 'height' ? 2 : 1} type="number" value={cropText[key]} /></label>)}
            </div>
          </fieldset>
          <fieldset className="matte-controls" disabled={busy || running}>
            <legend>抠像修边（固定合成步骤）</legend>
            <label className="toggle-row"><input checked={matte.enabled} onChange={(event) => setMatte({ ...matte, enabled: event.target.checked })} type="checkbox" />启用修边</label>
            <label>边缘偏移 <output>{matte.edge_offset.toFixed(1)} px</output><input disabled={!matte.enabled} max={20} min={-20} onChange={(event) => setMatte({ ...matte, edge_offset: Number(event.target.value) })} step={0.5} type="range" value={matte.edge_offset} /></label>
            <label>羽化 <output>{matte.feather_radius.toFixed(1)} px</output><input disabled={!matte.enabled} max={20} min={0} onChange={(event) => setMatte({ ...matte, feather_radius: Number(event.target.value) })} step={0.5} type="range" value={matte.feather_radius} /></label>
            <label>去色边强度 <output>{matte.decontaminate_strength.toFixed(0)}%</output><input disabled={!matte.enabled} max={100} min={0} onChange={(event) => setMatte({ ...matte, decontaminate_strength: Number(event.target.value) })} type="range" value={matte.decontaminate_strength} /></label>
            <label>去色边半径 <output>{matte.decontaminate_radius.toFixed(1)} px</output><input disabled={!matte.enabled || matte.decontaminate_strength === 0} max={20} min={1} onChange={(event) => setMatte({ ...matte, decontaminate_radius: Number(event.target.value) })} step={0.5} type="range" value={matte.decontaminate_radius} /></label>
          </fieldset>
          {!colorReady ? <div className="color-confirmation"><strong>源视频缺少完整 Rec.709 标记</strong><p>首版只处理 SDR Rec.709。请确认将该素材按 Rec.709 解释后再生成合成。</p><button disabled={busy || running} onClick={() => void backend.updateProject({ expected_project_id: project.project_id, source_color_interpretation: 'assumed_rec709' }).then(onProjectChange).catch(onError)} type="button">确认按 Rec.709 解释</button></div> : null}
          <p className="technical-note">X/Y 使用源画面像素坐标，可为负数；裁剪框允许超出源视频范围，外部区域由 GS 背景填充。宽高须为偶数，最大 3840×2160。</p>
          <div className="preview-actions">
            <button disabled={busy || running || !cropValid || !alignmentValid || (cropSaved && alignmentSaved && matteSaved)} onClick={() => void saveDraft()} type="button">保存参数</button>
            <button disabled={busy || running || previewRunning || !groundConfirmed || !cropValid || !alignmentValid || !colorReady} onClick={() => void generate()} type="button">{busy || running || previewRunning ? '生成中…' : '生成预览'}</button>
          </div>
          <div className="stage-list" aria-label="预览阶段缓存状态">
            {stageRows.map((name) => <div key={name}><span>{name}</span><strong>{project.stages[name]?.status ?? 'pending'}</strong></div>)}
          </div>
        </aside>
      </div>
      <div className="honest-state">
        <strong>{previewAuthority !== null && !showDraft ? '预览阶段已由后端确认' : '代表帧随参数实时更新'}</strong>
        <p>{previewAuthority !== null && !showDraft
          ? '播放器只使用当前成功合成缓存对应的后端验证视频。'
          : '代表帧使用当前 GS 比例、方位角和固定输出裁剪；生成预览时会保存这些参数并执行全片低分辨率合成。'}</p>
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
