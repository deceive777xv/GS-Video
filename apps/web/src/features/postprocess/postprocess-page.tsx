import {
  type ChangeEvent,
  type DragEvent,
  type PointerEvent as ReactPointerEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react'

import type { BackendClient } from '../../api/backend-client'
import type {
  EffectInstance,
  LibraryAssetRecordDto,
  ProjectDto,
  TaskDto,
} from '../../api/types'

interface PostProcessPageProps {
  activeTask: TaskDto | null
  backend: BackendClient
  busy: boolean
  project: ProjectDto
  onError(value: unknown): void
  onDirtyChange(dirty: boolean): void
  onProjectChange(project: ProjectDto): void
  onStartStage(stage: 'post_process'): Promise<TaskDto>
}

const cloneChain = (chain: EffectInstance[]): EffectInstance[] => structuredClone(chain)
const sameChain = (left: EffectInstance[], right: EffectInstance[]): boolean =>
  JSON.stringify(left) === JSON.stringify(right)

function baseEffect(): Pick<EffectInstance, 'instance_id' | 'params_version' | 'display_name' | 'enabled' | 'mix'> {
  return {
    instance_id: crypto.randomUUID(),
    params_version: 1,
    display_name: null,
    enabled: true,
    mix: 100,
  }
}

function newEffect(type: EffectInstance['type'], lutAssetId?: string): EffectInstance {
  const base = baseEffect()
  switch (type) {
    case 'primary_correction':
      return { ...base, type, parameters: { exposure: 0, contrast: 0, highlights: 0, shadows: 0, temperature: 0, tint: 0, saturation: 100, vibrance: 0 } }
    case 'lut_3d':
      if (lutAssetId === undefined) throw new Error('请选择托管 3D LUT。')
      return { ...base, type, parameters: { asset_id: lutAssetId } }
    case 'bloom':
      return { ...base, type, parameters: { threshold: 80, soft_knee: 50, radius: 32, intensity: 25, tint: [1, 1, 1] } }
    case 'vignette':
      return { ...base, type, parameters: { amount: 20, midpoint: 50, feather: 50, roundness: 0, center_x: 0, center_y: 0 } }
    case 'sharpen':
      return { ...base, type, parameters: { amount: 50, radius: 1, threshold: 1 } }
  }
}

const EFFECT_NAMES: Record<EffectInstance['type'], string> = {
  primary_correction: '基础校色',
  lut_3d: '3D LUT',
  bloom: '辉光',
  vignette: '暗角',
  sharpen: '锐化',
}

interface RangeProps {
  label: string
  value: number
  min: number
  max: number
  step?: number
  suffix?: string
  onBegin(): void
  onChange(value: number): void
  onEnd(): void
}

function RangeControl({ label, value, min, max, step = 1, suffix = '', onBegin, onChange, onEnd }: RangeProps) {
  const input = useRef<HTMLInputElement>(null)
  const output = useRef<HTMLOutputElement>(null)
  const liveValue = useRef(value)
  const editing = useRef(false)

  useEffect(() => {
    liveValue.current = value
    if (input.current !== null) input.current.value = String(value)
    if (output.current !== null) output.current.value = `${value}${suffix}`
  }, [suffix, value])

  const begin = (): void => {
    if (editing.current) return
    editing.current = true
    onBegin()
  }

  const finish = (): void => {
    if (!editing.current) return
    editing.current = false
    const next = liveValue.current
    onEnd()
    if (next !== value) onChange(next)
  }

  return (
    <label className="effect-range">
      <span>{label}<output ref={output}>{value}{suffix}</output></span>
      <input
        defaultValue={value}
        max={max}
        min={min}
        onBlur={finish}
        onInput={(event) => {
          begin()
          const next = event.currentTarget.valueAsNumber
          liveValue.current = next
          if (output.current !== null) output.current.value = `${next}${suffix}`
        }}
        onKeyDown={begin}
        onKeyUp={finish}
        onPointerDown={begin}
        onPointerCancel={finish}
        onPointerUp={finish}
        ref={input}
        step={step}
        type="range"
      />
    </label>
  )
}

export function PostProcessPage({ activeTask, backend, busy, project, onDirtyChange, onError, onProjectChange, onStartStage }: PostProcessPageProps) {
  const [draft, setDraft] = useState<EffectInstance[]>(() => cloneChain(project.workflow.effect_chain))
  const [saved, setSaved] = useState<EffectInstance[]>(() => cloneChain(project.workflow.effect_chain))
  const [revision, setRevision] = useState(project.workflow.effect_chain_revision)
  const [undo, setUndo] = useState<EffectInstance[][]>([])
  const [redo, setRedo] = useState<EffectInstance[][]>([])
  const [saving, setSaving] = useState(false)
  const [frameIndex, setFrameIndex] = useState(project.workflow.subject_prompt?.frame_index ?? 0)
  const [split, setSplit] = useState(50)
  const [bypass, setBypass] = useState(false)
  const [baseUrl, setBaseUrl] = useState<string | null>(null)
  const [processedUrl, setProcessedUrl] = useState<string | null>(null)
  const [previewBusy, setPreviewBusy] = useState(false)
  const [previewSuspended, setPreviewSuspended] = useState(false)
  const [videoUrl, setVideoUrl] = useState<string | null>(null)
  const [luts, setLuts] = useState<LibraryAssetRecordDto[]>([])
  const [selectedLut, setSelectedLut] = useState('')
  const [dragIndex, setDragIndex] = useState<number | null>(null)
  const [collapsed, setCollapsed] = useState<Set<string>>(() => new Set())
  const requestId = useRef(Date.now() * 1_000 + Math.floor(Math.random() * 1_000))
  const baseObjectUrl = useRef<string | null>(null)
  const processedObjectUrl = useRef<string | null>(null)
  const videoObjectUrl = useRef<string | null>(null)
  const continuousEdit = useRef<string | null>(null)
  const splitDrag = useRef<{ pointerId: number; offset: number } | null>(null)
  const finalizedTask = useRef<string | null>(null)
  const dirty = !sameChain(draft, saved)
  const postProcessAuthority = project.stages.post_process?.status === 'succeeded'
    ? project.stages.post_process.cache_key
    : null
  const frameCount = project.workflow.source_summary?.frame_count ?? 1
  const fps = useMemo(() => {
    const [numerator = '0', denominator = '1'] = (project.workflow.source_summary?.fps ?? '0/1').split('/')
    return Number(numerator) / Math.max(1, Number(denominator))
  }, [project.workflow.source_summary?.fps])

  useEffect(() => onDirtyChange(dirty), [dirty, onDirtyChange])
  useEffect(() => () => onDirtyChange(false), [onDirtyChange])

  useEffect(() => {
    const next = cloneChain(project.workflow.effect_chain)
    setDraft(next)
    setSaved(cloneChain(next))
    setRevision(project.workflow.effect_chain_revision)
    setUndo([])
    setRedo([])
  }, [project.project_id])

  useEffect(() => {
    void backend.listAssets('lut').then((items) => {
      const assets = items.map((item) => item.asset)
      setLuts(assets)
      setSelectedLut((current) => current || assets[0]?.asset_id || '')
    }).catch(onError)
  }, [backend, onError, project.project_id])

  useEffect(() => {
    const warn = (event: BeforeUnloadEvent): void => {
      if (!dirty) return
      event.preventDefault()
    }
    window.addEventListener('beforeunload', warn)
    return () => window.removeEventListener('beforeunload', warn)
  }, [dirty])

  const pushUndo = useCallback((): void => {
    setUndo((history) => [...history.slice(-49), cloneChain(draft)])
    setRedo([])
  }, [draft])

  const apply = useCallback((next: EffectInstance[]): void => {
    pushUndo()
    setDraft(next)
  }, [pushUndo])

  const beginContinuous = (key: string): void => {
    if (continuousEdit.current === key) return
    continuousEdit.current = key
    setPreviewSuspended(true)
    pushUndo()
  }
  const endContinuous = (): void => {
    if (continuousEdit.current === null) return
    continuousEdit.current = null
    setPreviewSuspended(false)
  }

  const undoOnce = useCallback((): void => {
    setUndo((history) => {
      const previous = history.at(-1)
      if (previous === undefined) return history
      setRedo((future) => [...future, cloneChain(draft)])
      setDraft(cloneChain(previous))
      return history.slice(0, -1)
    })
  }, [draft])
  const redoOnce = useCallback((): void => {
    setRedo((future) => {
      const next = future.at(-1)
      if (next === undefined) return future
      setUndo((history) => [...history, cloneChain(draft)])
      setDraft(cloneChain(next))
      return future.slice(0, -1)
    })
  }, [draft])

  useEffect(() => {
    const keyboard = (event: KeyboardEvent): void => {
      if (!(event.ctrlKey || event.metaKey) || event.key.toLowerCase() !== 'z') return
      event.preventDefault()
      if (event.shiftKey) redoOnce()
      else undoOnce()
    }
    window.addEventListener('keydown', keyboard)
    return () => window.removeEventListener('keydown', keyboard)
  }, [redoOnce, undoOnce])

  const setEffect = (index: number, effect: EffectInstance): void => {
    setDraft((current) => current.map((item, itemIndex) => itemIndex === index ? effect : item))
  }
  const setParameter = (index: number, key: string, value: number): void => {
    const effect = draft[index]
    if (effect === undefined) return
    setEffect(index, {
      ...effect,
      parameters: { ...effect.parameters, [key]: value },
    } as EffectInstance)
  }

  const updateSplitFromClientX = useCallback((control: HTMLDivElement, clientX: number): void => {
    const frame = control.parentElement
    if (frame === null) return
    const bounds = frame.getBoundingClientRect()
    if (bounds.width <= 0) return
    const dragOffset = splitDrag.current?.offset ?? 0
    const next = ((clientX - dragOffset - bounds.left) / bounds.width) * 100
    setSplit(Math.min(100, Math.max(0, next)))
  }, [])

  const beginSplitDrag = (event: ReactPointerEvent<HTMLDivElement>): void => {
    if (event.button !== 0) return
    const frame = event.currentTarget.parentElement
    if (frame === null) return
    const bounds = frame.getBoundingClientRect()
    const dividerX = bounds.left + (bounds.width * split / 100)
    splitDrag.current = { pointerId: event.pointerId, offset: event.clientX - dividerX }
    event.currentTarget.setPointerCapture?.(event.pointerId)
  }

  const moveSplitDrag = (event: ReactPointerEvent<HTMLDivElement>): void => {
    if (splitDrag.current?.pointerId !== event.pointerId) return
    updateSplitFromClientX(event.currentTarget, event.clientX)
  }

  const endSplitDrag = (event: ReactPointerEvent<HTMLDivElement>): void => {
    if (splitDrag.current?.pointerId !== event.pointerId) return
    updateSplitFromClientX(event.currentTarget, event.clientX)
    splitDrag.current = null
    if (event.currentTarget.hasPointerCapture?.(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId)
    }
  }

  const cancelSplitDrag = (event: ReactPointerEvent<HTMLDivElement>): void => {
    if (splitDrag.current?.pointerId !== event.pointerId) return
    splitDrag.current = null
    if (event.currentTarget.hasPointerCapture?.(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId)
    }
  }

  useEffect(() => {
    if (previewSuspended) return
    const controller = new AbortController()
    const timer = window.setTimeout(() => {
      setPreviewBusy(true)
      void (async () => {
        const baseBlob = await backend.renderDraftPostProcessPreview({
          expected_project_id: project.project_id,
          request_id: ++requestId.current,
          frame_index: frameIndex,
          maximum_width: 1280,
          maximum_height: 720,
          effect_chain: draft,
          bypass: true,
        }, controller.signal)
        if (controller.signal.aborted) return
        const base = URL.createObjectURL(baseBlob)
        if (baseObjectUrl.current !== null) URL.revokeObjectURL(baseObjectUrl.current)
        baseObjectUrl.current = base
        setBaseUrl(base)
        if (bypass) {
          if (processedObjectUrl.current !== null) URL.revokeObjectURL(processedObjectUrl.current)
          processedObjectUrl.current = null
          setProcessedUrl(null)
          return
        }
        const processedBlob = await backend.renderDraftPostProcessPreview({
          expected_project_id: project.project_id,
          request_id: ++requestId.current,
          frame_index: frameIndex,
          maximum_width: 1280,
          maximum_height: 720,
          effect_chain: draft,
          bypass: false,
        }, controller.signal)
        if (controller.signal.aborted) return
        const processed = URL.createObjectURL(processedBlob)
        if (processedObjectUrl.current !== null) URL.revokeObjectURL(processedObjectUrl.current)
        processedObjectUrl.current = processed
        setProcessedUrl(processed)
      })().catch((error: unknown) => {
        if (!controller.signal.aborted) onError(error)
      }).finally(() => {
        if (!controller.signal.aborted) setPreviewBusy(false)
      })
    }, 120)
    return () => { window.clearTimeout(timer); controller.abort() }
  }, [backend, bypass, draft, frameIndex, onError, previewSuspended, project.project_id])

  useEffect(() => () => {
    if (baseObjectUrl.current !== null) URL.revokeObjectURL(baseObjectUrl.current)
    if (processedObjectUrl.current !== null) URL.revokeObjectURL(processedObjectUrl.current)
    if (videoObjectUrl.current !== null) URL.revokeObjectURL(videoObjectUrl.current)
    void backend.closePostProcessPreview().catch(() => undefined)
  }, [backend])

  const save = async (): Promise<ProjectDto> => {
    setSaving(true)
    try {
      const next = await backend.updateProject({
        expected_project_id: project.project_id,
        effect_chain: draft,
        expected_effect_chain_revision: revision,
      })
      setSaved(cloneChain(next.workflow.effect_chain))
      setDraft(cloneChain(next.workflow.effect_chain))
      setRevision(next.workflow.effect_chain_revision)
      setUndo([])
      setRedo([])
      onProjectChange(next)
      return next
    } finally {
      setSaving(false)
    }
  }

  const generate = async (): Promise<void> => {
    try {
      await save()
      const task = await onStartStage('post_process')
      if (task.status === 'succeeded') onProjectChange(await backend.getProject())
    } catch (error) {
      onError(error)
    }
  }

  useEffect(() => {
    if (activeTask?.target_stage !== 'post_process' || activeTask.status !== 'succeeded'
      || finalizedTask.current === activeTask.id) return
    finalizedTask.current = activeTask.id
    void backend.getProject().then(onProjectChange).catch(onError)
  }, [activeTask, backend, onError, onProjectChange])

  useEffect(() => {
    const controller = new AbortController()
    if (videoObjectUrl.current !== null) {
      URL.revokeObjectURL(videoObjectUrl.current)
      videoObjectUrl.current = null
    }
    setVideoUrl(null)
    if (postProcessAuthority === null) return () => controller.abort()
    void (async () => {
      const descriptor = await backend.getCompositePreview(controller.signal)
      const blob = await backend.fetchCompositePreviewArtifact(
        descriptor.artifact_id,
        controller.signal,
      )
      if (controller.signal.aborted) return
      const url = URL.createObjectURL(blob)
      if (controller.signal.aborted) {
        URL.revokeObjectURL(url)
        return
      }
      videoObjectUrl.current = url
      setVideoUrl(url)
    })().catch((error: unknown) => {
      if (!controller.signal.aborted) onError(error)
    })
    return () => controller.abort()
  }, [backend, onError, postProcessAuthority])

  const move = (from: number, to: number): void => {
    if (from === to || to < 0 || to >= draft.length) return
    const next = cloneChain(draft)
    const [item] = next.splice(from, 1)
    if (item === undefined) return
    next.splice(to, 0, item)
    apply(next)
  }

  const uploadLut = async (event: ChangeEvent<HTMLInputElement>): Promise<void> => {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (file === undefined) return
    try {
      const digest = await crypto.subtle.digest('SHA-256', await file.arrayBuffer())
      const sha256 = [...new Uint8Array(digest)].map((value) => value.toString(16).padStart(2, '0')).join('')
      const upload = await backend.createUpload({
        kind: 'lut_3d', filename: file.name, mime_type: file.type || 'text/plain',
        total_size: file.size, sha256, assign_to_current: false,
      })
      for (let offset = 0, index = 0; offset < file.size; offset += upload.chunk_size, index += 1) {
        await backend.putUploadChunk(upload.id, index, file.slice(offset, offset + upload.chunk_size))
      }
      const completed = await backend.completeUpload(upload.id)
      const items = await backend.listAssets('lut')
      const assets = items.map((item) => item.asset)
      const imported = assets.find((asset) => asset.sha256 === completed.sha256)
      setLuts(assets)
      if (imported === undefined) throw new Error('LUT 已导入，但素材库未返回对应条目。')
      setSelectedLut(imported.asset_id)
    } catch (error) { onError(error) }
  }

  const renderParameters = (effect: EffectInstance, index: number) => {
    const range = (label: string, key: string, value: number, min: number, max: number, step = 1, suffix = '') => (
      <RangeControl key={key} label={label} max={max} min={min} onBegin={() => beginContinuous(`${effect.instance_id}:${key}`)} onChange={(next) => setParameter(index, key, next)} onEnd={endContinuous} step={step} suffix={suffix} value={value} />
    )
    switch (effect.type) {
      case 'primary_correction': return <>{range('曝光', 'exposure', effect.parameters.exposure, -5, 5, 0.1, ' EV')}{range('对比度', 'contrast', effect.parameters.contrast, -100, 100)}{range('高光', 'highlights', effect.parameters.highlights, -100, 100)}{range('阴影', 'shadows', effect.parameters.shadows, -100, 100)}{range('色温', 'temperature', effect.parameters.temperature, -100, 100)}{range('色调', 'tint', effect.parameters.tint, -100, 100)}{range('饱和度', 'saturation', effect.parameters.saturation, 0, 200, 1, '%')}{range('自然饱和度', 'vibrance', effect.parameters.vibrance, -100, 100)}</>
      case 'lut_3d': return <label className="effect-select">托管 LUT<select value={effect.parameters.asset_id} onChange={(event) => { pushUndo(); setEffect(index, { ...effect, parameters: { asset_id: event.target.value } }) }}>{luts.map((lut) => <option key={lut.asset_id} value={lut.asset_id}>{lut.original_filename}</option>)}</select></label>
      case 'bloom': {
        const color = `#${effect.parameters.tint.map((value) => Math.round(value * 255).toString(16).padStart(2, '0')).join('')}`
        return <>{range('阈值', 'threshold', effect.parameters.threshold, 0, 200, 1, '%')}{range('柔和过渡', 'soft_knee', effect.parameters.soft_knee, 0, 100, 1, '%')}{range('扩散半径', 'radius', effect.parameters.radius, 1, 256, 1, ' px')}{range('强度', 'intensity', effect.parameters.intensity, 0, 400, 1, '%')}<label className="effect-color">辉光颜色<input aria-label="辉光颜色" onChange={(event) => { pushUndo(); const hex = event.target.value.slice(1); setEffect(index, { ...effect, parameters: { ...effect.parameters, tint: ([0, 2, 4].map((offset) => Number.parseInt(hex.slice(offset, offset + 2), 16) / 255) as [number, number, number]) } }) }} type="color" value={color} /></label></>
      }
      case 'vignette': return <>{range('强度', 'amount', effect.parameters.amount, 0, 100, 1, '%')}{range('中点', 'midpoint', effect.parameters.midpoint, 0, 100, 1, '%')}{range('羽化', 'feather', effect.parameters.feather, 0, 100, 1, '%')}{range('圆度', 'roundness', effect.parameters.roundness, -100, 100)}{range('中心 X', 'center_x', effect.parameters.center_x, -100, 100, 1, '%')}{range('中心 Y', 'center_y', effect.parameters.center_y, -100, 100, 1, '%')}</>
      case 'sharpen': return <>{range('强度', 'amount', effect.parameters.amount, 0, 300, 1, '%')}{range('半径', 'radius', effect.parameters.radius, 0.1, 10, 0.1, ' px')}{range('阈值', 'threshold', effect.parameters.threshold, 0, 10, 0.1, '%')}</>
    }
  }

  return (
    <section aria-labelledby="postprocess-title" className="postprocess-page">
      <header className="page-heading postprocess-heading">
        <div><p className="eyebrow">05 · POST PROCESS</p><h2 id="postprocess-title">合成后处理</h2><p>效果按右栏顺序组合；代表帧与整片任务使用同一后端数学引擎。</p></div>
        <div className="postprocess-toolbar">
          <button className="button-secondary" disabled={undo.length === 0} onClick={undoOnce} type="button">撤销</button>
          <button className="button-secondary" disabled={redo.length === 0} onClick={redoOnce} type="button">重做</button>
          <button disabled={!dirty || saving || busy} onClick={() => void save().catch(onError)} type="button">{saving ? '保存中…' : '保存效果链'}</button>
        </div>
      </header>
      <div className="postprocess-workbench">
        <article className="postprocess-viewer">
          <div className="comparison-frame" aria-busy={previewBusy}>
            {baseUrl === null ? <div className="empty-state">正在准备代表帧…</div> : <img alt="基础合成代表帧" draggable={false} src={baseUrl} />}
            {processedUrl === null ? null : <div className="comparison-after" style={{ clipPath: `inset(0 ${100 - split}% 0 0)` }}><img alt="处理后代表帧" draggable={false} src={processedUrl} /></div>}
            {processedUrl === null ? null : <div
              aria-label="前后分割线"
              aria-orientation="horizontal"
              aria-valuemax={100}
              aria-valuemin={0}
              aria-valuenow={Math.round(split)}
              aria-valuetext={`${Math.round(split)}%`}
              className="comparison-divider"
              onKeyDown={(event) => {
                const increment = event.shiftKey ? 5 : 1
                if (event.key === 'ArrowLeft' || event.key === 'ArrowDown') {
                  event.preventDefault()
                  setSplit((current) => Math.max(0, current - increment))
                } else if (event.key === 'ArrowRight' || event.key === 'ArrowUp') {
                  event.preventDefault()
                  setSplit((current) => Math.min(100, current + increment))
                } else if (event.key === 'Home') {
                  event.preventDefault()
                  setSplit(0)
                } else if (event.key === 'End') {
                  event.preventDefault()
                  setSplit(100)
                }
              }}
              onPointerCancel={cancelSplitDrag}
              onPointerDown={beginSplitDrag}
              onPointerMove={moveSplitDrag}
              onPointerUp={endSplitDrag}
              role="slider"
              style={{ left: `${split}%` }}
              tabIndex={0}
            ><span aria-hidden="true">{Math.round(split)}%</span></div>}
          </div>
          <label className="frame-scrubber">代表帧 <input max={Math.max(0, frameCount - 1)} min={0} onChange={(event) => setFrameIndex(Number(event.target.value))} type="range" value={frameIndex} /><output>{frameIndex + 1} / {frameCount} · {fps > 0 ? (frameIndex / fps).toFixed(2) : '0.00'}s</output></label>
          <label className="toggle-row"><input checked={bypass} onChange={(event) => setBypass(event.target.checked)} type="checkbox" />临时旁路整条效果链</label>
          {videoUrl === null ? null : <div className="postprocess-video-frame"><video className="postprocess-video" controls src={videoUrl} /></div>}
        </article>
        <aside className="effect-stack" aria-label="效果链">
          <div className="effect-adders">
            {(['primary_correction', 'bloom', 'vignette', 'sharpen'] as const).map((type) => <button disabled={draft.length >= 32} key={type} onClick={() => apply([...draft, newEffect(type)])} type="button">+ {EFFECT_NAMES[type]}</button>)}
            <div className="lut-adder"><select aria-label="选择 3D LUT" value={selectedLut} onChange={(event) => setSelectedLut(event.target.value)}><option value="">选择 LUT</option>{luts.map((lut) => <option key={lut.asset_id} value={lut.asset_id}>{lut.original_filename}</option>)}</select><button disabled={selectedLut === '' || draft.length >= 32} onClick={() => apply([...draft, newEffect('lut_3d', selectedLut)])} type="button">+ LUT</button><label className="button-secondary lut-upload">导入 .cube<input accept=".cube" onChange={(event) => void uploadLut(event)} type="file" /></label></div>
          </div>
          {draft.length === 0 ? <div className="empty-state"><strong>无效果</strong><p>空链仍会生成权威处理后帧。</p></div> : null}
          {draft.map((effect, index) => (
            <article className={`effect-card ${effect.enabled ? '' : 'is-disabled'}`} key={effect.instance_id} onDragOver={(event: DragEvent) => event.preventDefault()} onDrop={() => { if (dragIndex !== null) move(dragIndex, index); setDragIndex(null) }}>
              <header><span aria-hidden="true" className="drag-handle" draggable onDragEnd={() => setDragIndex(null)} onDragStart={(event) => { event.dataTransfer.effectAllowed = 'move'; setDragIndex(index) }}>⠿</span><button aria-expanded={!collapsed.has(effect.instance_id)} aria-label={`${collapsed.has(effect.instance_id) ? '展开' : '折叠'} ${effect.display_name || EFFECT_NAMES[effect.type]}`} className="effect-collapse" onClick={() => setCollapsed((current) => { const next = new Set(current); if (next.has(effect.instance_id)) next.delete(effect.instance_id); else next.add(effect.instance_id); return next })} type="button">{collapsed.has(effect.instance_id) ? '▸' : '▾'}</button><input aria-label="效果名称" placeholder={EFFECT_NAMES[effect.type]} value={effect.display_name ?? ''} onChange={(event) => setEffect(index, { ...effect, display_name: event.target.value || null })} onFocus={pushUndo} /><label><input checked={effect.enabled} onChange={(event) => { pushUndo(); setEffect(index, { ...effect, enabled: event.target.checked }) }} type="checkbox" />启用</label></header>
              {collapsed.has(effect.instance_id) ? null : <><RangeControl label="Mix" max={100} min={0} onBegin={() => beginContinuous(`${effect.instance_id}:mix`)} onChange={(mix) => setEffect(index, { ...effect, mix })} onEnd={endContinuous} suffix="%" value={effect.mix} />
                {renderParameters(effect, index)}
                <footer><button disabled={index === 0} onClick={() => move(index, index - 1)} type="button">上移</button><button disabled={index === draft.length - 1} onClick={() => move(index, index + 1)} type="button">下移</button><button disabled={draft.length >= 32} onClick={() => { const copy = cloneChain([effect])[0]; if (copy !== undefined) apply([...draft.slice(0, index + 1), { ...copy, instance_id: crypto.randomUUID(), display_name: copy.display_name === null ? null : `${copy.display_name} 副本` } as EffectInstance, ...draft.slice(index + 1)]) }} type="button">复制</button><button onClick={() => { const reset = newEffect(effect.type, effect.type === 'lut_3d' ? effect.parameters.asset_id : undefined); apply(draft.map((item, itemIndex) => itemIndex === index ? { ...reset, instance_id: effect.instance_id, display_name: effect.display_name } as EffectInstance : item)) }} type="button">重置</button><button className="button-danger" onClick={() => apply(draft.filter((_, itemIndex) => itemIndex !== index))} type="button">删除</button></footer></>}
            </article>
          ))}
          <button className="button-secondary clear-chain" disabled={draft.length === 0} onClick={() => apply([])} type="button">清空效果链</button>
        </aside>
      </div>
      <footer className="postprocess-actions"><span>{dirty ? '有未保存修改' : `已保存 · revision ${revision}`}</span><button disabled={busy || saving} onClick={() => void generate()} type="button">生成完整处理后预览</button></footer>
    </section>
  )
}
