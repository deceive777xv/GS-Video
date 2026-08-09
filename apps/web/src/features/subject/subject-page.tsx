import { type PointerEvent, useEffect, useRef, useState } from 'react'

import type { BackendClient } from '../../api/backend-client'
import { BackendClientError } from '../../api/http-backend-client'
import type { ProjectDto, SubjectMediaDto } from '../../api/types'
import { toImagePoint } from '../camera/scene-viewport'

const SUBJECT_MEDIA_RETRY_DELAYS_MS = [250, 500, 1_000] as const

interface SubjectPageProps {
  backend: BackendClient
  busy: boolean
  project: ProjectDto
  onError(value: unknown): void
  onProjectChange(project: ProjectDto): void
  onStartStage(stage: 'segment'): Promise<unknown>
}

export function SubjectPage({ backend, busy, project, onError, onProjectChange, onStartStage }: SubjectPageProps) {
  const [proxy, setProxy] = useState<SubjectMediaDto | null>(null)
  const [proxyUrl, setProxyUrl] = useState<string | null>(null)
  const [alphaUrl, setAlphaUrl] = useState<string | null>(null)
  const [x, setX] = useState('')
  const [y, setY] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const frameRef = useRef<HTMLDivElement>(null)
  const proxyUrlRef = useRef<string | null>(null)
  const alphaUrlRef = useRef<string | null>(null)
  const onErrorRef = useRef(onError)

  useEffect(() => { onErrorRef.current = onError }, [onError])

  const replaceUrl = (role: 'proxy' | 'alpha', blob: Blob | null): void => {
    const owner = role === 'proxy' ? proxyUrlRef : alphaUrlRef
    if (owner.current !== null) URL.revokeObjectURL(owner.current)
    owner.current = blob === null ? null : URL.createObjectURL(blob)
    if (role === 'proxy') setProxyUrl(owner.current)
    else setAlphaUrl(owner.current)
  }

  useEffect(() => {
    if (project.stages.ingest?.status !== 'succeeded') return
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout> | null = null
    let attempt = 0
    const load = async (): Promise<void> => {
      try {
        const descriptor = await backend.getSubjectMedia('proxy')
        const blob = await backend.fetchSubjectMediaArtifact(
          'proxy', descriptor.artifact_id, controller.signal,
        )
        if (controller.signal.aborted) return
        setProxy(descriptor)
        replaceUrl('proxy', blob)
      } catch (error) {
        if (controller.signal.aborted) return
        const delay = SUBJECT_MEDIA_RETRY_DELAYS_MS[attempt]
        if (
          error instanceof BackendClientError
          && error.code === 'subject_media_not_ready'
          && delay !== undefined
        ) {
          attempt += 1
          timer = setTimeout(() => void load(), delay)
          return
        }
        onErrorRef.current(error instanceof Error ? error : '代表帧尚未生成，请重试导入阶段。')
      }
    }
    void load()
    return () => {
      controller.abort()
      if (timer !== null) clearTimeout(timer)
    }
  }, [backend, project.stages.ingest?.cache_key, project.stages.ingest?.status])

  useEffect(() => {
    if (project.stages.segment?.status !== 'succeeded' || project.workflow.subject_prompt === null) return
    const controller = new AbortController()
    void backend.getSubjectMedia('alpha').then(async (descriptor) => {
      const blob = await backend.fetchSubjectMediaArtifact('alpha', descriptor.artifact_id, controller.signal)
      if (!controller.signal.aborted) replaceUrl('alpha', blob)
    }).catch((error: unknown) => {
      if (!controller.signal.aborted) onErrorRef.current(error instanceof Error ? error : 'Alpha 预览载入失败。')
    })
    return () => controller.abort()
  }, [backend, project.stages.segment?.status, project.workflow.subject_prompt])

  useEffect(() => () => {
    if (proxyUrlRef.current !== null) URL.revokeObjectURL(proxyUrlRef.current)
    if (alphaUrlRef.current !== null) URL.revokeObjectURL(alphaUrlRef.current)
    proxyUrlRef.current = null
    alphaUrlRef.current = null
  }, [])

  const submit = async (point = { x: Number(x), y: Number(y) }): Promise<void> => {
    if (busy) return
    if (proxy === null || !Number.isInteger(point.x) || !Number.isInteger(point.y)
      || point.x < 0 || point.y < 0 || point.x >= proxy.width || point.y >= proxy.height) {
      onError('人物坐标必须位于代表帧图像内。')
      return
    }
    setSubmitting(true)
    try {
      const next = await backend.updateProject({
        subject_prompt: { frame_index: proxy.frame_index, x: point.x, y: point.y },
      })
      onProjectChange(next)
      await onStartStage('segment')
    } catch (error) {
      onError(error instanceof Error ? error : '人物选择失败。')
    } finally {
      setSubmitting(false)
    }
  }

  const pick = (event: PointerEvent<HTMLDivElement>): void => {
    if (proxy === null || frameRef.current === null) return
    const point = toImagePoint(event, frameRef.current.getBoundingClientRect(), proxy)
    if (point === null) {
      onError('点击位于代表帧内容之外。')
      return
    }
    setX(String(point.x)); setY(String(point.y))
    void submit(point)
  }

  const reselect = async (): Promise<void> => {
    try {
      onProjectChange(await backend.updateProject({ subject_prompt: null }))
      replaceUrl('alpha', null)
      setX(''); setY('')
    } catch (error) {
      onError(error instanceof Error ? error : '无法重新选择人物。')
    }
  }

  return (
    <section aria-labelledby="subject-title" className="page-grid">
      <div className="page-heading">
        <p className="eyebrow">02 · SUBJECT</p>
        <h2 id="subject-title">选择前景人物</h2>
        <p>在代表帧的人物身体内点击。服务会验证坐标后再写入项目并启动分割。</p>
      </div>
      <div className="subject-layout">
        <div aria-label="人物代表帧" className="subject-frame preview-surface" onClick={pick} ref={frameRef} tabIndex={0}>
          {proxyUrl === null ? <div className="viewport-empty">载入代表帧…</div> : <img alt="人物代表帧" src={proxyUrl} />}
          {alphaUrl === null ? null : <img alt="人物 Alpha 叠加" className="alpha-overlay" src={alphaUrl} />}
        </div>
        <aside className="control-card">
          <h3>精确坐标</h3>
          <p>键盘用户可输入代表帧像素坐标。</p>
          <label>人物 X 坐标<input aria-label="人物 X 坐标" inputMode="numeric" onChange={(event) => setX(event.currentTarget.value)} value={x} /></label>
          <label>人物 Y 坐标<input aria-label="人物 Y 坐标" inputMode="numeric" onChange={(event) => setY(event.currentTarget.value)} value={y} /></label>
          <button disabled={busy || proxy === null || submitting} onClick={() => void submit()} type="button">{submitting ? '处理中…' : '确认人物位置'}</button>
          {project.workflow.subject_prompt !== null ? <button className="button-secondary" onClick={() => void reselect()} type="button">重新选择人物</button> : null}
        </aside>
      </div>
    </section>
  )
}
