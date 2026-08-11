import { type PointerEvent, useEffect, useRef, useState } from 'react'

import type { BackendClient } from '../../api/backend-client'
import { BackendClientError } from '../../api/http-backend-client'
import type { ProjectDto, SubjectMediaDto } from '../../api/types'
import {
  ImagePointFields,
  ImagePointMarker,
  useImagePointDraft,
} from '../coordinates/image-point-editor'
import { toImagePoint } from '../coordinates/image-point'

const SUBJECT_MEDIA_RETRY_DELAYS_MS = [250, 500, 1_000] as const

function shouldRetrySubjectMedia(error: unknown): boolean {
  return error instanceof BackendClientError
    && (error.code === 'subject_media_not_ready' || error.retryable)
}

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
  const [proxyAuthority, setProxyAuthority] = useState<string | null>(null)
  const [alphaUrl, setAlphaUrl] = useState<string | null>(null)
  const [alphaAuthority, setAlphaAuthority] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const frameRef = useRef<HTMLDivElement>(null)
  const proxyImageRef = useRef<HTMLImageElement>(null)
  const proxyUrlRef = useRef<string | null>(null)
  const alphaUrlRef = useRef<string | null>(null)
  const onErrorRef = useRef(onError)
  const submissionId = useRef(0)

  useEffect(() => { onErrorRef.current = onError }, [onError])

  const subjectAuthority = `${project.project_id}:${project.stages.ingest?.cache_key ?? 'no-ingest'}`
  const currentProxy = proxyAuthority === subjectAuthority && proxyUrl !== null ? proxy : null
  const currentProxyUrl = currentProxy === null ? null : proxyUrl
  const currentAlphaUrl = alphaAuthority === subjectAuthority ? alphaUrl : null
  const savedPrompt = project.workflow.subject_prompt
  const savedPoint = currentProxy !== null && savedPrompt?.frame_index === currentProxy.frame_index
    ? { x: savedPrompt.x, y: savedPrompt.y }
    : null
  const draft = useImagePointDraft(
    currentProxy,
    currentProxy === null ? `${subjectAuthority}:no-proxy` : `${subjectAuthority}:${currentProxy.artifact_id}:${currentProxy.frame_index}`,
    savedPoint,
  )
  const submissionAuthority = currentProxy === null
    ? `${subjectAuthority}:no-proxy`
    : `${subjectAuthority}:${currentProxy.artifact_id}:${currentProxy.frame_index}`
  const latestSubmissionAuthority = useRef(submissionAuthority)
  latestSubmissionAuthority.current = submissionAuthority

  useEffect(() => {
    submissionId.current += 1
    setSubmitting(false)
  }, [subjectAuthority])

  const replaceUrl = (role: 'proxy' | 'alpha', blob: Blob | null): void => {
    const owner = role === 'proxy' ? proxyUrlRef : alphaUrlRef
    if (owner.current !== null) URL.revokeObjectURL(owner.current)
    owner.current = blob === null ? null : URL.createObjectURL(blob)
    if (role === 'proxy') setProxyUrl(owner.current)
    else setAlphaUrl(owner.current)
  }

  useEffect(() => {
    setProxy(null)
    setProxyAuthority(null)
    replaceUrl('proxy', null)
    setAlphaAuthority(null)
    replaceUrl('alpha', null)
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
        setProxyAuthority(subjectAuthority)
      } catch (error) {
        if (controller.signal.aborted) return
        const delay = SUBJECT_MEDIA_RETRY_DELAYS_MS[attempt]
        if (
          shouldRetrySubjectMedia(error)
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
  }, [backend, project.project_id, project.stages.ingest?.cache_key, project.stages.ingest?.status])

  useEffect(() => {
    if (project.stages.segment?.status !== 'succeeded' || project.workflow.subject_prompt === null) return
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout> | null = null
    let attempt = 0
    const load = async (): Promise<void> => {
      try {
        const descriptor = await backend.getSubjectMedia('alpha')
        const blob = await backend.fetchSubjectMediaArtifact(
          'alpha', descriptor.artifact_id, controller.signal,
        )
        if (!controller.signal.aborted) {
          replaceUrl('alpha', blob)
          setAlphaAuthority(subjectAuthority)
        }
      } catch (error) {
        if (controller.signal.aborted) return
        const delay = SUBJECT_MEDIA_RETRY_DELAYS_MS[attempt]
        if (shouldRetrySubjectMedia(error) && delay !== undefined) {
          attempt += 1
          timer = setTimeout(() => void load(), delay)
          return
        }
        onErrorRef.current(error instanceof Error ? error : 'Alpha 预览载入失败。')
      }
    }
    void load()
    return () => {
      controller.abort()
      if (timer !== null) clearTimeout(timer)
    }
  }, [
    backend,
    project.project_id,
    project.stages.ingest?.cache_key,
    project.stages.segment?.cache_key,
    project.stages.segment?.status,
    project.workflow.subject_prompt?.frame_index,
    project.workflow.subject_prompt?.x,
    project.workflow.subject_prompt?.y,
  ])

  useEffect(() => () => {
    if (proxyUrlRef.current !== null) URL.revokeObjectURL(proxyUrlRef.current)
    if (alphaUrlRef.current !== null) URL.revokeObjectURL(alphaUrlRef.current)
    proxyUrlRef.current = null
    alphaUrlRef.current = null
  }, [])

  const submit = async (): Promise<void> => {
    if (busy) return
    const point = draft.point
    if (currentProxy === null || point === null) {
      onError('人物坐标必须位于代表帧图像内。')
      return
    }
    const requestId = ++submissionId.current
    const requestAuthority = submissionAuthority
    setSubmitting(true)
    try {
      const next = await backend.updateProject({
        expected_project_id: project.project_id,
        expected_ingest_cache_key: project.stages.ingest?.cache_key ?? null,
        subject_prompt: { frame_index: currentProxy.frame_index, x: point.x, y: point.y },
      })
      if (requestId !== submissionId.current
        || latestSubmissionAuthority.current !== requestAuthority) return
      draft.accept(point)
      onProjectChange(next)
      await onStartStage('segment')
    } catch (error) {
      if (requestId === submissionId.current
        && latestSubmissionAuthority.current === requestAuthority) {
        onError(error instanceof Error ? error : '人物选择失败。')
      }
    } finally {
      if (requestId === submissionId.current) setSubmitting(false)
    }
  }

  const pick = (event: PointerEvent<HTMLDivElement>): void => {
    if (busy || submitting || currentProxy === null || proxyImageRef.current === null) return
    const point = toImagePoint(event, proxyImageRef.current.getBoundingClientRect(), currentProxy)
    if (point === null) {
      onError('点击位于代表帧内容之外。')
      return
    }
    draft.select(point)
  }

  const reselect = async (): Promise<void> => {
    const requestId = ++submissionId.current
    const requestAuthority = submissionAuthority
    setSubmitting(true)
    try {
      const next = await backend.updateProject({
        expected_project_id: project.project_id,
        expected_ingest_cache_key: project.stages.ingest?.cache_key ?? null,
        subject_prompt: null,
      })
      if (requestId !== submissionId.current
        || latestSubmissionAuthority.current !== requestAuthority) return
      onProjectChange(next)
      replaceUrl('alpha', null)
      setAlphaAuthority(null)
      draft.accept(null)
    } catch (error) {
      if (requestId === submissionId.current
        && latestSubmissionAuthority.current === requestAuthority) {
        onError(error instanceof Error ? error : '无法重新选择人物。')
      }
    } finally {
      if (requestId === submissionId.current) setSubmitting(false)
    }
  }

  return (
    <section aria-labelledby="subject-title" className="page-grid">
      <div className="page-heading">
        <p className="eyebrow">02 · SUBJECT</p>
        <h2 id="subject-title">选择前景人物</h2>
        <p>在代表帧的人物身体内点击或输入像素坐标；确认后才会启动分割。</p>
      </div>
      <div className="subject-layout">
        <div aria-label="人物代表帧" className="subject-frame preview-surface" onClick={pick} ref={frameRef} tabIndex={0}>
          {currentProxyUrl === null ? <div className="viewport-empty">载入代表帧…</div> : <img alt="人物代表帧" ref={proxyImageRef} src={currentProxyUrl} />}
          {currentAlphaUrl === null ? null : <img alt="人物 Alpha 叠加" className="alpha-overlay" src={currentAlphaUrl} />}
          <ImagePointMarker containerRef={frameRef} image={currentProxy} mediaRef={proxyImageRef} pending={draft.dirty} point={draft.point} />
        </div>
        <aside className="control-card">
          <h3>精确坐标</h3>
          <p>点击和键盘输入只更新候选点，不会自动开始计算。</p>
          <ImagePointFields
            confirmedLabel="人物坐标已保存"
            disabled={busy || submitting}
            draft={draft}
            emptyLabel="尚未选择人物坐标"
            image={currentProxy}
            pendingLabel="候选人物坐标待确认"
            xLabel="人物 X 坐标"
            yLabel="人物 Y 坐标"
          />
          <button disabled={busy || currentProxy === null || draft.point === null || submitting} onClick={() => void submit()} type="button">{submitting ? '处理中…' : '确认人物位置并开始分割'}</button>
          {project.workflow.subject_prompt !== null ? <button className="button-secondary" onClick={() => void reselect()} type="button">重新选择人物</button> : null}
        </aside>
      </div>
    </section>
  )
}
