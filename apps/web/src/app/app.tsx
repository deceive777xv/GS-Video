import {
  type ReactNode,
  useCallback,
  useEffect,
  useRef,
  useState,
} from 'react'

import type { BackendClient } from '../api/backend-client'
import { BackendClientError } from '../api/http-backend-client'
import {
  createTaskStore,
  type TaskEventSource,
  type TaskStore,
  useTaskStore,
} from '../api/task-events'
import type { BootstrapDto, ProjectDto, StageName, TaskDto } from '../api/types'
import type { PlatformBridge } from '../platform/platform-bridge'
import { CameraPage } from '../features/camera/camera-page'
import { ExportPage } from '../features/export/export-page'
import { ImportPage } from '../features/import/import-page'
import { PreviewPage } from '../features/preview/preview-page'
import { SubjectPage } from '../features/subject/subject-page'
import {
  canAdvance,
  canVisitStep,
  creativeInteractionCount,
  stageSucceeded,
  type WorkflowStep,
  WORKFLOW_STEPS,
  workflowStepForProject,
} from './project-store'
import './app.css'

const STEP_LABELS: Record<WorkflowStep, { number: string; title: string; detail: string }> = {
  import: { number: '01', title: '导入', detail: '视频 + PLY' },
  subject: { number: '02', title: '人物', detail: '一次提示' },
  camera: { number: '03', title: '机位', detail: '镜头 + 落脚点' },
  preview: { number: '04', title: '预览', detail: '运动与合成' },
  export: { number: '05', title: '导出', detail: '验证 MP4' },
}

export interface AppProps {
  backend: BackendClient
  platform: PlatformBridge
  eventSource?: TaskEventSource
  initialBootstrap?: BootstrapDto
  createOwnedTaskStore?: (backend: BackendClient) => TaskStore
}

interface UiError {
  message: string
  code: string | null
  category: string | null
  retryable: boolean
}

function uiError(error: unknown): UiError {
  if (typeof error === 'string') {
    return { message: error, code: null, category: null, retryable: true }
  }
  if (error instanceof BackendClientError) {
    return {
      message: error.message,
      code: error.code,
      category: error.category,
      retryable: error.retryable,
    }
  }
  return {
    message: error instanceof Error ? error.message : '本地工作流发生未知错误。',
    code: null,
    category: null,
    retryable: true,
  }
}

function isTaskNotFound(error: unknown): boolean {
  return error instanceof BackendClientError
    ? error.status === 404 || error.code === 'task_not_found'
    : typeof error === 'object'
      && error !== null
      && (('status' in error && error.status === 404)
        || ('code' in error && error.code === 'task_not_found'))
}

function AppShell({ children }: { children: ReactNode }) {
  return <div className="app-shell">{children}</div>
}

export function App({
  backend,
  platform,
  eventSource,
  initialBootstrap,
  createOwnedTaskStore = createTaskStore,
}: AppProps) {
  const [taskStore] = useState(() => createOwnedTaskStore(backend))
  const taskState = useTaskStore(taskStore)
  const [bootstrap, setBootstrap] = useState<BootstrapDto | null>(initialBootstrap ?? null)
  const [project, setProject] = useState<ProjectDto | null>(initialBootstrap?.project ?? null)
  const [step, setStep] = useState<WorkflowStep>(() => initialBootstrap === undefined
    ? 'import'
    : workflowStepForProject(initialBootstrap.project))
  const [error, setError] = useState<UiError | null>(null)
  const [loading, setLoading] = useState(initialBootstrap === undefined)
  const [startingStage, setStartingStage] = useState(false)
  const [missingTaskOwnerId, setMissingTaskOwnerId] = useState<string | null>(null)
  const errorRef = useRef<HTMLDivElement>(null)
  const disposalCycle = useRef(0)
  const recoveredTaskId = useRef<string | null>(null)
  const autoSolveKey = useRef<string | null>(null)
  const projectAuthority = useRef(0)
  const stageAdmission = useRef(false)

  const reportError = useCallback((value: unknown): void => setError(uiError(value)), [])
  const reportUnknownError = useCallback((value: unknown): void => setError(uiError(value)), [])
  const taskRequestStillAuthoritative = useCallback((taskId: string): boolean => {
    const snapshot = taskStore.snapshot()
    const latest = snapshot.latestEvent
    if (latest === null) return true
    if (latest.type === 'task_event') {
      return latest.task_id === taskId
        || (snapshot.task?.id === taskId && snapshot.task.revision >= latest.revision)
    }
    const owner = latest.taskId ?? latest.task_id
    return owner === undefined
      ? snapshot.pendingResyncRevision === null && snapshot.task?.id === taskId
      : owner === taskId
        || (snapshot.task?.id === taskId && snapshot.task.revision >= latest.revision)
  }, [taskStore])
  const acceptProject = useCallback((next: ProjectDto): void => {
    projectAuthority.current += 1
    setProject(next)
  }, [])

  useEffect(() => {
    const cycle = ++disposalCycle.current
    return () => {
      queueMicrotask(() => {
        if (disposalCycle.current === cycle) taskStore.dispose()
      })
    }
  }, [taskStore])

  useEffect(() => {
    if (eventSource === undefined) return
    return eventSource.subscribe({
      afterRevision: taskStore.snapshot().revision,
      getResumeRevision: () => taskStore.snapshot().revision,
      onEvent: taskStore.onEvent,
      onConnectionChange: taskStore.onConnectionChange,
    })
  }, [eventSource, taskStore])

  useEffect(() => {
    if (initialBootstrap !== undefined) return
    const controller = new AbortController()
    setLoading(true)
    void backend.bootstrap(controller.signal).then((next) => {
      if (controller.signal.aborted) return
      setBootstrap(next)
      acceptProject(next.project)
      setStep(workflowStepForProject(next.project))
    }).catch((value: unknown) => {
      if (!controller.signal.aborted) reportUnknownError(value)
    }).finally(() => {
      if (!controller.signal.aborted) setLoading(false)
    })
    return () => controller.abort()
  }, [acceptProject, backend, initialBootstrap, reportUnknownError])

  useEffect(() => {
    const taskId = project?.workflow.active_task_id
    if (taskId === null || taskId === undefined || recoveredTaskId.current === taskId) return
    let stopped = false
    let timer: ReturnType<typeof setTimeout> | null = null
    const retryDelays = [250, 500, 1_000]
    let attempt = 0
    const recover = async (): Promise<void> => {
      try {
        const task = await backend.getTask(taskId)
        if (stopped
          || project?.workflow.active_task_id !== taskId
          || !taskRequestStillAuthoritative(taskId)) return
        taskStore.replaceFromRest(task)
        recoveredTaskId.current = taskId
        setMissingTaskOwnerId(null)
      } catch (value) {
        if (stopped) return
        if (isTaskNotFound(value)) {
          if (!taskRequestStillAuthoritative(taskId)) return
          recoveredTaskId.current = taskId
          setMissingTaskOwnerId(taskId)
          return
        }
        const delay = retryDelays[attempt]
        attempt += 1
        if (delay === undefined) {
          reportUnknownError(value)
          return
        }
        timer = setTimeout(() => void recover(), delay)
      }
    }
    void recover()
    return () => {
      stopped = true
      if (timer !== null) clearTimeout(timer)
    }
  }, [backend, project?.workflow.active_task_id, reportUnknownError, taskRequestStillAuthoritative, taskStore])

  useEffect(() => {
    const task = taskState.task
    if (task === null || !['queued', 'running'].includes(task.status)
      || taskState.connection === 'connected') return
    let stopped = false
    let timer: ReturnType<typeof setTimeout> | null = null
    const poll = async (): Promise<void> => {
      const authority = ++projectAuthority.current
      try {
        const [nextTask, nextProject] = await Promise.all([
          backend.getTask(task.id),
          backend.getProject(),
        ])
        if (stopped) return
        if (taskRequestStillAuthoritative(task.id)) taskStore.replaceFromRest(nextTask)
        if (projectAuthority.current === authority) setProject(nextProject)
        if (['queued', 'running'].includes(nextTask.status)) {
          timer = setTimeout(() => void poll(), 1_000)
        }
      } catch (value) {
        if (!stopped) {
          reportUnknownError(value)
          timer = setTimeout(() => void poll(), 1_500)
        }
      }
    }
    timer = setTimeout(() => void poll(), 250)
    return () => {
      stopped = true
      if (timer !== null) clearTimeout(timer)
    }
  }, [backend, reportUnknownError, taskRequestStillAuthoritative, taskState.connection, taskState.task?.id, taskState.task?.status, taskStore])

  const refreshProject = useCallback(async (): Promise<ProjectDto> => {
    const authority = ++projectAuthority.current
    const next = await backend.getProject()
    if (projectAuthority.current === authority) setProject(next)
    return next
  }, [backend])

  const refreshBootstrap = useCallback(async (): Promise<void> => {
    const next = await backend.bootstrap()
    setBootstrap(next)
    acceptProject(next.project)
    setStep(workflowStepForProject(next.project))
  }, [acceptProject, backend])

  useEffect(() => {
    const latest = taskState.latestEvent
    if (latest === null) return
    const explicitTaskId = latest.type === 'task_event'
      ? latest.task_id
      : latest.taskId ?? latest.task_id
    const controller = new AbortController()
    const authority = ++projectAuthority.current
    void (async () => {
      const nextProject = await backend.getProject()
      if (controller.signal.aborted || projectAuthority.current !== authority) return
      setProject(nextProject)
      const taskId = explicitTaskId ?? nextProject.workflow.active_task_id
      let task: TaskDto | null = null
      if (taskId !== null) {
        try {
          task = await backend.getTask(taskId)
        } catch (value) {
          if (latest.type === 'resync_required' && isTaskNotFound(value)) {
            if (controller.signal.aborted || projectAuthority.current !== authority) return
            setMissingTaskOwnerId(taskId)
            taskStore.acknowledgeResync(latest.revision, null)
            return
          }
          throw value
        }
      }
      if (controller.signal.aborted || projectAuthority.current !== authority) return
      if (latest.type === 'resync_required') {
        taskStore.acknowledgeResync(
          latest.revision,
          task !== null && task.id === taskId ? task : null,
        )
      } else if (task !== null && task.id === taskId) {
        taskStore.replaceFromRest(task)
      }
    })().catch((value: unknown) => {
      if (!controller.signal.aborted) reportUnknownError(value)
    })
    return () => controller.abort()
  }, [backend, reportUnknownError, taskState.latestEvent, taskStore])

  const runStage = useCallback(async (target: StageName): Promise<TaskDto> => {
    setError(null)
    const owner = taskStore.snapshot().task
    const projectOwnerId = project?.workflow.active_task_id
    const unresolvedProjectOwner = projectOwnerId !== null
      && projectOwnerId !== undefined
      && missingTaskOwnerId !== projectOwnerId
      && (owner === null || owner.id !== projectOwnerId)
    if (stageAdmission.current
      || unresolvedProjectOwner
      || (owner !== null && ['queued', 'running'].includes(owner.status))) {
      const value = new Error('已有阶段任务正在运行，请等待完成或先取消。')
      reportUnknownError(value)
      throw value
    }
    stageAdmission.current = true
    setStartingStage(true)
    try {
      const task = await backend.startTask(target)
      setMissingTaskOwnerId(null)
      taskStore.replaceFromRest(task)
      await refreshProject()
      return task
    } catch (value) {
      reportUnknownError(value)
      throw value
    } finally {
      stageAdmission.current = false
      setStartingStage(false)
    }
  }, [backend, missingTaskOwnerId, project?.workflow.active_task_id, refreshProject, reportUnknownError, taskStore])

  useEffect(() => {
    if (project === null) return
    const task = taskState.task
    const prompt = project.workflow.subject_prompt
    const segment = project.stages.segment
    const key = prompt === null || segment?.cache_key === null || segment?.cache_key === undefined
      ? null
      : `${segment.cache_key}:${prompt.frame_index}:${prompt.x}:${prompt.y}`
    const segmentTerminal = stageSucceeded(project, 'segment')
      && (task === null || task.target_stage !== 'segment' || task.status === 'succeeded')
    const noActiveOwner = task === null || !['queued', 'running'].includes(task.status)
    if (key !== null && segmentTerminal && !stageSucceeded(project, 'solve_camera')
      && noActiveOwner && autoSolveKey.current !== key) {
      autoSolveKey.current = key
      void runStage('solve_camera').catch(() => {
        if (autoSolveKey.current === key) autoSolveKey.current = null
      })
    }
  }, [project, runStage, taskState.task])

  useEffect(() => {
    if (error !== null) errorRef.current?.focus()
  }, [error])

  const cancelActiveTask = async (): Promise<void> => {
    const task = taskState.task
    if (task === null || !['queued', 'running'].includes(task.status)) return
    try {
      taskStore.replaceFromRest(await backend.cancelTask(task.id))
      await refreshProject()
    } catch (value) { reportUnknownError(value) }
  }

  if (loading || bootstrap === null || project === null) {
    return (
      <AppShell>
        <main className="loading-screen" aria-live="polite">
          <div className="loading-mark"><span /></div>
          <p>正在恢复本地项目权威状态…</p>
        </main>
      </AppShell>
    )
  }

  const interactionCount = creativeInteractionCount(project)
  const currentIndex = WORKFLOW_STEPS.indexOf(step)
  const previous = WORKFLOW_STEPS[currentIndex - 1]
  const next = WORKFLOW_STEPS[currentIndex + 1]
  const activeTask = taskState.task
  const activeTaskRunning = activeTask !== null && ['queued', 'running'].includes(activeTask.status)
  const unresolvedProjectOwner = project.workflow.active_task_id !== null
    && missingTaskOwnerId !== project.workflow.active_task_id
    && (activeTask === null || activeTask.id !== project.workflow.active_task_id)
  const workflowBusy = startingStage
    || activeTaskRunning
    || unresolvedProjectOwner
    || !bootstrap.environment.ready
  const progressEvent = taskState.latestEvent?.type === 'task_event'
    && taskState.latestEvent.task_id === activeTask?.id
    && taskState.latestEvent.revision >= (activeTask?.revision ?? 0)
    ? taskState.latestEvent
    : null
  const restProgress = activeTask !== null
    && typeof activeTask.progress === 'number'
    && activeTask.current !== undefined
    && activeTask.total !== undefined
    && activeTask.message !== undefined
    && typeof activeTask.elapsed_seconds === 'number'
    && activeTask.eta_seconds !== undefined
    ? {
        stage: activeTask.target_stage,
        revision: activeTask.revision,
        progress: activeTask.progress,
        current: activeTask.current,
        total: activeTask.total,
        message: activeTask.message,
        elapsed_seconds: activeTask.elapsed_seconds,
        eta_seconds: activeTask.eta_seconds,
      }
    : null
  const authoritativeProgress = progressEvent !== null
    && (restProgress === null || progressEvent.revision >= restProgress.revision)
    ? progressEvent
    : restProgress
  const progressPercent = authoritativeProgress === null
    || !Number.isFinite(authoritativeProgress.progress)
    ? null
    : Math.round(Math.min(1, Math.max(0, authoritativeProgress.progress)) * 100)
  const retryableTask = activeTask?.status === 'failed'
    && progressEvent?.error?.retryable === true

  let page: ReactNode
  switch (step) {
    case 'import':
      page = <ImportPage backend={backend} busy={workflowBusy} environment={bootstrap.environment} onEnvironmentRefresh={refreshBootstrap} onError={reportError} onProjectChange={acceptProject} onStartStage={runStage} platform={platform} project={project} />
      break
    case 'subject':
      page = <SubjectPage backend={backend} busy={workflowBusy} onError={reportError} onProjectChange={acceptProject} onStartStage={runStage} project={project} />
      break
    case 'camera':
      page = <CameraPage backend={backend} onError={reportError} onProjectChange={acceptProject} onRefresh={refreshProject} project={project} />
      break
    case 'preview':
      page = <PreviewPage
        activeTask={activeTask}
        backend={backend}
        busy={workflowBusy}
        latestEvent={taskState.latestEvent}
        onBackToCamera={() => setStep('camera')}
        onError={reportError}
        onProjectChange={acceptProject}
        onReselectSubject={() => setStep('subject')}
        onStartStage={runStage}
        project={project}
      />
      break
    case 'export':
      page = <ExportPage activeTask={activeTask} backend={backend} busy={workflowBusy} onError={reportError} onProjectChange={acceptProject} onStartStage={runStage} platform={platform} project={project} />
      break
  }

  return (
    <AppShell>
      <header className="topbar">
        <a className="brand" href="#workflow-main" aria-label="GS Video 工作流首页">
          <span className="brand-mark" aria-hidden="true"><i /><i /><i /></span>
          <span><strong>GS VIDEO</strong><small>GAUSSIAN COMPOSITOR</small></span>
        </a>
        <div className="project-heading">
          <span>当前项目</span><strong>{project.name}</strong>
        </div>
        <div className="header-status">
          <span className={`connection-dot connection-${taskState.connection}`} />
          <span>{taskState.connection === 'connected' ? '实时事件已连接' : 'REST 权威恢复可用'}</span>
          <strong>8 GB 模式</strong>
        </div>
      </header>

      <div className="workspace">
        <nav aria-label="制作流程" className="workflow-rail">
          <div className="rail-intro"><span>WORKFLOW</span><strong>背景替换</strong></div>
          <ol>
            {WORKFLOW_STEPS.map((item) => {
              const meta = STEP_LABELS[item]
              const reachable = canVisitStep(project, item)
              const completed = WORKFLOW_STEPS.indexOf(item) < WORKFLOW_STEPS.indexOf(workflowStepForProject(project))
              return (
                <li className={`${item === step ? 'is-current' : ''} ${completed ? 'is-complete' : ''}`} key={item}>
                  <button disabled={!reachable} onClick={() => setStep(item)} type="button">
                    <span className="step-number">{completed ? '✓' : meta.number}</span>
                    <span><strong>{meta.title}</strong><small>{meta.detail}</small></span>
                  </button>
                </li>
              )
            })}
          </ol>
          <div aria-label={`创作交互 ${interactionCount} / 3`} className="interaction-meter">
            <div><span>创作交互</span><strong>{interactionCount} / 3</strong></div>
            <div className="meter-track"><span style={{ width: `${interactionCount / 3 * 100}%` }} /></div>
            <p>只统计后端已持久化的人物、机位与落脚点。</p>
          </div>
        </nav>

        <main className="workflow-main" id="workflow-main">
          {error !== null ? (
            <div aria-atomic="true" className="error-banner" ref={errorRef} role="alert" tabIndex={-1}>
              <div><strong>{error.category === 'repairable' ? '可以修复' : '操作未完成'}</strong><p>{error.message}</p>{error.code !== null ? <code>{error.code}</code> : null}</div>
              <button aria-label="关闭错误提示" onClick={() => setError(null)} type="button">×</button>
            </div>
          ) : null}
          {page}
        </main>
      </div>

      <footer className="workflow-footer">
        <div aria-live="polite" className="task-progress">
          <span className={`task-indicator ${activeTaskRunning ? 'is-running' : ''}`} />
          {activeTask === null ? (
            <span className="task-copy"><strong>准备就绪</strong><small>阶段状态保存在项目中</small></span>
          ) : (
            <span className="task-copy">
              <strong>{activeTask.target_stage} · {activeTask.status}</strong>
              <small className="task-message">
                {authoritativeProgress?.message
                  ?? (authoritativeProgress === null ? `revision ${activeTask.revision}` : '')}
              </small>
              <small className="task-timing">
                {authoritativeProgress?.current !== null
                  && authoritativeProgress?.current !== undefined
                  && authoritativeProgress.total !== null
                  && authoritativeProgress.total !== undefined
                  ? <span>{authoritativeProgress.current} / {authoritativeProgress.total}</span>
                  : null}
                {authoritativeProgress?.elapsed_seconds !== undefined
                  ? <span>已用时 {authoritativeProgress.elapsed_seconds} 秒</span>
                  : null}
                {authoritativeProgress?.eta_seconds !== null
                  && authoritativeProgress?.eta_seconds !== undefined
                  ? <span>预计剩余 {authoritativeProgress.eta_seconds} 秒</span>
                  : null}
                <span>{taskState.connection}</span>
              </small>
            </span>
          )}
          {authoritativeProgress !== null && progressPercent !== null ? (
            <div
              aria-label={`${authoritativeProgress.stage} 进度`}
              aria-valuemax={100}
              aria-valuemin={0}
              aria-valuenow={progressPercent}
              className="task-progressbar"
              role="progressbar"
            >
              <span style={{ width: `${progressPercent}%` }} />
              <small>{progressPercent}%</small>
            </div>
          ) : null}
        </div>
        <div className="footer-actions">
          {activeTaskRunning ? <button className="button-danger" onClick={() => void cancelActiveTask()} type="button">取消任务</button> : null}
          {activeTask !== null && retryableTask ? <button className="button-secondary" onClick={() => void runStage(activeTask.target_stage)} type="button">重试阶段</button> : null}
          <button className="button-secondary" disabled={previous === undefined} onClick={() => previous !== undefined && setStep(previous)} type="button">上一步</button>
          <button disabled={next === undefined || !canAdvance(project, step)} onClick={() => next !== undefined && setStep(next)} type="button">下一步</button>
        </div>
      </footer>
    </AppShell>
  )
}
