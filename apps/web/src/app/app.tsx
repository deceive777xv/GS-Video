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
  const errorRef = useRef<HTMLDivElement>(null)
  const disposalCycle = useRef(0)
  const recoveredTaskId = useRef<string | null>(null)
  const autoSolveKey = useRef<string | null>(null)
  const projectAuthority = useRef(0)

  const reportError = useCallback((value: unknown): void => setError(uiError(value)), [])
  const reportUnknownError = useCallback((value: unknown): void => setError(uiError(value)), [])
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
    recoveredTaskId.current = taskId
    void backend.getTask(taskId).then(taskStore.replaceFromRest).catch(reportUnknownError)
  }, [backend, project?.workflow.active_task_id, reportUnknownError, taskStore])

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
        taskStore.replaceFromRest(nextTask)
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
  }, [backend, reportUnknownError, taskState.connection, taskState.task?.id, taskState.task?.status, taskStore])

  const refreshProject = useCallback(async (): Promise<ProjectDto> => {
    const authority = ++projectAuthority.current
    const next = await backend.getProject()
    if (projectAuthority.current === authority) setProject(next)
    return next
  }, [backend])

  useEffect(() => {
    const latest = taskState.latestEvent
    if (latest === null) return
    const taskId = latest.type === 'task_event'
      ? latest.task_id
      : taskState.task?.id ?? project?.workflow.active_task_id
    if (taskId === null || taskId === undefined) return
    const controller = new AbortController()
    const authority = ++projectAuthority.current
    void Promise.all([backend.getTask(taskId), backend.getProject()]).then(([task, nextProject]) => {
      if (controller.signal.aborted) return
      taskStore.replaceFromRest(task)
      if (projectAuthority.current === authority) setProject(nextProject)
    }).catch((value: unknown) => {
      if (!controller.signal.aborted) reportUnknownError(value)
    })
    return () => controller.abort()
  }, [backend, project?.workflow.active_task_id, reportUnknownError, taskState.latestEvent, taskStore])

  const runStage = useCallback(async (target: StageName): Promise<TaskDto> => {
    setError(null)
    try {
      const task = await backend.startTask(target)
      taskStore.replaceFromRest(task)
      await refreshProject()
      return task
    } catch (value) {
      reportUnknownError(value)
      throw value
    }
  }, [backend, refreshProject, reportUnknownError, taskStore])

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

  let page: ReactNode
  switch (step) {
    case 'import':
      page = <ImportPage backend={backend} environment={bootstrap.environment} onError={reportError} onProjectChange={acceptProject} onStartStage={runStage} platform={platform} project={project} />
      break
    case 'subject':
      page = <SubjectPage backend={backend} onError={reportError} onProjectChange={acceptProject} onStartStage={runStage} project={project} />
      break
    case 'camera':
      page = <CameraPage backend={backend} onError={reportError} onProjectChange={acceptProject} onRefresh={refreshProject} project={project} />
      break
    case 'preview':
      page = <PreviewPage
        activeTask={activeTask}
        backend={backend}
        onBackToCamera={() => setStep('camera')}
        onError={reportError}
        onProjectChange={acceptProject}
        onReselectSubject={() => setStep('subject')}
        onStartStage={runStage}
        project={project}
      />
      break
    case 'export':
      page = <ExportPage activeTask={activeTask} backend={backend} onError={reportError} onStartStage={runStage} platform={platform} project={project} />
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
            <span><strong>准备就绪</strong><small>阶段状态保存在项目中</small></span>
          ) : (
            <span><strong>{activeTask.target_stage} · {activeTask.status}</strong><small>revision {activeTask.revision} · {taskState.connection}</small></span>
          )}
        </div>
        <div className="footer-actions">
          {activeTaskRunning ? <button className="button-danger" onClick={() => void cancelActiveTask()} type="button">取消任务</button> : null}
          {activeTask !== null && ['failed', 'cancelled'].includes(activeTask.status) && activeTask.error !== null ? <button className="button-secondary" onClick={() => void runStage(activeTask.target_stage)} type="button">重试阶段</button> : null}
          <button className="button-secondary" disabled={previous === undefined} onClick={() => previous !== undefined && setStep(previous)} type="button">上一步</button>
          <button disabled={next === undefined || !canAdvance(project, step)} onClick={() => next !== undefined && setStep(next)} type="button">下一步</button>
        </div>
      </footer>
    </AppShell>
  )
}
