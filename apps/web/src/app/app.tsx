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
import type { BootstrapDto, ProjectDto, ProjectSummaryDto, StageName, StorageLayoutDto, TaskDto, TaskEvent, VramBudgetDto } from '../api/types'
import type { PlatformBridge } from '../platform/platform-bridge'
import { CameraPage } from '../features/camera/camera-page'
import { ExportPage } from '../features/export/export-page'
import { ImportPage } from '../features/import/import-page'
import { HomePage } from '../features/home/home-page'
import { AssetLibraryPage, type AssetSelectionContext } from '../features/assets/asset-library-page'
import { PreviewPage } from '../features/preview/preview-page'
import { SubjectPage } from '../features/subject/subject-page'
import { VramBudgetControl } from '../features/settings/vram-budget-control'
import { StorageSettingsPage } from '../features/settings/storage-settings-page'
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
  startAtHome?: boolean
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

function authoritativeSolveCameraError(
  task: TaskDto | null,
  event: TaskEvent | null,
): { key: string; error: UiError } | null {
  if (
    task?.status !== 'failed'
    || task.target_stage !== 'solve_camera'
    || event?.type !== 'task_event'
    || event.task_id !== task.id
    || event.stage !== 'solve_camera'
    || event.revision < task.revision
    || event.error?.code !== 'unsupported_material'
  ) return null
  return {
    key: `${event.task_id}:${event.revision}:unsupported_material`,
    error: {
      message: '相机运动求解失败：当前视频无法生成可信的相机轨迹。请更换镜头运动更连续、画面纹理更清晰的视频后重试。',
      code: 'unsupported_material',
      category: typeof event.error.category === 'string' ? event.error.category : 'subject',
      retryable: event.error.retryable === true,
    },
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

function formatTaskSeconds(value: number): string {
  return value.toFixed(3)
}

function AppShell({ children }: { children: ReactNode }) {
  return <div className="app-shell">{children}</div>
}

function HubHeader({ project }: { project: ProjectDto | null }) {
  return (
    <header className="topbar hub-topbar">
      <a className="brand" href="#/" aria-label="返回项目首页">
        <span className="brand-mark" aria-hidden="true"><i /><i /><i /></span>
        <span><strong>GS VIDEO</strong><small>GAUSSIAN COMPOSITOR</small></span>
      </a>
      <nav className="hub-nav" aria-label="主导航"><a href="#/">项目</a><a href="#/assets/video">素材库</a><a href="#/settings">设置</a></nav>
      <div className="project-heading"><span>当前项目</span><strong>{project?.name ?? '未选择'}</strong></div>
    </header>
  )
}

type AppView = 'home' | 'workflow' | 'assets-video' | 'assets-ply' | 'settings'

interface WorkflowRoute {
  projectId: string
  step: WorkflowStep
}

function workflowRouteFromHash(hash: string): WorkflowRoute | null {
  const match = /^#\/projects\/([^/]+)\/workflow\/(import|subject|camera|preview|export)$/.exec(hash)
  if (match === null) return null
  return {
    projectId: decodeURIComponent(match[1] ?? ''),
    step: match[2] as WorkflowStep,
  }
}

function routeFromHash(hash: string, fallback: AppView): AppView {
  if (hash === '#/' || hash === '#') return 'home'
  if (hash.startsWith('#/assets/ply')) return 'assets-ply'
  if (hash.startsWith('#/assets/video') || hash.startsWith('#/assets')) return 'assets-video'
  if (hash.startsWith('#/settings')) return 'settings'
  if (hash.startsWith('#/workflow') || workflowRouteFromHash(hash) !== null) return 'workflow'
  return fallback
}

function assetReturnProjectFromHash(hash: string): string | null {
  const match = /^#\/assets\/(?:video|ply)\?returnProject=([^&]+)$/.exec(hash)
  if (match === null) return null
  try {
    const value = decodeURIComponent(match[1] ?? '')
    return value.length > 0 ? value : null
  } catch {
    return null
  }
}

export function App({
  backend,
  platform,
  eventSource,
  initialBootstrap,
  createOwnedTaskStore = createTaskStore,
  startAtHome = false,
}: AppProps) {
  const [taskStore] = useState(() => {
    const store = createOwnedTaskStore(backend)
    if (initialBootstrap !== undefined) {
      store.reset?.(initialBootstrap.project?.workflow.active_task_id)
    }
    return store
  })
  const taskState = useTaskStore(taskStore)
  const [bootstrap, setBootstrap] = useState<BootstrapDto | null>(initialBootstrap ?? null)
  const [project, setProject] = useState<ProjectDto | null>(initialBootstrap?.project ?? null)
  const [step, setStep] = useState<WorkflowStep>(() => initialBootstrap?.project === null
    || initialBootstrap === undefined
    ? 'import'
    : workflowStepForProject(initialBootstrap.project))
  const [view, setView] = useState<AppView>(() => routeFromHash(
    startAtHome ? window.location.hash : '#/workflow',
    startAtHome ? 'home' : 'workflow',
  ))
  const [workflowRoute, setWorkflowRoute] = useState<WorkflowRoute | null>(() => (
    startAtHome ? workflowRouteFromHash(window.location.hash) : null
  ))
  const [error, setError] = useState<UiError | null>(null)
  const [dismissedTaskErrorKey, setDismissedTaskErrorKey] = useState<string | null>(null)
  const [loading, setLoading] = useState(initialBootstrap === undefined)
  const [startingStage, setStartingStage] = useState(false)
  const [missingTaskOwnerId, setMissingTaskOwnerId] = useState<string | null>(null)
  const errorRef = useRef<HTMLDivElement>(null)
  const disposalCycle = useRef(0)
  const recoveredTaskId = useRef<string | null>(null)
  const autoSolveKey = useRef<string | null>(null)
  const projectAuthority = useRef(0)
  const activeProjectId = useRef<string | null>(initialBootstrap?.project?.project_id ?? null)
  const stageAdmission = useRef(false)

  useEffect(() => {
    const syncRoute = (): void => {
      const nextWorkflowRoute = workflowRouteFromHash(window.location.hash)
      setWorkflowRoute(nextWorkflowRoute)
      if (nextWorkflowRoute !== null) setStep(nextWorkflowRoute.step)
      setView(routeFromHash(window.location.hash, 'home'))
    }
    window.addEventListener('hashchange', syncRoute)
    return () => window.removeEventListener('hashchange', syncRoute)
  }, [])

  const navigate = useCallback((next: AppView): void => {
    const hash = next === 'home' ? '#/'
      : next === 'workflow' ? '#/workflow'
        : next === 'assets-ply' ? '#/assets/ply' : '#/assets/video'
    if (window.location.hash === hash) setView(next)
    else window.location.hash = hash
  }, [])

  const navigateWorkflow = useCallback((projectId: string, nextStep: WorkflowStep): void => {
    const hash = `#/projects/${encodeURIComponent(projectId)}/workflow/${nextStep}`
    if (window.location.hash === hash) {
      setWorkflowRoute({ projectId, step: nextStep })
      setStep(nextStep)
      setView('workflow')
    } else {
      window.location.hash = hash
    }
  }, [])

  useEffect(() => {
    if (!loading && bootstrap !== null && project === null && view === 'workflow'
      && workflowRoute === null) {
      navigate('home')
    }
  }, [bootstrap, loading, navigate, project, view, workflowRoute])

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
    if (activeProjectId.current !== next.project_id) {
      taskStore.reset?.(next.workflow.active_task_id)
    }
    activeProjectId.current = next.project_id
    projectAuthority.current += 1
    setProject(next)
  }, [taskStore])
  const clearProject = useCallback((): void => {
    if (activeProjectId.current !== null) taskStore.reset?.(null)
    activeProjectId.current = null
    projectAuthority.current += 1
    setProject(null)
  }, [taskStore])

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
      if (next.project === null) clearProject()
      else {
        acceptProject(next.project)
        setStep(workflowStepForProject(next.project))
      }
    }).catch((value: unknown) => {
      if (!controller.signal.aborted) reportUnknownError(value)
    }).finally(() => {
      if (!controller.signal.aborted) setLoading(false)
    })
    return () => controller.abort()
  }, [acceptProject, backend, clearProject, initialBootstrap, reportUnknownError])

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
    if (next.project === null) clearProject()
    else {
      acceptProject(next.project)
      setStep(workflowStepForProject(next.project))
    }
  }, [acceptProject, backend, clearProject])

  useEffect(() => {
    if (loading || bootstrap === null || workflowRoute === null) return
    let stopped = false
    const openRoute = async (): Promise<void> => {
      try {
        if (project?.project_id === workflowRoute.projectId) {
          const reachableStep = canVisitStep(project, workflowRoute.step)
            ? workflowRoute.step
            : workflowStepForProject(project)
          if (reachableStep === workflowRoute.step) setStep(reachableStep)
          else navigateWorkflow(project.project_id, reachableStep)
          return
        }
        const next = await backend.activateProject(workflowRoute.projectId)
        if (stopped) return
        acceptProject(next)
        const reachableStep = canVisitStep(next, workflowRoute.step)
          ? workflowRoute.step
          : workflowStepForProject(next)
        const nextBootstrap = await backend.bootstrap()
        if (stopped) return
        setBootstrap(nextBootstrap)
        if (reachableStep === workflowRoute.step) setStep(reachableStep)
        else navigateWorkflow(next.project_id, reachableStep)
      } catch (value) {
        if (stopped) return
        reportUnknownError(value)
        navigate('home')
      }
    }
    void openRoute()
    return () => { stopped = true }
  }, [
    acceptProject,
    backend,
    bootstrap === null,
    loading,
    navigate,
    navigateWorkflow,
    project?.project_id,
    reportUnknownError,
    workflowRoute,
  ])

  useEffect(() => {
    if (loading || bootstrap === null || view !== 'home') return
    const controller = new AbortController()
    void backend.bootstrap(controller.signal).then((next) => {
      if (!controller.signal.aborted) setBootstrap(next)
    }).catch((value: unknown) => {
      if (!controller.signal.aborted) reportUnknownError(value)
    })
    return () => controller.abort()
  }, [backend, bootstrap === null, loading, reportUnknownError, view])

  const acceptVramBudget = useCallback((next: VramBudgetDto): void => {
    setBootstrap((current) => current === null ? null : {
      ...current,
      environment: {
        ...current.environment,
        vram_mb: next.total_vram_mb,
        vram_limit_mb: next.selected_vram_mb,
      },
      vram_budget: next,
    })
  }, [])

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
    const expectedProjectId = project?.project_id
    if (expectedProjectId === undefined) {
      const value = new Error('请先打开项目，再启动阶段任务。')
      reportUnknownError(value)
      throw value
    }
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
      const task = await backend.startTask(target, expectedProjectId)
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
  }, [backend, missingTaskOwnerId, project?.project_id, project?.workflow.active_task_id, refreshProject, reportUnknownError, taskStore])

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

  if (loading || bootstrap === null) {
    return (
      <AppShell>
        <main className="loading-screen" aria-live="polite">
          <div className="loading-mark"><span /></div>
          <p>正在恢复本地项目权威状态…</p>
        </main>
      </AppShell>
    )
  }

  const shellTask = taskState.task
  const shellBusy = startingStage
    || (shellTask !== null && ['queued', 'running'].includes(shellTask.status))
    || (project?.workflow.active_task_id !== null
      && project?.workflow.active_task_id !== undefined
      && missingTaskOwnerId !== project.workflow.active_task_id
      && (shellTask === null || shellTask.id !== project.workflow.active_task_id))

  const refreshCatalog = async (): Promise<BootstrapDto> => {
    const next = await backend.bootstrap()
    setBootstrap(next)
    if (next.project === null) clearProject()
    else acceptProject(next.project)
    return next
  }

  const acceptStorageLayout = (value: StorageLayoutDto): void => {
    setBootstrap((current) => current === null ? current : { ...current, storage_layout: value })
  }

  const createProject = async (name: string): Promise<void> => {
    try {
      const next = await backend.createProject(name.trim())
      acceptProject(next)
      setStep(workflowStepForProject(next))
      await refreshCatalog()
      navigateWorkflow(next.project_id, workflowStepForProject(next))
    } catch (value) {
      reportUnknownError(value)
      throw value
    }
  }

  const openProject = async (summary: ProjectSummaryDto): Promise<void> => {
    try {
      const next = project?.project_id === summary.project_id
        ? project
        : await backend.activateProject(summary.project_id)
      acceptProject(next)
      setStep(workflowStepForProject(next))
      navigateWorkflow(next.project_id, workflowStepForProject(next))
    } catch (value) {
      reportUnknownError(value)
      throw value
    }
  }

  const renameProject = async (summary: ProjectSummaryDto, name: string): Promise<void> => {
    try {
      await backend.renameProject(summary.project_id, name.trim())
      await refreshCatalog()
    } catch (value) {
      reportUnknownError(value)
      throw value
    }
  }

  const deleteProject = async (summary: ProjectSummaryDto): Promise<void> => {
    try {
      await backend.deleteProject(summary.project_id)
      await refreshCatalog()
    } catch (value) {
      reportUnknownError(value)
      throw value
    }
  }

  const acceptLibraryProject = async (
    next: ProjectDto,
    context: AssetSelectionContext,
  ): Promise<void> => {
    if (activeProjectId.current !== context.expectedProjectId
      || next.project_id !== context.expectedProjectId) return
    acceptProject(next)
    if (next.workflow.source_summary !== null
      && next.workflow.scene_summary !== null
      && !stageSucceeded(next, 'ingest')) {
      try {
        await runStage('ingest')
      } catch {
        // runStage already reports the authoritative task-admission error.
      }
    }
    void refreshCatalog().catch(reportUnknownError)
    if (context.returnProjectId === next.project_id) {
      navigateWorkflow(next.project_id, 'import')
    }
  }

  const requestedReturnProjectId = assetReturnProjectFromHash(window.location.hash)
  const returnProjectId = requestedReturnProjectId === project?.project_id
    ? requestedReturnProjectId
    : null

  if (view === 'home' || view === 'assets-video' || view === 'assets-ply' || view === 'settings') {
    return (
      <AppShell>
        <HubHeader project={project} />
        {error !== null ? (
          <div aria-atomic="true" className="error-banner hub-error" ref={errorRef} role="alert" tabIndex={-1}>
            <div><strong>操作未完成</strong><p>{error.message}</p>{error.code !== null ? <code>{error.code}</code> : null}</div>
            <button aria-label="关闭错误提示" onClick={() => setError(null)} type="button">×</button>
          </div>
        ) : null}
        {view === 'home' ? (
          <HomePage
            activeProjectId={project?.project_id ?? null}
            busy={shellBusy}
            onCreate={createProject}
            onDelete={deleteProject}
            onOpen={openProject}
            onRename={renameProject}
            projects={bootstrap.projects ?? []}
          />
        ) : view === 'settings' ? (
          bootstrap.storage_layout === null || bootstrap.storage_layout === undefined ? (
            <main className="hub-main"><div className="empty-state"><strong>存储设置不可用</strong><p>当前后端未提供机器级目录控制。</p></div></main>
          ) : (
            <StorageSettingsPage
              backend={backend}
              busy={shellBusy}
              initial={bootstrap.storage_layout}
              onChange={acceptStorageLayout}
              onError={reportUnknownError}
              platform={platform}
            />
          )
        ) : (
          <AssetLibraryPage
            backend={backend}
            busy={shellBusy}
            kind={view === 'assets-ply' ? 'ply' : 'video'}
            onError={reportUnknownError}
            onProjectChange={acceptLibraryProject}
            platform={platform}
            project={project}
            returnProjectId={returnProjectId}
          />
        )}
        <footer className="hub-footer">GS VIDEO · 本地项目与共享素材</footer>
      </AppShell>
    )
  }

  if (project === null) {
    return null
  }

  const interactionCount = creativeInteractionCount(project)
  const currentIndex = WORKFLOW_STEPS.indexOf(step)
  const previous = WORKFLOW_STEPS[currentIndex - 1]
  const next = WORKFLOW_STEPS[currentIndex + 1]
  const activeTask = taskState.task
  const taskFailure = authoritativeSolveCameraError(activeTask, taskState.latestEvent)
  const visibleTaskFailure = taskFailure?.key === dismissedTaskErrorKey ? null : taskFailure
  const visibleError = visibleTaskFailure?.error ?? error
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
        onBackToCamera={() => navigateWorkflow(project.project_id, 'camera')}
        onError={reportError}
        onProjectChange={acceptProject}
        onReselectSubject={() => navigateWorkflow(project.project_id, 'subject')}
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
        <a className="brand" href="#/" aria-label="返回项目首页">
          <span className="brand-mark" aria-hidden="true"><i /><i /><i /></span>
          <span><strong>GS VIDEO</strong><small>GAUSSIAN COMPOSITOR</small></span>
        </a>
        <nav className="workflow-global-nav" aria-label="主导航"><a href="#/">项目</a><a href="#/assets/video">素材库</a><a href="#/settings">设置</a></nav>
        <div className="project-heading">
          <span>当前项目</span><strong>{project.name}</strong>
        </div>
        <div className="header-status">
          <span className={`connection-dot connection-${taskState.connection}`} />
          <span>{taskState.connection === 'connected' ? '实时事件已连接' : 'REST 权威恢复可用'}</span>
          <VramBudgetControl
            backend={backend}
            budget={bootstrap.vram_budget}
            busy={startingStage || activeTaskRunning || unresolvedProjectOwner}
            onBudgetChange={acceptVramBudget}
          />
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
                  <button disabled={!reachable} onClick={() => navigateWorkflow(project.project_id, item)} type="button">
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
          {visibleError !== null ? (
            <div aria-atomic="true" className="error-banner" ref={errorRef} role="alert" tabIndex={-1}>
              <div><strong>{visibleError.category === 'repairable' ? '可以修复' : '操作未完成'}</strong><p>{visibleError.message}</p>{visibleError.code !== null ? <code>{visibleError.code}</code> : null}</div>
              <button aria-label="关闭错误提示" onClick={() => {
                setError(null)
                if (visibleTaskFailure !== null) setDismissedTaskErrorKey(visibleTaskFailure.key)
              }} type="button">×</button>
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
                  ? <span className="task-time task-time-elapsed">已用时 {formatTaskSeconds(authoritativeProgress.elapsed_seconds)} 秒</span>
                  : null}
                {authoritativeProgress?.eta_seconds !== null
                  && authoritativeProgress?.eta_seconds !== undefined
                  ? <span className="task-time task-time-eta">预计剩余 {formatTaskSeconds(authoritativeProgress.eta_seconds)} 秒</span>
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
          <button className="button-secondary" disabled={previous === undefined} onClick={() => previous !== undefined && navigateWorkflow(project.project_id, previous)} type="button">上一步</button>
          <button disabled={next === undefined || !canAdvance(project, step)} onClick={() => next !== undefined && navigateWorkflow(project.project_id, next)} type="button">下一步</button>
        </div>
      </footer>
    </AppShell>
  )
}
