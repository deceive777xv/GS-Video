import { StrictMode } from 'react'
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { BackendClient } from '../../api/backend-client'
import type {
  BootstrapDto,
  AssetKind,
  ProjectDto,
  StageStateDto,
  StageName,
  SubjectMediaDto,
  SubjectMediaRole,
  TaskDto,
  UploadCompleteDto,
} from '../../api/types'
import type { PickedFile, PickFileOptions, PlatformBridge } from '../../platform/platform-bridge'
import { App } from '../../app/app'
import { readUploadResume, writeUploadResume } from '../import/upload-resume'

afterEach(() => {
  sessionStorage.clear()
  vi.useRealTimers()
})

function stage(status: 'pending' | 'succeeded' = 'pending'): StageStateDto {
  return {
    status,
    cache_key: status === 'succeeded' ? 'cache' : null,
    output_paths: [],
    error_code: null,
    artifacts: {},
  }
}

function project(): ProjectDto {
  return {
    schema_version: 3,
    project_id: 'project-1',
    name: 'Studio replacement',
    created_at: '2026-07-17T00:00:00Z',
    source_video: null,
    scene_ply: null,
    stages: {
      ingest: stage(),
      segment: stage(),
      solve_camera: stage(),
      map_trajectory: stage(),
      render: stage(),
      composite: stage(),
      export: stage(),
    },
    workflow: {
      source_summary: null,
      scene_summary: null,
      subject_prompt: null,
      target_camera: null,
      preview_epoch: 0,
      confirmed_camera_revision: null,
      confirmed_preview_artifact_id: null,
      foot_point: null,
      motion_scale: 1,
      preview_height: 540,
      active_task_id: null,
      preview: null,
      export_result: null,
    },
  }
}

function bootstrap(current: ProjectDto): BootstrapDto {
  return {
    api_version: '1',
    capabilities: ['preview', 'export'],
    project: structuredClone(current),
    environment: {
      ready: true,
      vram_mb: 8192,
      vram_limit_mb: 8192,
      issues: [],
      renderer_versions: { renderer: 'fake' },
    },
  }
}

function createHarness(initial = project()) {
  let current = structuredClone(initial)
  let revision = 0
  const task = (target: StageName, status: TaskDto['status']): TaskDto => ({
    id: `task-${target}`,
    target_stage: target,
    status,
    revision: ++revision,
    error: null,
  })
  const client: BackendClient = {
    bootstrap: vi.fn(async () => bootstrap(current)),
    importLocalPath: vi.fn(),
    createUpload: vi.fn(async (input) => ({ id: `upload-${input.kind}`, chunk_size: 4 })),
    putUploadChunk: vi.fn(async () => undefined),
    getUpload: vi.fn(),
    completeUpload: vi.fn(async (id: string): Promise<UploadCompleteDto> => {
      const kind: AssetKind = id.includes('scene') ? 'scene_ply' : 'source_video'
      if (kind === 'source_video') {
        current.source_video = 'opaque:source'
        current.workflow.source_summary = {
          filename: 'portrait.mp4',
          size: 1024,
          sha256: 'source-sha',
          width: 1920,
          height: 1080,
          duration_seconds: 12,
          fps: '30/1',
          has_audio: true,
          frame_count: 360,
        }
      } else {
        current.scene_ply = 'opaque:scene'
        current.workflow.scene_summary = {
          filename: 'garden.ply',
          size: 2048,
          sha256: 'scene-sha',
          gaussian_count: 1_250_000,
          estimated_vram_mb: 3072,
        }
      }
      return { path: `opaque:${kind}`, kind, size: 1, sha256: `${kind}-sha` }
    }),
    cancelUpload: vi.fn(async () => undefined),
    getProject: vi.fn(async () => structuredClone(current)),
    updateProject: vi.fn(async (patch) => {
      if (patch.subject_prompt !== undefined) current.workflow.subject_prompt = patch.subject_prompt
      if (patch.motion_scale !== undefined) current.workflow.motion_scale = patch.motion_scale
      if (patch.preview_height !== undefined) current.workflow.preview_height = patch.preview_height
      return structuredClone(current)
    }),
    renderPreview: vi.fn(async (input) => {
      current.workflow.target_camera = { ...input.camera, revision: input.generation }
      current.workflow.preview = {
        artifact_id: `preview-${input.generation}`,
        artifact_size: 8,
        artifact_sha256: 'preview-sha',
        generation: input.generation,
        width: input.width,
        height: input.height,
        camera_revision: input.generation,
        pick_buffer_revision: input.generation,
      }
      return {
        artifact_id: `preview-${input.generation}`,
        generation: input.generation,
        width: input.width,
        height: input.height,
        camera_revision: input.generation,
        pick_buffer_revision: input.generation,
      }
    }),
    fetchPreviewArtifact: vi.fn(async () => new Blob(['preview'], { type: 'image/png' })),
    pickFootPoint: vi.fn(async (input) => {
      const foot = {
        image: [input.x, input.y] as [number, number],
        world: [0, 0, 0] as [number, number, number],
        preview_artifact_id: input.preview_artifact_id,
        camera_revision: input.camera_revision,
        pick_buffer_revision: input.pick_buffer_revision,
      }
      current.workflow.foot_point = foot
      return foot
    }),
    confirmCamera: vi.fn(async (cameraRevision) => {
      current.workflow.confirmed_camera_revision = cameraRevision
      current.workflow.confirmed_preview_artifact_id = current.workflow.preview?.artifact_id ?? null
      return structuredClone(current)
    }),
    getVerifiedExport: vi.fn(async () => ({
      artifact_id: 'export-1', filename: 'result.mp4', size: 12,
      duration_seconds: 12, fps: '30/1', frame_count: 360,
      has_audio: true, verified: true,
    })),
    fetchExportArtifact: vi.fn(async () => new Blob(['mp4'], { type: 'video/mp4' })),
    getCompositePreview: vi.fn(async () => ({
      artifact_id: 'composite-1', filename: 'composite-preview.mp4' as const, size: 12,
      sha256: 'c'.repeat(64), duration_seconds: 2, fps: '24/1', frame_count: 48,
    })),
    fetchCompositePreviewArtifact: vi.fn(async () => new Blob(['composite'], { type: 'video/mp4' })),
    copyVerifiedExport: vi.fn(async () => undefined),
    getSubjectMedia: vi.fn(async (role: SubjectMediaRole): Promise<SubjectMediaDto> => ({
      role, artifact_id: `${role}-1`, frame_index: 0,
      width: 640, height: 360, size: 8,
      mime_type: role === 'proxy' ? 'image/jpeg' as const : 'image/png' as const,
    })),
    fetchSubjectMediaArtifact: vi.fn(async (role) => new Blob([role], {
      type: role === 'proxy' ? 'image/jpeg' : 'image/png',
    })),
    startTask: vi.fn(async (target: StageName): Promise<TaskDto> => {
      current.stages[target] = stage('succeeded')
      if (target === 'composite') current.stages.composite = stage('succeeded')
      if (target === 'export') {
        current.stages.export = stage('succeeded')
        current.workflow.export_result = {
          artifact_id: 'export-1', filename: 'result.mp4', size: 12,
          sha256: 'export-sha', duration_seconds: 12, fps: '30/1',
          frame_count: 360, has_audio: true, verified: true,
        }
      }
      const completed = task(target, 'succeeded')
      current.workflow.active_task_id = completed.id
      return completed
    }),
    getTask: vi.fn(async (id: string): Promise<TaskDto> => ({
      id, target_stage: id.replace('task-', '') as StageName,
      status: 'succeeded' as const, revision: ++revision, error: null,
    })),
    cancelTask: vi.fn(async (id: string): Promise<TaskDto> => ({
      id, target_stage: 'segment' as const, status: 'cancelled' as const, revision: ++revision, error: null,
    })),
  }
  const platform: PlatformBridge = {
    kind: 'browser',
    pickInputFile: vi.fn(async ({ kind }: PickFileOptions): Promise<PickedFile> => ({
      kind: 'browser-file' as const,
      file: new File([kind], kind === 'source_video' ? 'portrait.mp4' : 'garden.ply'),
    })),
    saveExport: vi.fn(async () => undefined),
    openExternal: vi.fn(async () => undefined),
  }
  return { client, platform, getProject: () => structuredClone(current) }
}

describe('guided workflow', () => {
  it('completes the happy path with exactly three authoritative creative interactions', async () => {
    const user = userEvent.setup()
    const harness = createHarness()
    render(<App backend={harness.client} platform={harness.platform} />)

    await screen.findByRole('heading', { name: '导入素材' })
    await user.click(screen.getByRole('button', { name: '选择源视频' }))
    await user.click(screen.getByRole('button', { name: '选择 Gaussian 场景' }))
    await user.click(screen.getByRole('button', { name: '下一步' }))

    await user.type(screen.getByLabelText('人物 X 坐标'), '100')
    await user.type(screen.getByLabelText('人物 Y 坐标'), '120')
    await user.click(screen.getByRole('button', { name: '确认人物位置' }))
    expect(await screen.findByLabelText('创作交互 1 / 3')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: '下一步' }))
    const confirmCamera = await screen.findByRole('button', { name: '确认初始机位' })
    await waitFor(() => expect(confirmCamera).toBeEnabled())
    await user.click(confirmCamera)
    expect(await screen.findByLabelText('创作交互 2 / 3')).toBeInTheDocument()
    await user.type(screen.getByLabelText('落脚点 X 坐标'), '320')
    await user.type(screen.getByLabelText('落脚点 Y 坐标'), '180')
    await user.click(screen.getByRole('button', { name: '确认场景落脚点' }))
    expect(await screen.findByLabelText('创作交互 3 / 3')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: '下一步' }))
    await user.click(screen.getByRole('button', { name: '生成预览' }))
    await user.click(screen.getByRole('button', { name: '下一步' }))
    expect(await screen.findByRole('button', { name: '导出视频' })).toBeEnabled()
    expect(harness.client.updateProject).toHaveBeenCalledWith({
      subject_prompt: { frame_index: 0, x: 100, y: 120 },
    })
  })

  it('recovers the authoritative page and interaction count from bootstrap', async () => {
    const recovered = project()
    recovered.source_video = 'opaque:source'
    recovered.scene_ply = 'opaque:scene'
    recovered.workflow.source_summary = {
      filename: 'portrait.mp4', size: 10, sha256: 's', width: 640, height: 360,
      duration_seconds: 12, fps: '30/1', has_audio: true, frame_count: 360,
    }
    recovered.workflow.scene_summary = {
      filename: 'garden.ply', size: 20, sha256: 'g', gaussian_count: 100,
      estimated_vram_mb: 128,
    }
    recovered.workflow.subject_prompt = { frame_index: 0, x: 100, y: 120 }
    recovered.stages.segment = stage('succeeded')
    recovered.stages.solve_camera = stage('succeeded')
    recovered.workflow.target_camera = {
      target: [0, 0, 0], distance: 4, yaw: 0, pitch: 0,
      fov_y_degrees: 50, revision: 7,
    }
    recovered.workflow.preview = {
      artifact_id: 'preview-7', artifact_size: 8, artifact_sha256: 'p',
      generation: 7, width: 960, height: 540, camera_revision: 7,
      pick_buffer_revision: 7,
    }
    recovered.workflow.confirmed_camera_revision = 7
    recovered.workflow.confirmed_preview_artifact_id = 'preview-7'
    const harness = createHarness(recovered)
    render(
      <App
        backend={harness.client}
        initialBootstrap={bootstrap(recovered)}
        platform={harness.platform}
      />,
    )

    expect(await screen.findByLabelText('创作交互 2 / 3')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: '放置目标镜头' })).toBeInTheDocument()
    expect(harness.client.bootstrap).not.toHaveBeenCalled()
  })

  it('keeps the camera step gated until source camera analysis is authoritative', async () => {
    const gated = project()
    gated.source_video = 'opaque:source'
    gated.scene_ply = 'opaque:scene'
    gated.workflow.source_summary = {
      filename: 'portrait.mp4', size: 10, sha256: 's', width: 640, height: 360,
      duration_seconds: 12, fps: '30/1', has_audio: true, frame_count: 360,
    }
    gated.workflow.scene_summary = {
      filename: 'garden.ply', size: 20, sha256: 'g', gaussian_count: 100,
      estimated_vram_mb: 128,
    }
    gated.workflow.subject_prompt = { frame_index: 0, x: 10, y: 10 }
    gated.stages.segment = stage('succeeded')
    const harness = createHarness(gated)
    vi.mocked(harness.client.startTask).mockImplementation(async () => new Promise<TaskDto>(() => undefined))
    render(<App backend={harness.client} initialBootstrap={bootstrap(gated)} platform={harness.platform} />)

    expect(screen.getByRole('button', { name: /机位/ })).toBeDisabled()
    await waitFor(() => expect(harness.client.startTask).toHaveBeenCalledWith('solve_camera'))
  })

  it('owns and disposes its task store once across StrictMode cleanup', async () => {
    const harness = createHarness()
    const dispose = vi.fn()
    const snapshot = {
      task: null,
      revision: 0,
      pendingResyncRevision: null,
      connection: 'disconnected' as const,
      latestEvent: null,
    }
    const createOwnedTaskStore = vi.fn(() => ({
      subscribe: () => () => undefined,
      snapshot: () => snapshot,
      onEvent: vi.fn(), onConnectionChange: vi.fn(), replaceFromRest: vi.fn(), acknowledgeResync: vi.fn(),
      whenIdle: async () => undefined, dispose,
    }))
    const view = render(
      <StrictMode>
        <App
          backend={harness.client}
          createOwnedTaskStore={createOwnedTaskStore}
          platform={harness.platform}
        />
      </StrictMode>,
    )
    await screen.findByRole('heading', { name: '导入素材' })
    view.unmount()
    expect(createOwnedTaskStore).toHaveBeenCalledTimes(2)
    await waitFor(() => expect(dispose).toHaveBeenCalledTimes(1))
  })

  it('cancels an active task without presenting cancelled work as retryable', async () => {
    const active = project()
    active.workflow.active_task_id = 'task-composite'
    const harness = createHarness(active)
    vi.mocked(harness.client.getTask).mockResolvedValueOnce({
      id: 'task-composite', target_stage: 'composite', status: 'running',
      revision: 1, error: null,
    })
    vi.mocked(harness.client.cancelTask).mockResolvedValueOnce({
      id: 'task-composite', target_stage: 'composite', status: 'cancelled',
      revision: 2, error: 'cancelled_by_user',
    })
    const user = userEvent.setup()
    render(<App backend={harness.client} initialBootstrap={bootstrap(active)} platform={harness.platform} />)

    await user.click(await screen.findByRole('button', { name: '取消任务' }))
    expect(harness.client.cancelTask).toHaveBeenCalledWith('task-composite')
    expect(screen.queryByRole('button', { name: '重试阶段' })).toBeNull()
    expect(harness.client.startTask).not.toHaveBeenCalled()
  })

  it('polls REST until an active task converges while realtime events are disconnected', async () => {
    const active = project()
    active.workflow.active_task_id = 'task-ingest'
    const harness = createHarness(active)
    vi.mocked(harness.client.getTask)
      .mockResolvedValueOnce({ id: 'task-ingest', target_stage: 'ingest', status: 'running', revision: 1, error: null })
      .mockResolvedValueOnce({ id: 'task-ingest', target_stage: 'ingest', status: 'succeeded', revision: 2, error: null })
    render(<App backend={harness.client} initialBootstrap={bootstrap(active)} platform={harness.platform} />)

    expect(await screen.findByText(/ingest · succeeded/)).toBeInTheDocument()
    expect(harness.client.getTask).toHaveBeenCalledTimes(2)
    expect(harness.client.getProject).toHaveBeenCalled()
  })

  it('retries initial active-task recovery after a transient failure and acknowledges only success', async () => {
    vi.useFakeTimers()
    const active = project()
    active.workflow.active_task_id = 'task-ingest'
    const harness = createHarness(active)
    vi.mocked(harness.client.getTask)
      .mockRejectedValueOnce(new Error('temporarily offline'))
      .mockResolvedValueOnce({
        id: 'task-ingest', target_stage: 'ingest', status: 'succeeded',
        revision: 2, error: null,
      })

    render(<App backend={harness.client} initialBootstrap={bootstrap(active)} platform={harness.platform} />)
    await act(async () => { await Promise.resolve() })
    expect(harness.client.getTask).toHaveBeenCalledTimes(1)

    await act(async () => { await vi.advanceTimersByTimeAsync(250) })
    expect(harness.client.getTask).toHaveBeenCalledTimes(2)
    expect(screen.getByText(/ingest · succeeded/)).toBeInTheDocument()
    vi.useRealTimers()
  })

  it('cleans up bounded initial active-task recovery when the app unmounts', async () => {
    vi.useFakeTimers()
    const active = project()
    active.workflow.active_task_id = 'task-ingest'
    const harness = createHarness(active)
    vi.mocked(harness.client.getTask).mockRejectedValue(new Error('offline'))
    const view = render(<App backend={harness.client} initialBootstrap={bootstrap(active)} platform={harness.platform} />)
    await act(async () => { await Promise.resolve() })
    expect(harness.client.getTask).toHaveBeenCalledOnce()

    view.unmount()
    await act(async () => { await vi.advanceTimersByTimeAsync(5_000) })
    expect(harness.client.getTask).toHaveBeenCalledOnce()
    expect(vi.getTimerCount()).toBe(0)
    vi.useRealTimers()
  })

  it('treats an unrecovered project task owner as provisionally busy', async () => {
    const active = project()
    active.workflow.active_task_id = 'task-ingest'
    const harness = createHarness(active)
    vi.mocked(harness.client.getTask).mockImplementation(() => new Promise(() => undefined))

    render(<App backend={harness.client} initialBootstrap={bootstrap(active)} platform={harness.platform} />)
    expect(screen.getByRole('button', { name: '选择源视频' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '选择 Gaussian 场景' })).toBeDisabled()
    expect(harness.client.startTask).not.toHaveBeenCalled()
  })

  it('releases a stale persisted owner only after a deterministic task-not-found response', async () => {
    const active = project()
    active.workflow.active_task_id = 'task-from-old-service'
    const harness = createHarness(active)
    vi.mocked(harness.client.getTask).mockRejectedValueOnce(Object.assign(
      new Error('task not found'),
      { status: 404, code: 'task_not_found' },
    ))

    render(<App backend={harness.client} initialBootstrap={bootstrap(active)} platform={harness.platform} />)
    expect(screen.getByRole('button', { name: '选择源视频' })).toBeDisabled()
    await waitFor(() => expect(screen.getByRole('button', { name: '选择源视频' })).toBeEnabled())
    expect(screen.queryByRole('alert')).toBeNull()
  })

  it('switches App REST recovery from an old project owner to the newer gap event task', async () => {
    const active = project()
    active.workflow.active_task_id = 'task-old'
    const harness = createHarness(active)
    let resolveOld: ((value: TaskDto) => void) | undefined
    vi.mocked(harness.client.getTask).mockImplementation((id) => id === 'task-old'
      ? new Promise((resolve) => { resolveOld = resolve })
      : Promise.resolve({
          id, target_stage: 'render', status: 'running', revision: 9, error: null,
        }))
    let subscription: Parameters<NonNullable<Parameters<typeof App>[0]['eventSource']>['subscribe']>[0] | undefined
    const eventSource = {
      subscribe: vi.fn((next: NonNullable<typeof subscription>) => {
        subscription = next
        return () => undefined
      }),
    }
    render(<App backend={harness.client} eventSource={eventSource} initialBootstrap={bootstrap(active)} platform={harness.platform} />)
    await waitFor(() => expect(harness.client.getTask).toHaveBeenCalledWith('task-old'))

    act(() => subscription?.onEvent({
      type: 'resync_required', task_id: 'task-new', revision: 9,
    }))
    await waitFor(() => expect(harness.client.getTask).toHaveBeenCalledWith('task-new'))
    await screen.findByText(/render · running/)

    resolveOld?.({
      id: 'task-old', target_stage: 'segment', status: 'running', revision: 1, error: null,
    })
    await act(async () => { await Promise.resolve() })
    expect(screen.getByText(/render · running/)).toBeInTheDocument()
    expect(screen.queryByText(/segment · running/)).toBeNull()
  })

  it('keeps a browser upload session for retry and focuses the structured error alert', async () => {
    const user = userEvent.setup()
    const harness = createHarness()
    vi.mocked(harness.client.putUploadChunk)
      .mockRejectedValueOnce(new Error('chunk interrupted'))
      .mockResolvedValue(undefined)
    vi.mocked(harness.client.getUpload).mockResolvedValue({
      id: 'upload-source_video', chunk_size: 4, kind: 'source_video',
      filename: 'portrait.mp4', total_size: 12, uploaded_chunks: [],
    })
    render(<App backend={harness.client} platform={harness.platform} />)
    await screen.findByRole('heading', { name: '导入素材' })

    await user.click(screen.getByRole('button', { name: '选择源视频' }))
    const alert = await screen.findByRole('alert')
    expect(alert).toHaveFocus()
    await user.click(screen.getByRole('button', { name: '选择源视频' }))
    await waitFor(() => expect(harness.client.completeUpload).toHaveBeenCalledTimes(1))
    expect(harness.client.createUpload).toHaveBeenCalledTimes(1)
    expect(harness.client.getUpload).toHaveBeenCalledWith('upload-source_video')
  })

  it('admits only one local import while file selection is still pending', async () => {
    const harness = createHarness()
    let resolvePick: ((value: PickedFile | null) => void) | undefined
    vi.mocked(harness.platform.pickInputFile).mockImplementation(
      () => new Promise((resolve) => { resolvePick = resolve }),
    )
    const user = userEvent.setup()
    render(<App backend={harness.client} platform={harness.platform} />)
    await screen.findByRole('heading', { name: '导入素材' })

    await user.click(screen.getByRole('button', { name: '选择源视频' }))
    expect(screen.getByRole('button', { name: '选择源视频' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '选择 Gaussian 场景' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: '选择 Gaussian 场景' }))
    expect(harness.platform.pickInputFile).toHaveBeenCalledOnce()

    resolvePick?.(null)
    await waitFor(() => expect(screen.getByRole('button', { name: '选择源视频' })).toBeEnabled())
  })

  it('keeps a server upload resumable across unmount and resumes only missing chunks after reselection', async () => {
    const first = createHarness()
    let abortedSignal: AbortSignal | undefined
    vi.mocked(first.client.putUploadChunk).mockImplementation(async (_id, index, _blob, signal) => {
      if (index === 0) return
      abortedSignal = signal
      await new Promise<void>((_resolve, reject) => {
        signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')), { once: true })
      })
    })
    const user = userEvent.setup()
    const firstView = render(<App backend={first.client} platform={first.platform} />)
    await screen.findByRole('heading', { name: '导入素材' })
    await user.click(screen.getByRole('button', { name: '选择源视频' }))
    await waitFor(() => expect(first.client.putUploadChunk).toHaveBeenCalledTimes(2))
    expect(readUploadResume('project-1', 'source_video')?.id).toBe('upload-source_video')

    firstView.unmount()
    expect(abortedSignal?.aborted).toBe(true)
    expect(first.client.cancelUpload).not.toHaveBeenCalled()

    const second = createHarness()
    vi.mocked(second.client.getUpload).mockResolvedValue({
      id: 'upload-source_video', chunk_size: 4, kind: 'source_video',
      filename: 'portrait.mp4', total_size: 12, uploaded_chunks: [0],
    })
    render(<App backend={second.client} platform={second.platform} />)
    await screen.findByRole('heading', { name: '导入素材' })
    await user.click(screen.getByRole('button', { name: '选择源视频' }))
    await waitFor(() => expect(second.client.completeUpload).toHaveBeenCalledWith('upload-source_video'))
    expect(second.client.createUpload).not.toHaveBeenCalled()
    expect(second.client.putUploadChunk).toHaveBeenCalledTimes(2)
    expect(second.client.putUploadChunk).toHaveBeenNthCalledWith(1, 'upload-source_video', 1, expect.any(Blob), expect.any(AbortSignal))
    expect(readUploadResume('project-1', 'source_video')).toBeNull()
  })

  it('cancels and clears a mismatched superseded upload before creating a new session', async () => {
    writeUploadResume({
      version: 1, projectId: 'project-1', kind: 'source_video',
      filename: 'old.mp4', mimeType: 'video/mp4', size: 99,
      sha256: 'a'.repeat(64), id: 'upload-old', chunkSize: 4,
    })
    const harness = createHarness()
    const user = userEvent.setup()
    render(<App backend={harness.client} platform={harness.platform} />)
    await screen.findByRole('heading', { name: '导入素材' })

    await user.click(screen.getByRole('button', { name: '选择源视频' }))
    await waitFor(() => expect(harness.client.completeUpload).toHaveBeenCalled())
    expect(harness.client.cancelUpload).toHaveBeenCalledWith('upload-old')
    expect(vi.mocked(harness.client.cancelUpload).mock.invocationCallOrder[0])
      .toBeLessThan(vi.mocked(harness.client.createUpload).mock.invocationCallOrder[0]!)
  })

  it('replaces an expired matching upload session instead of trapping later selections', async () => {
    const selected = new File(['source_video'], 'portrait.mp4')
    const digest = await crypto.subtle.digest('SHA-256', await selected.arrayBuffer())
    const sha256 = Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('')
    writeUploadResume({
      version: 1, projectId: 'project-1', kind: 'source_video',
      filename: selected.name, mimeType: 'application/octet-stream', size: selected.size,
      sha256, id: 'upload-expired', chunkSize: 4,
    })
    const harness = createHarness()
    vi.mocked(harness.client.getUpload).mockRejectedValueOnce(Object.assign(
      new Error('upload not found'),
      { status: 404 },
    ))
    const user = userEvent.setup()
    render(<App backend={harness.client} platform={harness.platform} />)
    await screen.findByRole('heading', { name: '导入素材' })

    await user.click(screen.getByRole('button', { name: '选择源视频' }))
    await waitFor(() => expect(harness.client.completeUpload).toHaveBeenCalledWith('upload-source_video'))
    expect(harness.client.createUpload).toHaveBeenCalledOnce()
    expect(harness.client.cancelUpload).not.toHaveBeenCalledWith('upload-expired')
    expect(readUploadResume('project-1', 'source_video')).toBeNull()
  })

  it('continues after a mismatched upload was already removed by the service', async () => {
    writeUploadResume({
      version: 1, projectId: 'project-1', kind: 'source_video',
      filename: 'old.mp4', mimeType: 'video/mp4', size: 99,
      sha256: 'a'.repeat(64), id: 'upload-already-gone', chunkSize: 4,
    })
    const harness = createHarness()
    vi.mocked(harness.client.cancelUpload).mockRejectedValueOnce(Object.assign(
      new Error('upload not found'),
      { status: 404 },
    ))
    const user = userEvent.setup()
    render(<App backend={harness.client} platform={harness.platform} />)
    await screen.findByRole('heading', { name: '导入素材' })

    await user.click(screen.getByRole('button', { name: '选择源视频' }))
    await waitFor(() => expect(harness.client.completeUpload).toHaveBeenCalledWith('upload-source_video'))
    expect(harness.client.createUpload).toHaveBeenCalledOnce()
    expect(readUploadResume('project-1', 'source_video')).toBeNull()
  })

  it('explicit upload cancellation destroys the server session and resume metadata', async () => {
    writeUploadResume({
      version: 1, projectId: 'project-1', kind: 'scene_ply',
      filename: 'scene.ply', mimeType: 'application/octet-stream', size: 10,
      sha256: 'b'.repeat(64), id: 'upload-scene', chunkSize: 4,
    })
    const harness = createHarness()
    vi.mocked(harness.client.cancelUpload).mockRejectedValueOnce(Object.assign(
      new Error('upload already removed'),
      { status: 404 },
    ))
    vi.mocked(harness.client.putUploadChunk).mockImplementation(async (_id, _index, _blob, signal) => {
      await new Promise<void>((_resolve, reject) => {
        signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')), { once: true })
      })
    })
    const user = userEvent.setup()
    render(<App backend={harness.client} platform={harness.platform} />)
    await screen.findByRole('heading', { name: '导入素材' })
    await user.click(screen.getByRole('button', { name: '选择源视频' }))
    await user.click(await screen.findByRole('button', { name: '取消当前上传' }))

    expect(harness.client.cancelUpload).toHaveBeenCalledWith('upload-source_video')
    expect(readUploadResume('project-1', 'source_video')).toBeNull()
    expect(readUploadResume('project-1', 'scene_ply')?.id).toBe('upload-scene')
    await waitFor(() => expect(screen.queryByRole('alert')).toBeNull())
  })

  it('keeps local import admission closed until server cancellation finishes', async () => {
    const harness = createHarness()
    vi.mocked(harness.client.putUploadChunk).mockImplementation(async (_id, _index, _blob, signal) => {
      await new Promise<void>((_resolve, reject) => {
        signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')), { once: true })
      })
    })
    let resolveCancel: (() => void) | undefined
    vi.mocked(harness.client.cancelUpload).mockImplementation(
      () => new Promise((resolve) => { resolveCancel = resolve }),
    )
    const user = userEvent.setup()
    render(<App backend={harness.client} platform={harness.platform} />)
    await screen.findByRole('heading', { name: '导入素材' })
    await user.click(screen.getByRole('button', { name: '选择源视频' }))
    await waitFor(() => expect(harness.client.putUploadChunk).toHaveBeenCalled())

    await user.click(screen.getByRole('button', { name: '取消当前上传' }))
    await waitFor(() => expect(harness.client.cancelUpload).toHaveBeenCalled())
    expect(screen.getByRole('button', { name: '选择源视频' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '选择 Gaussian 场景' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: '选择 Gaussian 场景' }))
    expect(harness.platform.pickInputFile).toHaveBeenCalledOnce()

    resolveCancel?.()
    await waitFor(() => expect(screen.getByRole('button', { name: '选择源视频' })).toBeEnabled())
    expect(readUploadResume('project-1', 'source_video')).toBeNull()
  })

  it('does not let an older preview refresh overwrite a newer camera confirmation', async () => {
    const recovered = project()
    recovered.source_video = 'opaque:source'
    recovered.scene_ply = 'opaque:scene'
    recovered.workflow.source_summary = {
      filename: 'portrait.mp4', size: 10, sha256: 's', width: 640, height: 360,
      duration_seconds: 12, fps: '30/1', has_audio: true, frame_count: 360,
    }
    recovered.workflow.scene_summary = {
      filename: 'garden.ply', size: 20, sha256: 'g', gaussian_count: 100,
      estimated_vram_mb: 128,
    }
    recovered.workflow.subject_prompt = { frame_index: 0, x: 100, y: 120 }
    recovered.stages.segment = stage('succeeded')
    recovered.stages.solve_camera = stage('succeeded')
    recovered.workflow.target_camera = {
      target: [0, 0, 0], distance: 4, yaw: 0, pitch: 0,
      fov_y_degrees: 50, revision: 1,
    }
    recovered.workflow.preview = {
      artifact_id: 'preview-1', artifact_size: 8, artifact_sha256: 'p',
      generation: 1, width: 960, height: 540, camera_revision: 1,
      pick_buffer_revision: 1,
    }
    const harness = createHarness(recovered)
    const stale = harness.getProject()
    let resolveRefresh!: (project: ProjectDto) => void
    vi.mocked(harness.client.getProject).mockImplementationOnce(async () => new Promise<ProjectDto>((resolve) => { resolveRefresh = resolve }))
    const user = userEvent.setup()
    render(<App backend={harness.client} initialBootstrap={bootstrap(recovered)} platform={harness.platform} />)

    const fov = screen.getByRole('slider', { name: '垂直视场角' })
    await user.click(fov)
    await user.keyboard('{ArrowRight}')
    await waitFor(() => expect(harness.client.getProject).toHaveBeenCalledTimes(1))
    await user.click(screen.getByRole('button', { name: '确认初始机位' }))
    expect(await screen.findByLabelText('创作交互 2 / 3')).toBeInTheDocument()
    await act(async () => {
      resolveRefresh(stale)
      await Promise.resolve()
    })
    expect(screen.getByLabelText('创作交互 2 / 3')).toBeInTheDocument()
  })

  it('disables every task-starting control while any authoritative owner is active', async () => {
    const ready = project()
    ready.source_video = 'opaque:source'
    ready.scene_ply = 'opaque:scene'
    ready.workflow.source_summary = {
      filename: 'portrait.mp4', size: 10, sha256: 's', width: 640, height: 360,
      duration_seconds: 12, fps: '30/1', has_audio: true, frame_count: 360,
    }
    ready.workflow.scene_summary = {
      filename: 'garden.ply', size: 20, sha256: 'g', gaussian_count: 100,
      estimated_vram_mb: 128,
    }
    ready.workflow.subject_prompt = { frame_index: 0, x: 10, y: 10 }
    ready.stages.segment = stage('succeeded')
    ready.stages.solve_camera = stage('succeeded')
    ready.stages.composite = stage('succeeded')
    ready.workflow.target_camera = { target: [0, 0, 0], distance: 4, yaw: 0, pitch: 0, fov_y_degrees: 50, revision: 1 }
    ready.workflow.preview = { artifact_id: 'preview-1', artifact_size: 1, artifact_sha256: 'p', generation: 1, width: 960, height: 540, camera_revision: 1, pick_buffer_revision: 1 }
    ready.workflow.confirmed_camera_revision = 1
    ready.workflow.confirmed_preview_artifact_id = 'preview-1'
    ready.workflow.foot_point = { image: [10, 10], world: [0, 0, 0], preview_artifact_id: 'preview-1', camera_revision: 1, pick_buffer_revision: 1 }
    ready.workflow.active_task_id = 'task-render'
    const harness = createHarness(ready)
    const active = { id: 'task-render', target_stage: 'render' as const, status: 'running' as const, revision: 4, error: null }
    const snapshot = { task: active, revision: 4, pendingResyncRevision: null, connection: 'connected' as const, latestEvent: null }
    const createOwnedTaskStore = () => ({
      subscribe: () => () => undefined,
      snapshot: () => snapshot,
      onEvent: vi.fn(), onConnectionChange: vi.fn(), replaceFromRest: vi.fn(), acknowledgeResync: vi.fn(),
      whenIdle: async () => undefined, dispose: vi.fn(),
    })
    const user = userEvent.setup()
    render(<App backend={harness.client} createOwnedTaskStore={createOwnedTaskStore} initialBootstrap={bootstrap(ready)} platform={harness.platform} />)

    expect(screen.getByRole('button', { name: '验证导出中…' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: /导入.*视频 \+ PLY/ }))
    expect(screen.getByRole('button', { name: '选择源视频' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '选择 Gaussian 场景' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: /人物.*一次提示/ }))
    expect(await screen.findByRole('button', { name: '确认人物位置' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: /预览.*运动与合成/ }))
    expect(screen.getByRole('button', { name: '生成中…' })).toBeDisabled()
    expect(harness.client.startTask).not.toHaveBeenCalled()
  })

  it('lets disconnected REST polling converge a newer owner past an older event', async () => {
    vi.useFakeTimers()
    const current = project()
    const harness = createHarness(current)
    const queued: TaskDto = {
      id: 'task-new', target_stage: 'render', status: 'queued', revision: 9, error: null,
    }
    const succeeded: TaskDto = { ...queued, status: 'succeeded', revision: 10 }
    vi.mocked(harness.client.getTask).mockResolvedValue(succeeded)
    const snapshot = {
      task: queued,
      revision: 9,
      pendingResyncRevision: null,
      connection: 'disconnected' as const,
      latestEvent: {
        type: 'task_event' as const,
        task_id: 'task-old',
        revision: 8,
        stage: 'segment' as const,
        progress: 1,
        error: null,
      },
    }
    const replaceFromRest = vi.fn()
    const createOwnedTaskStore = () => ({
      subscribe: () => () => undefined,
      snapshot: () => snapshot,
      onEvent: vi.fn(), onConnectionChange: vi.fn(), replaceFromRest, acknowledgeResync: vi.fn(),
      whenIdle: async () => undefined, dispose: vi.fn(),
    })
    render(<App backend={harness.client} createOwnedTaskStore={createOwnedTaskStore} initialBootstrap={bootstrap(current)} platform={harness.platform} />)

    await act(async () => { await vi.advanceTimersByTimeAsync(250) })
    expect(replaceFromRest).toHaveBeenCalledWith(succeeded)
    vi.useRealTimers()
  })

  it('admits only one stage request for a rapid double click', async () => {
    const ready = project()
    ready.workflow.source_summary = {
      filename: 'portrait.mp4', size: 10, sha256: 's', width: 640, height: 360,
      duration_seconds: 12, fps: '30/1', has_audio: true, frame_count: 360,
    }
    ready.workflow.scene_summary = { filename: 'garden.ply', size: 20, sha256: 'g', gaussian_count: 100, estimated_vram_mb: 128 }
    ready.workflow.subject_prompt = { frame_index: 0, x: 10, y: 10 }
    ready.stages.segment = stage('succeeded')
    ready.stages.solve_camera = stage('succeeded')
    ready.workflow.target_camera = { target: [0, 0, 0], distance: 4, yaw: 0, pitch: 0, fov_y_degrees: 50, revision: 1 }
    ready.workflow.confirmed_camera_revision = 1
    ready.workflow.confirmed_preview_artifact_id = 'preview-1'
    ready.workflow.foot_point = { image: [10, 10], world: [0, 0, 0], preview_artifact_id: 'preview-1', camera_revision: 1, pick_buffer_revision: 1 }
    const harness = createHarness(ready)
    vi.mocked(harness.client.startTask).mockImplementation(() => new Promise(() => undefined))
    render(<App backend={harness.client} initialBootstrap={bootstrap(ready)} platform={harness.platform} />)

    const generate = screen.getByRole('button', { name: '生成预览' })
    fireEvent.click(generate)
    fireEvent.click(generate)
    expect(harness.client.startTask).toHaveBeenCalledTimes(1)
  })

  it('renders latest matching event progress as a determinate accessible progressbar', () => {
    const current = project()
    current.workflow.active_task_id = 'task-render'
    const harness = createHarness(current)
    const task = { id: 'task-render', target_stage: 'render' as const, status: 'running' as const, revision: 8, error: null }
    const snapshot = {
      task, revision: 8, pendingResyncRevision: null, connection: 'connected' as const,
      latestEvent: { type: 'task_event' as const, task_id: task.id, revision: 8, stage: 'render' as const, progress: 0.42, error: null },
    }
    const createOwnedTaskStore = () => ({
      subscribe: () => () => undefined, snapshot: () => snapshot,
      onEvent: vi.fn(), onConnectionChange: vi.fn(), replaceFromRest: vi.fn(), acknowledgeResync: vi.fn(),
      whenIdle: async () => undefined, dispose: vi.fn(),
    })
    render(<App backend={harness.client} createOwnedTaskStore={createOwnedTaskStore} initialBootstrap={bootstrap(current)} platform={harness.platform} />)

    const progress = screen.getByRole('progressbar', { name: 'render 进度' })
    expect(progress).toHaveAttribute('aria-valuenow', '42')
    expect(progress).toHaveTextContent('42%')
    expect(screen.getByText('render · running').parentElement).toHaveClass('task-copy')
  })

  it('shows backend current-frame timing from the latest matching WebSocket event', () => {
    const current = project()
    current.workflow.active_task_id = 'task-render'
    const harness = createHarness(current)
    const task = {
      id: 'task-render', target_stage: 'render' as const, status: 'running' as const,
      revision: 8, error: null,
    }
    const snapshot = {
      task, revision: 8, pendingResyncRevision: null, connection: 'connected' as const,
      latestEvent: {
        type: 'task_event' as const,
        task_id: task.id,
        revision: 8,
        stage: 'render' as const,
        progress: 0.4,
        current: 12,
        total: 30,
        message: '渲染背景 12/30',
        elapsed_seconds: 8,
        eta_seconds: 12,
        error: null,
      },
    }
    const createOwnedTaskStore = () => ({
      subscribe: () => () => undefined, snapshot: () => snapshot,
      onEvent: vi.fn(), onConnectionChange: vi.fn(), replaceFromRest: vi.fn(), acknowledgeResync: vi.fn(),
      whenIdle: async () => undefined, dispose: vi.fn(),
    })

    render(<App backend={harness.client} createOwnedTaskStore={createOwnedTaskStore} initialBootstrap={bootstrap(current)} platform={harness.platform} />)

    expect(screen.getByText('渲染背景 12/30')).toBeVisible()
    expect(screen.getByText('12 / 30')).toBeVisible()
    expect(screen.getByText('已用时 8 秒')).toBeVisible()
    expect(screen.getByText('预计剩余 12 秒')).toBeVisible()
  })

  it('uses the matching REST task snapshot and omits a null ETA without inventing one', () => {
    const current = project()
    current.workflow.active_task_id = 'task-render'
    const harness = createHarness(current)
    const task = {
      id: 'task-render', target_stage: 'render' as const, status: 'running' as const,
      revision: 9, error: null, progress: 0.5, current: 15, total: 30,
      message: 'REST 恢复渲染 15/30', elapsed_seconds: 10, eta_seconds: null,
    }
    const snapshot = {
      task, revision: 9, pendingResyncRevision: null, connection: 'disconnected' as const,
      latestEvent: {
        type: 'task_event' as const, task_id: 'task-stale', revision: 10,
        stage: 'render' as const, progress: 0.9, current: 27, total: 30,
        message: '过期事件', elapsed_seconds: 20, eta_seconds: 2, error: null,
      },
    }
    const createOwnedTaskStore = () => ({
      subscribe: () => () => undefined, snapshot: () => snapshot,
      onEvent: vi.fn(), onConnectionChange: vi.fn(), replaceFromRest: vi.fn(), acknowledgeResync: vi.fn(),
      whenIdle: async () => undefined, dispose: vi.fn(),
    })

    render(<App backend={harness.client} createOwnedTaskStore={createOwnedTaskStore} initialBootstrap={bootstrap(current)} platform={harness.platform} />)

    expect(screen.getByText('REST 恢复渲染 15/30')).toBeVisible()
    expect(screen.getByText('15 / 30')).toBeVisible()
    expect(screen.getByText('已用时 10 秒')).toBeVisible()
    expect(screen.queryByText(/预计剩余/)).toBeNull()
    expect(screen.queryByText('过期事件')).toBeNull()
  })

  it('does not display an older matching event after a newer terminal REST snapshot', () => {
    const current = project()
    current.workflow.active_task_id = 'task-render'
    const harness = createHarness(current)
    const task = {
      id: 'task-render', target_stage: 'render' as const, status: 'cancelled' as const,
      revision: 9, error: null,
    }
    const snapshot = {
      task, revision: 9, pendingResyncRevision: null, connection: 'connected' as const,
      latestEvent: {
        type: 'task_event' as const, task_id: task.id, revision: 8,
        stage: 'render' as const, progress: 0.5, current: 15, total: 30,
        message: '过期的运行进度', elapsed_seconds: 10, eta_seconds: 10, error: null,
      },
    }
    const createOwnedTaskStore = () => ({
      subscribe: () => () => undefined, snapshot: () => snapshot,
      onEvent: vi.fn(), onConnectionChange: vi.fn(), replaceFromRest: vi.fn(), acknowledgeResync: vi.fn(),
      whenIdle: async () => undefined, dispose: vi.fn(),
    })

    render(<App backend={harness.client} createOwnedTaskStore={createOwnedTaskStore} initialBootstrap={bootstrap(current)} platform={harness.platform} />)

    expect(screen.getByText('render · cancelled')).toBeVisible()
    expect(screen.queryByText('过期的运行进度')).toBeNull()
    expect(screen.queryByRole('progressbar', { name: 'render 进度' })).toBeNull()
  })
})
