import { StrictMode } from 'react'
import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

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
      onEvent: vi.fn(), onConnectionChange: vi.fn(), replaceFromRest: vi.fn(),
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

  it('cancels an active task and retries a failed task through backend authority', async () => {
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
    const retry = await screen.findByRole('button', { name: '重试阶段' })
    await user.click(retry)
    expect(harness.client.startTask).toHaveBeenCalledWith('composite')
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

  it('keeps a browser upload session for retry and focuses the structured error alert', async () => {
    const user = userEvent.setup()
    const harness = createHarness()
    vi.mocked(harness.client.putUploadChunk)
      .mockRejectedValueOnce(new Error('chunk interrupted'))
      .mockResolvedValue(undefined)
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
})
