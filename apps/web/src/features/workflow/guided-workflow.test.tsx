import { StrictMode } from 'react'
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { BackendClient } from '../../api/backend-client'
import { BackendClientError } from '../../api/http-backend-client'
import type {
  BootstrapDto,
  AssetKind,
  AssetListItemDto,
  ProjectDto,
  Matrix4,
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
  window.history.replaceState(null, '', `${window.location.pathname}${window.location.search}`)
  sessionStorage.clear()
  vi.useRealTimers()
  vi.restoreAllMocks()
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
    schema_version: 8,
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
      post_process: stage(),
      export: stage(),
    },
    workflow: {
      source_summary: null,
      scene_summary: null,
      subject_prompt: null,
      target_camera: null,
      exploration_camera: null,
      preview_epoch: 0,
      target_ground: null,
      gs_scale: 1,
      scene_azimuth: 0,
      output_crop: null,
      preview_height: 540,
      source_color_interpretation: 'assumed_rec709',
      matte_refinement: {
        enabled: true, edge_offset: -1, feather_radius: 1,
        decontaminate_strength: 0, decontaminate_radius: 3,
      },
      effect_chain: [], effect_chain_revision: 0,
      export_settings: {
        codec: 'h264', rate_control: 'constant_quality', quality: 75,
        target_bitrate_mbps: 12, compression_preset: 'balanced',
      },
      active_task_id: null,
      preview: null,
      export_result: null,
    },
  }
}

function authorizeTargetGround(current: ProjectDto): void {
  const identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -4], [0, 0, 0, 1]] as Matrix4
  current.workflow.target_ground = {
    scene_asset_id: current.scene_ply ?? 'scene',
    hint_pixels: [[4, 7], [11, 7], [8, 5]],
    p0_world: [0, 0, 0], p1_world: [0, 0, 1], p2_world: [1, 0, 0],
    plane_normal: [0, -1, 0], plane_offset: 0,
    exploration_camera_to_world: identity,
    camera_fingerprint: 'f'.repeat(64), preview_artifact_id: 'preview-1',
    camera_revision: 1, pick_buffer_revision: 1,
    support_counts: [20, 20, 20], weighted_inlier_ratio: 0.9,
    rms_residual: 0.01, confidence: 0.9, revision: 1, confirmed: true,
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
    vram_budget: {
      mode: 'standard',
      minimum_vram_mb: 1024,
      total_vram_mb: 8192,
      selected_vram_mb: 8192,
      editable: true,
      blocked_reason: null,
      recovered_from_invalid_preference: false,
    },
  }
}

function libraryPly(): AssetListItemDto {
  return {
    asset: {
      asset_id: 'asset-ply', kind: 'ply', original_filename: 'library.ply',
      stored_relative_path: 'aa/library.ply', size: 2048,
      sha256: 'b'.repeat(64), imported_at: '2026-08-09T00:00:00Z',
      video_summary: null,
      scene_summary: {
        filename: 'library.ply', size: 2048, sha256: 'library-scene-sha',
        gaussian_count: 1_250_000, estimated_vram_mb: 3072,
      },
    },
    references: [],
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
    refreshEnvironment: vi.fn().mockResolvedValue(bootstrap(current).environment),
    getVramBudget: vi.fn().mockResolvedValue(bootstrap(current).vram_budget),
    updateVramBudget: vi.fn(async (input) => ({
      ...bootstrap(current).vram_budget,
      mode: input.mode,
      selected_vram_mb: input.mode === 'standard' ? 8192 : input.selected_vram_mb,
    })),
    getStorageLayout: vi.fn(),
    updateStorageLayout: vi.fn(),
    cleanupStorageCache: vi.fn(),
    planStorageCacheCleanup: vi.fn(),
    getEnvironmentRepair: vi.fn().mockResolvedValue({
      state: 'idle',
      job_id: null,
      step: null,
      resource_id: null,
      resource_name: null,
      progress: 0,
      downloaded_bytes: 0,
      total_bytes: null,
      message: null,
      resume_available: false,
      restart_required: false,
      error: null,
      environment: null,
    }),
    startEnvironmentRepair: vi.fn(),
    cancelEnvironmentRepair: vi.fn(),
    listProjects: vi.fn(),
    createProject: vi.fn(),
    activateProject: vi.fn(),
    renameProject: vi.fn(),
    deleteProject: vi.fn(),
    listAssets: vi.fn(async () => []),
    deleteAsset: vi.fn(),
    selectProjectAsset: vi.fn(async (kind, assetId) => {
      if (kind === 'source_video') {
        current.source_video_asset_id = assetId
        current.workflow.source_summary = assetId === null ? null : {
          filename: 'library.mp4', size: 1024, sha256: 'library-video-sha',
          width: 1920, height: 1080, duration_seconds: 12, fps: '30/1',
          has_audio: true, frame_count: 360,
        }
      } else {
        current.scene_ply_asset_id = assetId
        current.workflow.scene_summary = assetId === null ? null : {
          filename: 'library.ply', size: 2048, sha256: 'library-scene-sha',
          gaussian_count: 1_250_000, estimated_vram_mb: 3072,
        }
      }
      current.stages.ingest = stage()
      return structuredClone(current)
    }),
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
      if (patch.preview_height !== undefined) current.workflow.preview_height = patch.preview_height
      if (patch.source_color_interpretation !== undefined) current.workflow.source_color_interpretation = patch.source_color_interpretation
      if (patch.matte_refinement !== undefined) current.workflow.matte_refinement = patch.matte_refinement
      if (patch.effect_chain !== undefined) {
        current.workflow.effect_chain = structuredClone(patch.effect_chain)
        current.workflow.effect_chain_revision += 1
      }
      if (patch.export_settings !== undefined) current.workflow.export_settings = patch.export_settings
      return structuredClone(current)
    }),
    renderPreview: vi.fn(async (input) => {
      if ('camera_to_world' in input.camera) {
        current.workflow.exploration_camera = { ...input.camera, revision: input.generation }
        current.workflow.target_camera = null
      } else {
        current.workflow.target_camera = { ...input.camera, revision: input.generation }
        current.workflow.exploration_camera = null
      }
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
    renderLivePreview: vi.fn().mockResolvedValue(new Blob()),
    renderDraftCompositePreview: vi.fn().mockResolvedValue(new Blob()),
    renderDraftPostProcessPreview: vi.fn().mockResolvedValue(new Blob()),
    closePostProcessPreview: vi.fn().mockResolvedValue(undefined),
    closeLivePreview: vi.fn().mockResolvedValue(undefined),
    fetchPreviewArtifact: vi.fn(async () => new Blob(['preview'], { type: 'image/png' })),
    fitTargetGround: vi.fn(async () => structuredClone(current)),
    confirmTargetGround: vi.fn(async () => structuredClone(current)),
    getVerifiedExport: vi.fn(async () => ({
      artifact_id: 'export-1', filename: 'result.mp4', size: 12,
      duration_seconds: 12, fps: '30/1', frame_count: 360,
      has_audio: true, verified: true,
    })),
    fetchExportArtifact: vi.fn(async () => new Blob(['mp4'], { type: 'video/mp4' })),
    getCompositePreview: vi.fn(async () => ({
      artifact_id: 'composite-1', filename: 'post-process-preview.mp4' as const, size: 12,
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
  it('replaces an older request timeout with the authoritative camera material failure', async () => {
    const ready = project()
    ready.source_video = 'opaque:source'
    ready.scene_ply = 'opaque:scene'
    ready.workflow.source_summary = {
      filename: 'portrait.mp4', size: 10, sha256: 's', width: 1080, height: 1920,
      duration_seconds: 25, fps: '30/1', has_audio: false, frame_count: 758,
    }
    ready.workflow.scene_summary = {
      filename: 'garden.ply', size: 20, sha256: 'g', gaussian_count: 100,
      estimated_vram_mb: 128,
    }
    ready.workflow.subject_prompt = { frame_index: 0, x: 10, y: 10 }
    ready.stages.ingest = stage('succeeded')
    ready.stages.segment = stage('succeeded')
    const harness = createHarness(ready)
    vi.mocked(harness.client.startTask).mockRejectedValue(new BackendClientError(0, {
      code: 'request_timeout', category: 'network',
      message: 'The local service request timed out.', retryable: true,
    }))
    const failedTask: TaskDto = {
      id: 'task-solve_camera', target_stage: 'solve_camera', status: 'failed',
      revision: 7, error: 'unsupported_material',
    }
    vi.mocked(harness.client.getTask).mockResolvedValue(failedTask)
    let snapshot = {
      task: null as TaskDto | null,
      revision: 0,
      pendingResyncRevision: null as number | null,
      connection: 'connected' as const,
      latestEvent: null as import('../../api/types').TaskEvent | null,
    }
    const listeners = new Set<() => void>()
    const createOwnedTaskStore = () => ({
      subscribe: (listener: () => void) => {
        listeners.add(listener)
        return () => listeners.delete(listener)
      },
      snapshot: () => snapshot,
      onEvent: vi.fn(), onConnectionChange: vi.fn(), replaceFromRest: vi.fn(),
      acknowledgeResync: vi.fn(), whenIdle: async () => undefined, dispose: vi.fn(),
    })

    render(
      <App
        backend={harness.client}
        createOwnedTaskStore={createOwnedTaskStore}
        initialBootstrap={bootstrap(ready)}
        platform={harness.platform}
      />,
    )

    expect(await screen.findByRole('alert')).toHaveTextContent('The local service request timed out.')

    await act(async () => {
      snapshot = {
        ...snapshot,
        task: failedTask,
        revision: 7,
        latestEvent: {
          type: 'task_event', task_id: failedTask.id, revision: 7,
          stage: 'solve_camera', progress: 1,
          error: { code: 'unsupported_material', category: 'subject', retryable: false },
        },
      }
      listeners.forEach((listener) => listener())
    })

    expect(screen.getByRole('alert')).toHaveTextContent('当前视频无法生成可信的相机轨迹')
    expect(screen.getByRole('alert')).toHaveTextContent('unsupported_material')
    expect(screen.getByRole('alert')).not.toHaveTextContent('request timed out')
  })

  it('links each import input to its matching asset-library tab with a project return context', async () => {
    const harness = createHarness()

    render(<App backend={harness.client} platform={harness.platform} />)

    const videoCard = (await screen.findByRole('heading', { name: '单人短视频' })).closest('article')
    const sceneCard = screen.getByRole('heading', { name: '静态 Gaussian PLY' }).closest('article')
    expect(videoCard).not.toBeNull()
    expect(sceneCard).not.toBeNull()
    expect(within(videoCard!).getByRole('link', { name: '从素材库选择' }))
      .toHaveAttribute('href', '#/assets/video?returnProject=project-1')
    expect(within(sceneCard!).getByRole('link', { name: '从素材库选择' }))
      .toHaveAttribute('href', '#/assets/ply?returnProject=project-1')
  })

  it('returns from a contextual library selection and unlocks the next step after ingest succeeds', async () => {
    const current = project()
    current.source_video_asset_id = 'asset-video'
    current.workflow.source_summary = {
      filename: 'portrait.mp4', size: 1024, sha256: 'source-sha',
      width: 1920, height: 1080, duration_seconds: 12, fps: '30/1',
      has_audio: true, frame_count: 360,
    }
    const harness = createHarness(current)
    vi.mocked(harness.client.listAssets).mockResolvedValue([libraryPly()])
    window.location.hash = '#/assets/ply?returnProject=project-1'
    const user = userEvent.setup()

    render(<App backend={harness.client} initialBootstrap={bootstrap(current)} platform={harness.platform} startAtHome />)

    await user.click(await screen.findByRole('button', { name: '用于当前项目' }))
    expect(await screen.findByRole('heading', { name: '导入素材' })).toBeVisible()
    await waitFor(() => expect(screen.getByRole('button', { name: '下一步' })).toBeEnabled())
    expect(window.location.hash).toBe('#/projects/project-1/workflow/import')
  })

  it('returns after selecting the first input without starting ingest early', async () => {
    const current = project()
    const harness = createHarness(current)
    vi.mocked(harness.client.listAssets).mockResolvedValue([libraryPly()])
    window.location.hash = '#/assets/ply?returnProject=project-1'
    const user = userEvent.setup()

    render(<App backend={harness.client} initialBootstrap={bootstrap(current)} platform={harness.platform} startAtHome />)

    await user.click(await screen.findByRole('button', { name: '用于当前项目' }))
    expect(await screen.findByRole('heading', { name: '导入素材' })).toBeVisible()
    expect(screen.getByRole('button', { name: '下一步' })).toBeDisabled()
    expect(screen.getByText('准备就绪')).toBeVisible()
  })

  it.each([
    ['global library route', '#/assets/ply'],
    ['mismatched return project', '#/assets/ply?returnProject=project-2'],
  ])('keeps %s in the library while preparing complete inputs', async (_case, hash) => {
    const current = project()
    current.source_video_asset_id = 'asset-video'
    current.workflow.source_summary = {
      filename: 'portrait.mp4', size: 1024, sha256: 'source-sha',
      width: 1920, height: 1080, duration_seconds: 12, fps: '30/1',
      has_audio: true, frame_count: 360,
    }
    const harness = createHarness(current)
    vi.mocked(harness.client.listAssets).mockResolvedValue([libraryPly()])
    window.location.hash = hash
    const user = userEvent.setup()

    render(<App backend={harness.client} initialBootstrap={bootstrap(current)} platform={harness.platform} startAtHome />)

    await user.click(await screen.findByRole('button', { name: '用于当前项目' }))
    expect(await screen.findByRole('heading', { name: '素材库' })).toBeVisible()
    expect(window.location.hash).toBe(hash)

    window.location.hash = '#/projects/project-1/workflow/import'
    expect(await screen.findByRole('heading', { name: '导入素材' })).toBeVisible()
    await waitFor(() => expect(screen.getByRole('button', { name: '下一步' })).toBeEnabled())
  })

  it('keeps a successful asset selection when a concurrent project switch fails', async () => {
    const first = project()
    first.source_video_asset_id = 'asset-video'
    first.workflow.source_summary = {
      filename: 'portrait.mp4', size: 1024, sha256: 'source-sha',
      width: 1920, height: 1080, duration_seconds: 12, fps: '30/1',
      has_audio: true, frame_count: 360,
    }
    const second = project()
    second.project_id = 'project-2'
    second.name = 'Second project'
    const initial = bootstrap(first)
    initial.projects = [first, second].map((item) => ({
      project_id: item.project_id,
      name: item.name,
      created_at: item.created_at,
      updated_at: item.created_at,
      workflow_step: 'import',
      active_task_id: null,
    }))
    const harness = createHarness(first)
    vi.mocked(harness.client.listAssets).mockResolvedValue([libraryPly()])
    vi.mocked(harness.client.bootstrap).mockImplementation(
      () => new Promise<BootstrapDto>(() => undefined),
    )
    let rejectActivation: ((reason: Error) => void) | undefined
    vi.mocked(harness.client.activateProject).mockImplementation(() => (
      new Promise<ProjectDto>((_resolve, reject) => { rejectActivation = reject })
    ))
    let resolveSelection: ((value: ProjectDto) => void) | undefined
    vi.mocked(harness.client.selectProjectAsset).mockImplementation(() => (
      new Promise<ProjectDto>((resolve) => { resolveSelection = resolve })
    ))
    window.location.hash = '#/assets/ply?returnProject=project-1'
    const user = userEvent.setup()

    render(<App backend={harness.client} initialBootstrap={initial} platform={harness.platform} startAtHome />)

    await user.click(await screen.findByRole('button', { name: '用于当前项目' }))
    await user.click(screen.getByRole('link', { name: '返回项目首页' }))
    const secondCard = (await screen.findByRole('heading', { name: second.name })).closest('article')
    expect(secondCard).not.toBeNull()
    void user.click(within(secondCard!).getByRole('button', { name: '继续制作' }))
    await waitFor(() => expect(harness.client.activateProject).toHaveBeenCalledWith('project-2'))

    const stale = structuredClone(first)
    stale.scene_ply_asset_id = 'asset-ply'
    stale.workflow.scene_summary = libraryPly().asset.scene_summary
    await act(async () => { resolveSelection?.(stale) })

    await waitFor(() => {
      expect(harness.client.startTask).toHaveBeenCalledWith('ingest', first.project_id)
    })
    await act(async () => { rejectActivation?.(new Error('Target unavailable')) })
    expect(await screen.findByRole('alert')).toHaveTextContent('Target unavailable')
    expect(window.location.hash).toBe(`#/projects/${first.project_id}/workflow/import`)
  })

  it('opens a project-aware workflow URL and restores its requested step', async () => {
    const requested = project()
    requested.project_id = 'project-2'
    requested.name = 'Direct route project'
    requested.source_video = 'opaque:source'
    requested.scene_ply = 'opaque:scene'
    requested.workflow.source_summary = {
      filename: 'portrait.mp4', size: 10, sha256: 's', width: 640, height: 360,
      duration_seconds: 12, fps: '30/1', has_audio: true, frame_count: 360,
    }
    requested.workflow.scene_summary = {
      filename: 'garden.ply', size: 20, sha256: 'g', gaussian_count: 100,
      estimated_vram_mb: 128,
    }
    requested.workflow.subject_prompt = { frame_index: 0, x: 100, y: 120 }
    requested.stages.ingest = stage('succeeded')
    requested.stages.segment = stage('succeeded')
    requested.stages.solve_camera = stage('succeeded')
    const harness = createHarness()
    vi.mocked(harness.client.activateProject).mockResolvedValue(requested)
    window.location.hash = '#/projects/project-2/workflow/camera'

    render(
      <App
        backend={harness.client}
        initialBootstrap={bootstrap(project())}
        platform={harness.platform}
        startAtHome
      />,
    )

    expect(await screen.findByRole('heading', { name: '对齐自动相机轨迹与 GS 地面' })).toBeInTheDocument()
    expect(screen.queryByText(/GS 比例/)).not.toBeInTheDocument()
    expect(harness.client.activateProject).toHaveBeenCalledWith('project-2')
    expect(window.location.hash).toBe('#/projects/project-2/workflow/camera')
  })

  it('opens the current project without waiting for a bootstrap refresh', async () => {
    const current = project()
    const initial = bootstrap(current)
    initial.projects = [{
      project_id: current.project_id,
      name: current.name,
      created_at: current.created_at,
      updated_at: current.created_at,
      workflow_step: 'import',
      active_task_id: null,
    }]
    const harness = createHarness(current)
    vi.mocked(harness.client.bootstrap).mockImplementation(
      () => new Promise<BootstrapDto>(() => undefined),
    )
    window.location.hash = '#/'

    render(
      <App
        backend={harness.client}
        initialBootstrap={initial}
        platform={harness.platform}
        startAtHome
      />,
    )

    await userEvent.click(screen.getByRole('button', { name: '继续制作' }))

    await waitFor(() => {
      expect(window.location.hash).toBe('#/projects/project-1/workflow/import')
    }, { timeout: 250 })
    expect(await screen.findByRole('heading', { name: '导入素材' })).toBeInTheDocument()
    expect(harness.client.activateProject).not.toHaveBeenCalled()
  })

  it('refreshes only the environment report after a repair succeeds', async () => {
    const current = project()
    const harness = createHarness(current)
    vi.mocked(harness.client.getEnvironmentRepair).mockResolvedValue({
      state: 'succeeded',
      job_id: 'repair-1',
      step: null,
      resource_id: null,
      resource_name: null,
      progress: 1,
      downloaded_bytes: 0,
      total_bytes: null,
      message: null,
      resume_available: false,
      restart_required: false,
      error: null,
      environment: null,
    })

    render(
      <App
        backend={harness.client}
        initialBootstrap={bootstrap(current)}
        platform={harness.platform}
      />,
    )

    await waitFor(() => expect(harness.client.refreshEnvironment).toHaveBeenCalledOnce())
    expect(harness.client.bootstrap).not.toHaveBeenCalled()
  })

  it('waits only for target activation before opening another project', async () => {
    const current = project()
    const target = project()
    target.project_id = 'project-2'
    target.name = 'Second project'
    const initial = bootstrap(current)
    initial.projects = [current, target].map((item) => ({
      project_id: item.project_id,
      name: item.name,
      created_at: item.created_at,
      updated_at: item.created_at,
      workflow_step: 'import',
      active_task_id: null,
    }))
    const harness = createHarness(current)
    let resolveActivation: ((value: ProjectDto) => void) | undefined
    vi.mocked(harness.client.bootstrap).mockImplementation(
      () => new Promise<BootstrapDto>(() => undefined),
    )
    vi.mocked(harness.client.activateProject).mockImplementation(() => (
      new Promise<ProjectDto>((resolve) => { resolveActivation = resolve })
    ))
    window.location.hash = '#/'

    render(
      <App
        backend={harness.client}
        initialBootstrap={initial}
        platform={harness.platform}
        startAtHome
      />,
    )

    const targetCard = screen.getByRole('heading', { name: target.name }).closest('article')
    expect(targetCard).not.toBeNull()
    await userEvent.click(within(targetCard!).getByRole('button', { name: '继续制作' }))
    expect(window.location.hash).toBe('#/')

    await act(async () => { resolveActivation?.(target) })

    await waitFor(() => {
      expect(window.location.hash).toBe('#/projects/project-2/workflow/import')
    })
    expect(harness.client.activateProject).toHaveBeenCalledTimes(1)
    expect(harness.client.activateProject).toHaveBeenCalledWith('project-2')
  })

  it('stays on the project home when target activation fails', async () => {
    const current = project()
    const target = project()
    target.project_id = 'project-2'
    target.name = 'Unavailable project'
    const initial = bootstrap(current)
    initial.projects = [current, target].map((item) => ({
      project_id: item.project_id,
      name: item.name,
      created_at: item.created_at,
      updated_at: item.created_at,
      workflow_step: 'import',
      active_task_id: null,
    }))
    const harness = createHarness(current)
    vi.mocked(harness.client.bootstrap).mockImplementation(
      () => new Promise<BootstrapDto>(() => undefined),
    )
    vi.mocked(harness.client.activateProject).mockRejectedValue(new Error('Target unavailable'))
    window.location.hash = '#/'

    render(
      <App
        backend={harness.client}
        initialBootstrap={initial}
        platform={harness.platform}
        startAtHome
      />,
    )

    const targetCard = screen.getByRole('heading', { name: target.name }).closest('article')
    expect(targetCard).not.toBeNull()
    await userEvent.click(within(targetCard!).getByRole('button', { name: '继续制作' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('Target unavailable')
    expect(window.location.hash).toBe('#/')
    expect(screen.getByRole('heading', { name: '从一个项目继续，或开始新的合成。' })).toBeInTheDocument()
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
    gated.stages.ingest = stage('succeeded')
    gated.stages.segment = stage('succeeded')
    const harness = createHarness(gated)
    vi.mocked(harness.client.startTask).mockImplementation(async () => new Promise<TaskDto>(() => undefined))
    render(<App backend={harness.client} initialBootstrap={bootstrap(gated)} platform={harness.platform} />)

    expect(screen.getByRole('button', { name: /场景对齐/ })).toBeDisabled()
    await waitFor(() => expect(harness.client.startTask).toHaveBeenCalledWith('solve_camera', gated.project_id))
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

  it('rejects a gap event whose task owner differs from current project authority', async () => {
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
    await act(async () => { await Promise.resolve() })
    expect(harness.client.getTask).not.toHaveBeenCalledWith('task-new')

    resolveOld?.({
      id: 'task-old', target_stage: 'segment', status: 'running', revision: 1, error: null,
    })
    expect(await screen.findByText(/segment · running/)).toBeInTheDocument()
    expect(screen.queryByText(/render · running/)).toBeNull()
  })

  it('establishes initial bootstrap task ownership before subscribing to events', async () => {
    const active = project()
    active.workflow.active_task_id = 'task-current'
    const harness = createHarness(active)
    vi.mocked(harness.client.getTask).mockImplementation(
      () => new Promise(() => undefined),
    )
    let subscription: Parameters<NonNullable<Parameters<typeof App>[0]['eventSource']>['subscribe']>[0] | undefined
    const eventSource = {
      subscribe: vi.fn((next: NonNullable<typeof subscription>) => {
        subscription = next
        return () => undefined
      }),
    }

    render(<App backend={harness.client} eventSource={eventSource} initialBootstrap={bootstrap(active)} platform={harness.platform} />)
    await waitFor(() => expect(harness.client.getTask).toHaveBeenCalledWith('task-current'))
    act(() => subscription?.onEvent({
      type: 'task_event', task_id: 'task-from-another-project', revision: 99,
      stage: 'render', progress: 0.5, error: null,
    }))
    await act(async () => { await Promise.resolve() })

    expect(harness.client.getTask).not.toHaveBeenCalledWith('task-from-another-project')
    expect(screen.queryByText(/render · running/)).toBeNull()
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
    ready.stages.ingest = stage('succeeded')
    ready.stages.segment = stage('succeeded')
    ready.stages.solve_camera = stage('succeeded')
    ready.stages.composite = stage('succeeded')
    ready.stages.post_process = stage('succeeded')
    ready.workflow.target_camera = { target: [0, 0, 0], distance: 4, yaw: 0, pitch: 0, fov_y_degrees: 50, revision: 1 }
    ready.workflow.preview = { artifact_id: 'preview-1', artifact_size: 1, artifact_sha256: 'p', generation: 1, width: 960, height: 540, camera_revision: 1, pick_buffer_revision: 1 }
    authorizeTargetGround(ready)
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
    expect(await screen.findByRole('button', { name: '确认人物位置并开始分割' })).toBeDisabled()
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
    ready.stages.ingest = stage('succeeded')
    ready.stages.segment = stage('succeeded')
    ready.stages.solve_camera = stage('succeeded')
    ready.workflow.target_camera = { target: [0, 0, 0], distance: 4, yaw: 0, pitch: 0, fov_y_degrees: 50, revision: 1 }
    authorizeTargetGround(ready)
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
        eta_seconds: 12.3456,
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
    expect(screen.getByText('已用时 8.000 秒')).toBeVisible()
    expect(screen.getByText('预计剩余 12.346 秒')).toBeVisible()
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
    expect(screen.getByText('已用时 10.000 秒')).toBeVisible()
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
