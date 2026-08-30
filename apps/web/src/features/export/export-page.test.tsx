import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import type { BackendClient } from '../../api/backend-client'
import type { ProjectDto, TaskDto, VerifiedExportDto } from '../../api/types'
import type { ExportSource, PlatformBridge } from '../../platform/platform-bridge'
import { ExportPage } from './export-page'

const descriptor: VerifiedExportDto = {
  artifact_id: 'export-1',
  filename: 'result.mp4',
  size: 4096,
  duration_seconds: 12,
  fps: '30/1',
  frame_count: 360,
  has_audio: true,
  verified: true,
}

function readyProject(exportReady = false): ProjectDto {
  return {
    schema_version: 3,
    project_id: 'project-1',
    name: 'Project',
    created_at: '2026-07-17T00:00:00Z',
    source_video: 'opaque:source',
    scene_ply: 'opaque:scene',
    stages: {
      composite: { status: 'succeeded', cache_key: 'composite', output_paths: [], error_code: null, artifacts: {} },
      post_process: { status: 'succeeded', cache_key: 'post-process', output_paths: [], error_code: null, artifacts: {} },
      export: {
        status: exportReady ? 'succeeded' : 'pending',
        cache_key: exportReady ? 'export' : null,
        output_paths: [], error_code: null, artifacts: {},
      },
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
      source_color_interpretation: 'rec709_metadata',
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
      export_result: exportReady ? {
        ...descriptor,
        sha256: 'a'.repeat(64),
      } : null,
    },
  }
}

function exportBackend(): BackendClient {
  return {
    getVerifiedExport: vi.fn(async () => descriptor),
    getProject: vi.fn(async () => readyProject(true)),
    fetchExportArtifact: vi.fn(async () => new Blob(['video/mp4'])),
    copyVerifiedExport: vi.fn(async () => undefined),
  } as unknown as BackendClient
}

const successfulTask = (): TaskDto => ({
  id: 'task-export', target_stage: 'export', status: 'succeeded',
  revision: 7, error: null,
})

describe('ExportPage', () => {
  it('prefetches a verified browser Blob but saves only from the explicit user action', async () => {
    const backend = exportBackend()
    const platform: PlatformBridge = {
      kind: 'browser',
      pickInputFile: vi.fn(),
      saveExport: vi.fn(async () => undefined),
      openExternal: vi.fn(async () => undefined),
    }
    const onStartStage = vi.fn(async () => successfulTask())
    const user = userEvent.setup()
    const view = render(
      <ExportPage
        activeTask={null}
        backend={backend}
        busy={false}
        onError={vi.fn()}
        onProjectChange={vi.fn()}
        onStartStage={onStartStage}
        platform={platform}
        project={readyProject()}
      />,
    )
    await user.click(screen.getByRole('button', { name: '导出视频' }))
    view.rerender(
      <ExportPage
        activeTask={successfulTask()}
        backend={backend}
        busy={false}
        onError={vi.fn()}
        onProjectChange={vi.fn()}
        onStartStage={onStartStage}
        platform={platform}
        project={readyProject(true)}
      />,
    )

    const save = await screen.findByRole('button', { name: '保存已验证视频' })
    expect(backend.getVerifiedExport).toHaveBeenCalledAfter(onStartStage)
    expect(backend.fetchExportArtifact).toHaveBeenCalledWith('export-1')
    expect(platform.saveExport).not.toHaveBeenCalled()
    await user.click(save)
    expect(platform.saveExport).toHaveBeenCalledOnce()
    expect(vi.mocked(platform.saveExport).mock.calls[0]?.[1]).toMatchObject({ kind: 'browser-download' })
  })

  it('rejects late export prefetches and clears save authority on invalidation', async () => {
    let resolveOld: ((value: VerifiedExportDto) => void) | undefined
    let resolveNew: ((value: VerifiedExportDto) => void) | undefined
    const newer = { ...descriptor, artifact_id: 'export-2', filename: 'result-2.mp4' }
    const backend = exportBackend()
    vi.mocked(backend.getVerifiedExport)
      .mockImplementationOnce(() => new Promise((resolve) => { resolveOld = resolve }))
      .mockImplementationOnce(() => new Promise((resolve) => { resolveNew = resolve }))
    vi.mocked(backend.fetchExportArtifact).mockImplementation(async (id) => new Blob([id]))
    const platform: PlatformBridge = {
      kind: 'browser', pickInputFile: vi.fn(), saveExport: vi.fn(async () => undefined),
      openExternal: vi.fn(async () => undefined),
    }
    const first = readyProject(true)
    const second = readyProject(true)
    if (second.workflow.export_result === null) throw new Error('export fixture missing')
    second.workflow.export_result = {
      ...second.workflow.export_result,
      artifact_id: newer.artifact_id,
      filename: newer.filename,
    }
    const props = {
      activeTask: null, backend, busy: false, onError: vi.fn(),
      onProjectChange: vi.fn(), onStartStage: vi.fn(async () => successfulTask()), platform,
    }
    const view = render(<ExportPage {...props} project={first} />)
    await waitFor(() => expect(backend.getVerifiedExport).toHaveBeenCalledTimes(1))
    view.rerender(<ExportPage {...props} project={second} />)
    await waitFor(() => expect(backend.getVerifiedExport).toHaveBeenCalledTimes(2))

    resolveNew?.(newer)
    expect(await screen.findByText('result-2.mp4')).toBeInTheDocument()
    resolveOld?.(descriptor)
    await Promise.resolve()
    expect(screen.getByText('result-2.mp4')).toBeInTheDocument()

    await userEvent.setup().click(screen.getByRole('button', { name: '保存已验证视频' }))
    expect(platform.saveExport).toHaveBeenCalledOnce()
    expect(backend.fetchExportArtifact).toHaveBeenCalledTimes(1)
    expect(backend.fetchExportArtifact).toHaveBeenCalledWith('export-2')

    view.rerender(<ExportPage {...props} project={readyProject(false)} />)
    await waitFor(() => expect(screen.queryByRole('button', { name: '保存已验证视频' })).toBeNull())
  })

  it('verifies a newly completed browser export before exposing explicit save', async () => {
    const backend = exportBackend()
    const platform: PlatformBridge = {
      kind: 'browser', pickInputFile: vi.fn(), saveExport: vi.fn(async () => undefined),
      openExternal: vi.fn(async () => undefined),
    }
    const running: TaskDto = {
      ...successfulTask(), status: 'running', revision: 6,
    }
    const props = {
      backend, busy: false, onError: vi.fn(),
      onStartStage: vi.fn(async () => successfulTask()), platform,
    }
    let view: ReturnType<typeof render>
    const onProjectChange = vi.fn((next: ProjectDto) => {
      view.rerender(
        <ExportPage {...props} activeTask={successfulTask()} onProjectChange={onProjectChange} project={next} />,
      )
    })
    view = render(
      <ExportPage {...props} activeTask={running} onProjectChange={onProjectChange} project={readyProject(false)} />,
    )

    view.rerender(
      <ExportPage {...props} activeTask={successfulTask()} onProjectChange={onProjectChange} project={readyProject(false)} />,
    )

    const save = await screen.findByRole('button', { name: '保存已验证视频' })
    expect(backend.getVerifiedExport).toHaveBeenCalled()
    expect(backend.getProject).toHaveBeenCalled()
    expect(backend.fetchExportArtifact).toHaveBeenCalledWith('export-1')
    expect(platform.saveExport).not.toHaveBeenCalled()
    await userEvent.setup().click(save)
    expect(platform.saveExport).toHaveBeenCalledOnce()
  })

  it('passes a verified opaque copy closure to the Tauri bridge', async () => {
    const backend = exportBackend()
    const sources: ExportSource[] = []
    const platform: PlatformBridge = {
      kind: 'tauri',
      pickInputFile: vi.fn(),
      saveExport: vi.fn(async (_name, value) => { sources.push(value) }),
      openExternal: vi.fn(async () => undefined),
    }
    const user = userEvent.setup()
    render(
      <ExportPage
        activeTask={null}
        backend={backend}
        busy={false}
        onError={vi.fn()}
        onProjectChange={vi.fn()}
        onStartStage={vi.fn(async () => successfulTask())}
        platform={platform}
        project={readyProject()}
      />,
    )
    await user.click(screen.getByRole('button', { name: '导出视频' }))
    await waitFor(() => expect(platform.saveExport).toHaveBeenCalledOnce())
    const source = sources[0]
    expect(source).toMatchObject({ kind: 'local-export', path: 'verified-export:export-1' })
    if (source?.kind !== 'local-export') throw new Error('local export source missing')
    await source.saveTo('E:\\Exports\\result.mp4')
    expect(backend.copyVerifiedExport).toHaveBeenCalledWith('export-1', 'E:\\Exports\\result.mp4')
  })

  it('recovers verified metadata on refresh without rerunning or opening a save dialog', async () => {
    const backend = exportBackend()
    const sources: ExportSource[] = []
    const platform: PlatformBridge = {
      kind: 'tauri',
      pickInputFile: vi.fn(),
      saveExport: vi.fn(async (_name, value) => { sources.push(value) }),
      openExternal: vi.fn(async () => undefined),
    }
    const onStartStage = vi.fn(async () => successfulTask())
    const user = userEvent.setup()
    render(
      <ExportPage
        activeTask={null}
        backend={backend}
        busy={false}
        onError={vi.fn()}
        onProjectChange={vi.fn()}
        onStartStage={onStartStage}
        platform={platform}
        project={readyProject(true)}
      />,
    )

    expect(await screen.findByText('result.mp4')).toBeInTheDocument()
    expect(onStartStage).not.toHaveBeenCalled()
    expect(platform.saveExport).not.toHaveBeenCalled()
    await user.click(screen.getByRole('button', { name: '保存已验证视频' }))
    expect(platform.saveExport).toHaveBeenCalledOnce()
    if (sources[0]?.kind !== 'local-export') throw new Error('local export source missing')
  })
})
