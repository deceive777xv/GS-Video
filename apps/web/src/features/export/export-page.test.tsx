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
      preview_epoch: 0,
      confirmed_camera_revision: null,
      confirmed_preview_artifact_id: null,
      foot_point: null,
      motion_scale: 1,
      preview_height: 540,
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
    fetchExportArtifact: vi.fn(async () => new Blob(['video/mp4'])),
    copyVerifiedExport: vi.fn(async () => undefined),
  } as unknown as BackendClient
}

const successfulTask = (): TaskDto => ({
  id: 'task-export', target_stage: 'export', status: 'succeeded',
  revision: 7, error: null,
})

describe('ExportPage', () => {
  it('fetches and saves a browser Blob only after export task success', async () => {
    const backend = exportBackend()
    const platform: PlatformBridge = {
      kind: 'browser',
      pickInputFile: vi.fn(),
      saveExport: vi.fn(async () => undefined),
      openExternal: vi.fn(async () => undefined),
    }
    const onStartStage = vi.fn(async () => successfulTask())
    const user = userEvent.setup()
    render(
      <ExportPage
        activeTask={null}
        backend={backend}
        onError={vi.fn()}
        onStartStage={onStartStage}
        platform={platform}
        project={readyProject()}
      />,
    )
    await user.click(screen.getByRole('button', { name: '导出视频' }))

    await waitFor(() => expect(platform.saveExport).toHaveBeenCalledOnce())
    expect(backend.getVerifiedExport).toHaveBeenCalledAfter(onStartStage)
    expect(backend.fetchExportArtifact).toHaveBeenCalledWith('export-1')
    expect(vi.mocked(platform.saveExport).mock.calls[0]?.[1]).toMatchObject({ kind: 'browser-download' })
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
        onError={vi.fn()}
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
        onError={vi.fn()}
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
