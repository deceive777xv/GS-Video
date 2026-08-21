import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { BackendClient } from '../../api/backend-client'
import { BackendClientError } from '../../api/http-backend-client'
import type { ProjectDto, StageStateDto } from '../../api/types'
import { SubjectPage } from './subject-page'

afterEach(() => vi.useRealTimers())

function readyProject(): ProjectDto {
  const stage = (status: 'pending' | 'succeeded'): StageStateDto => ({
    status, cache_key: status === 'succeeded' ? 'ingest-cache' : null,
    output_paths: [], error_code: null, artifacts: {},
  })
  return {
    schema_version: 3,
    project_id: 'project-1',
    name: 'Portrait import',
    created_at: '2026-08-01T00:00:00Z',
    source_video: 'opaque:source',
    scene_ply: 'opaque:scene',
    stages: {
      ingest: stage('succeeded'), segment: stage('pending'), solve_camera: stage('pending'),
      map_trajectory: stage('pending'), render: stage('pending'),
      composite: stage('pending'), export: stage('pending'),
    },
    workflow: {
      source_summary: {
        filename: 'portrait.mp4', size: 10, sha256: 'source', width: 480, height: 852,
        duration_seconds: 12, fps: '30', has_audio: false, frame_count: 360,
      },
      scene_summary: {
        filename: 'scene.ply', size: 20, sha256: 'scene', gaussian_count: 100,
        estimated_vram_mb: 128,
      },
      subject_prompt: null, target_camera: null, exploration_camera: null, preview_epoch: 0,
      target_ground: null,
      output_crop: null,
      gs_scale: 1, scene_azimuth: 0, preview_height: 540,
      active_task_id: null, preview: null, export_result: null,
    },
  }
}

describe('subject media readiness', () => {
  it('retries a transient not-ready response without requiring remount', async () => {
    vi.useFakeTimers()
    const notReady = new BackendClientError(409, {
      code: 'subject_media_not_ready', category: 'project',
      message: 'The requested subject media is not ready.', retryable: false,
    })
    const getSubjectMedia = vi.fn()
      .mockRejectedValueOnce(notReady)
      .mockResolvedValue({
        role: 'proxy', artifact_id: 'proxy-1', frame_index: 0,
        width: 480, height: 852, size: 8, mime_type: 'image/jpeg',
      })
    const backend = {
      getSubjectMedia,
      fetchSubjectMediaArtifact: vi.fn(async () => new Blob(['proxy'], { type: 'image/jpeg' })),
    } as unknown as BackendClient
    const onError = vi.fn()

    render(
      <SubjectPage
        backend={backend}
        busy={false}
        onError={onError}
        onProjectChange={vi.fn()}
        onStartStage={vi.fn()}
        project={readyProject()}
      />,
    )

    await act(async () => { await Promise.resolve() })
    expect(getSubjectMedia).toHaveBeenCalledTimes(1)
    expect(screen.getByText('载入代表帧…')).toBeVisible()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(250)
    })

    expect(getSubjectMedia).toHaveBeenCalledTimes(2)
    expect(screen.getByRole('img', { name: '人物代表帧' })).toBeVisible()
    expect(onError).not.toHaveBeenCalled()
  })

  it('retries a transient alpha response after segmentation without remount', async () => {
    vi.useFakeTimers()
    const project = readyProject()
    project.stages.segment = {
      status: 'succeeded', cache_key: 'segment-cache', output_paths: [],
      error_code: null, artifacts: {},
    }
    project.workflow.subject_prompt = { frame_index: 0, x: 120, y: 240 }
    const notReady = new BackendClientError(409, {
      code: 'subject_media_not_ready', category: 'project',
      message: 'The requested subject media is not ready.', retryable: false,
    })
    let alphaAttempts = 0
    const getSubjectMedia = vi.fn(async (role: 'proxy' | 'alpha') => {
      if (role === 'alpha' && alphaAttempts++ === 0) throw notReady
      return {
        role, artifact_id: `${role}-1`, frame_index: 0,
        width: 480, height: 852, size: 8,
        mime_type: role === 'proxy' ? 'image/jpeg' : 'image/png',
      }
    })
    const backend = {
      getSubjectMedia,
      fetchSubjectMediaArtifact: vi.fn(async (role: 'proxy' | 'alpha') => (
        new Blob([role], { type: role === 'proxy' ? 'image/jpeg' : 'image/png' })
      )),
    } as unknown as BackendClient
    const onError = vi.fn()

    render(
      <SubjectPage
        backend={backend}
        busy={false}
        onError={onError}
        onProjectChange={vi.fn()}
        onStartStage={vi.fn()}
        project={project}
      />,
    )

    await act(async () => { await Promise.resolve() })
    expect(getSubjectMedia).toHaveBeenCalledWith('alpha')
    expect(screen.queryByRole('img', { name: '人物 Alpha 叠加' })).not.toBeInTheDocument()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(250)
    })

    expect(getSubjectMedia.mock.calls.filter(([role]) => role === 'alpha')).toHaveLength(2)
    expect(screen.getByRole('img', { name: '人物 Alpha 叠加' })).toBeVisible()
    expect(onError).not.toHaveBeenCalled()
  })
})

describe('subject coordinate confirmation', () => {
  it('keeps a representative-frame click local until explicit segmentation confirmation', async () => {
    const project = readyProject()
    const updated = readyProject()
    updated.workflow.subject_prompt = { frame_index: 0, x: 120, y: 240 }
    const backend = {
      getSubjectMedia: vi.fn(async () => ({
        role: 'proxy', artifact_id: 'proxy-1', frame_index: 0,
        width: 480, height: 852, size: 8, mime_type: 'image/jpeg',
      })),
      fetchSubjectMediaArtifact: vi.fn(async () => new Blob(['proxy'], { type: 'image/jpeg' })),
      updateProject: vi.fn(async () => updated),
    } as unknown as BackendClient
    const onStartStage = vi.fn().mockResolvedValue(undefined)

    render(
      <SubjectPage
        backend={backend}
        busy={false}
        onError={vi.fn()}
        onProjectChange={vi.fn()}
        onStartStage={onStartStage}
        project={project}
      />,
    )

    await screen.findByRole('img', { name: '人物代表帧' })
    const frame = screen.getByLabelText('人物代表帧')
    vi.spyOn(frame, 'getBoundingClientRect').mockReturnValue({
      x: 0, y: 0, left: 0, top: 0, right: 482, bottom: 854,
      width: 482, height: 854, toJSON: () => ({}),
    })
    vi.spyOn(screen.getByRole('img', { name: '人物代表帧' }), 'getBoundingClientRect').mockReturnValue({
      x: 1, y: 1, left: 1, top: 1, right: 481, bottom: 853,
      width: 480, height: 852, toJSON: () => ({}),
    })

    fireEvent.click(frame, { clientX: 121, clientY: 241 })

    expect(screen.getByLabelText('人物 X 坐标')).toHaveValue(120)
    expect(screen.getByLabelText('人物 Y 坐标')).toHaveValue(240)
    expect(screen.getByText('X 120 · Y 240')).toBeInTheDocument()
    expect(backend.updateProject).not.toHaveBeenCalled()
    expect(onStartStage).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: '确认人物位置并开始分割' }))
    await waitFor(() => expect(onStartStage).toHaveBeenCalledWith('segment'))
    expect(backend.updateProject).toHaveBeenCalledWith({
      expected_project_id: 'project-1',
      expected_ingest_cache_key: 'ingest-cache',
      subject_prompt: { frame_index: 0, x: 120, y: 240 },
    })
  })

  it('shows exact ranges, restores a persisted point, and rejects invalid direct input', async () => {
    const project = readyProject()
    project.workflow.subject_prompt = { frame_index: 0, x: 100, y: 200 }
    const backend = {
      getSubjectMedia: vi.fn(async () => ({
        role: 'proxy', artifact_id: 'proxy-1', frame_index: 0,
        width: 480, height: 852, size: 8, mime_type: 'image/jpeg',
      })),
      fetchSubjectMediaArtifact: vi.fn(async () => new Blob(['proxy'], { type: 'image/jpeg' })),
    } as unknown as BackendClient

    render(
      <SubjectPage
        backend={backend}
        busy={false}
        onError={vi.fn()}
        onProjectChange={vi.fn()}
        onStartStage={vi.fn()}
        project={project}
      />,
    )

    const x = await screen.findByLabelText('人物 X 坐标')
    const y = screen.getByLabelText('人物 Y 坐标')
    await waitFor(() => expect(x).toHaveValue(100))
    expect(y).toHaveValue(200)
    expect(x).toHaveAttribute('min', '0')
    expect(x).toHaveAttribute('max', '479')
    expect(y).toHaveAttribute('max', '851')
    expect(screen.getByText('人物坐标已保存')).toBeVisible()

    fireEvent.change(x, { target: { value: '480' } })
    expect(screen.getByText(/坐标超出图像范围/)).toBeVisible()
    expect(screen.getByRole('button', { name: '确认人物位置并开始分割' })).toBeDisabled()
  })

  it('fails closed and clears an unconfirmed draft when the active project changes', async () => {
    let resolveSecondArtifact: ((value: Blob) => void) | undefined
    const descriptor = {
      role: 'proxy' as const, artifact_id: 'shared-proxy', frame_index: 0,
      width: 480, height: 852, size: 8, mime_type: 'image/jpeg',
    }
    const backend = {
      getSubjectMedia: vi.fn(async () => descriptor),
      fetchSubjectMediaArtifact: vi.fn()
        .mockResolvedValueOnce(new Blob(['project-a'], { type: 'image/jpeg' }))
        .mockImplementationOnce(() => new Promise<Blob>((resolve) => { resolveSecondArtifact = resolve })),
      updateProject: vi.fn(),
    } as unknown as BackendClient
    const props = {
      backend, busy: false, onError: vi.fn(), onProjectChange: vi.fn(), onStartStage: vi.fn(),
    }
    const projectA = readyProject()
    const view = render(<SubjectPage {...props} project={projectA} />)

    const imageA = await screen.findByRole('img', { name: '人物代表帧' })
    vi.spyOn(imageA, 'getBoundingClientRect').mockReturnValue({
      x: 0, y: 0, left: 0, top: 0, right: 480, bottom: 852,
      width: 480, height: 852, toJSON: () => ({}),
    })
    fireEvent.click(screen.getByLabelText('人物代表帧'), { clientX: 120, clientY: 240 })
    expect(screen.getByLabelText('人物 X 坐标')).toHaveValue(120)

    const projectB = readyProject()
    projectB.project_id = 'project-2'
    view.rerender(<SubjectPage {...props} project={projectB} />)

    expect(screen.getByText('载入代表帧…')).toBeVisible()
    expect(screen.getByLabelText('人物 X 坐标')).toBeDisabled()
    expect(screen.getByLabelText('人物 X 坐标')).toHaveValue(null)
    expect(screen.getByRole('button', { name: '确认人物位置并开始分割' })).toBeDisabled()
    expect(backend.updateProject).not.toHaveBeenCalled()

    await waitFor(() => expect(backend.fetchSubjectMediaArtifact).toHaveBeenCalledTimes(2))
    resolveSecondArtifact?.(new Blob(['project-b'], { type: 'image/jpeg' }))
    await screen.findByRole('img', { name: '人物代表帧' })
    expect(screen.getByLabelText('人物 X 坐标')).toHaveValue(null)
    expect(screen.getByLabelText('人物 Y 坐标')).toHaveValue(null)
  })

  it('does not publish or start segmentation when authority changes during submission', async () => {
    let resolveUpdate: ((value: ProjectDto) => void) | undefined
    const backend = {
      getSubjectMedia: vi.fn(async () => ({
        role: 'proxy', artifact_id: 'shared-proxy', frame_index: 0,
        width: 480, height: 852, size: 8, mime_type: 'image/jpeg',
      })),
      fetchSubjectMediaArtifact: vi.fn(async () => new Blob(['proxy'], { type: 'image/jpeg' })),
      updateProject: vi.fn(() => new Promise<ProjectDto>((resolve) => { resolveUpdate = resolve })),
    } as unknown as BackendClient
    const onProjectChange = vi.fn()
    const onStartStage = vi.fn()
    const props = { backend, busy: false, onError: vi.fn(), onProjectChange, onStartStage }
    const projectA = readyProject()
    const view = render(<SubjectPage {...props} project={projectA} />)

    await screen.findByRole('img', { name: '人物代表帧' })
    fireEvent.change(screen.getByLabelText('人物 X 坐标'), { target: { value: '120' } })
    fireEvent.change(screen.getByLabelText('人物 Y 坐标'), { target: { value: '240' } })
    fireEvent.click(screen.getByRole('button', { name: '确认人物位置并开始分割' }))
    await waitFor(() => expect(backend.updateProject).toHaveBeenCalledWith({
      expected_project_id: 'project-1',
      expected_ingest_cache_key: 'ingest-cache',
      subject_prompt: { frame_index: 0, x: 120, y: 240 },
    }))

    const projectB = readyProject()
    projectB.project_id = 'project-2'
    view.rerender(<SubjectPage {...props} project={projectB} />)
    await waitFor(() => expect(screen.getByRole('img', { name: '人物代表帧' })).toBeInTheDocument())
    resolveUpdate?.(projectA)
    await act(async () => { await Promise.resolve() })

    expect(onProjectChange).not.toHaveBeenCalled()
    expect(onStartStage).not.toHaveBeenCalled()
    expect(screen.getByLabelText('人物 X 坐标')).toHaveValue(null)
    expect(screen.getByLabelText('人物 Y 坐标')).toHaveValue(null)
  })

  it('does not carry an alpha overlay across project authority', async () => {
    const projectA = readyProject()
    projectA.workflow.subject_prompt = { frame_index: 0, x: 100, y: 200 }
    projectA.stages.segment = {
      status: 'succeeded', cache_key: 'segment-cache', output_paths: [], error_code: null, artifacts: {},
    }
    const backend = {
      getSubjectMedia: vi.fn(async (role: 'proxy' | 'alpha') => ({
        role, artifact_id: `${role}-shared`, frame_index: 0,
        width: 480, height: 852, size: 8, mime_type: 'image/png',
      })),
      fetchSubjectMediaArtifact: vi.fn(async (role: 'proxy' | 'alpha') => new Blob([role], { type: 'image/png' })),
    } as unknown as BackendClient
    const props = {
      backend, busy: false, onError: vi.fn(), onProjectChange: vi.fn(), onStartStage: vi.fn(),
    }
    const view = render(<SubjectPage {...props} project={projectA} />)
    expect(await screen.findByRole('img', { name: '人物 Alpha 叠加' })).toBeInTheDocument()

    const projectB = readyProject()
    projectB.project_id = 'project-2'
    view.rerender(<SubjectPage {...props} project={projectB} />)

    expect(screen.queryByRole('img', { name: '人物 Alpha 叠加' })).toBeNull()
  })
})
