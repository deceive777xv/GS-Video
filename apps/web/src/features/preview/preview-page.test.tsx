import { render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import type { BackendClient } from '../../api/backend-client'
import type { ProjectDto, TaskDto } from '../../api/types'
import { PreviewPage } from './preview-page'

afterEach(() => vi.restoreAllMocks())

it('does not admit a second composite task while the authoritative owner is active', () => {
  const project = {
    stages: { composite: { status: 'running' } },
    workflow: {
      motion_scale: 1,
      preview: null,
      target_camera: null,
      foot_point: {
        image: [1, 1], world: [0, 0, 0], preview_artifact_id: 'preview',
        camera_revision: 1, pick_buffer_revision: 1,
      },
    },
  } as unknown as ProjectDto
  const task: TaskDto = {
    id: 'task-composite', target_stage: 'composite', status: 'running',
    revision: 1, error: null,
  }
  const onStartStage = vi.fn()
  render(
    <PreviewPage
      activeTask={task}
      backend={{} as BackendClient}
      busy
      latestEvent={null}
      onBackToCamera={vi.fn()}
      onError={vi.fn()}
      onProjectChange={vi.fn()}
      onReselectSubject={vi.fn()}
      onStartStage={onStartStage}
      project={project}
    />,
  )

  expect(screen.getByRole('button', { name: '生成中…' })).toBeDisabled()
  expect(onStartStage).not.toHaveBeenCalled()
})

const failedProject = (target: TaskDto['target_stage'] = 'composite') => ({
  stages: {
    composite: target === 'composite'
      ? { status: 'failed', cache_key: null, output_paths: [], error_code: 'composite_failed', artifacts: {} }
      : { status: 'pending', cache_key: null, output_paths: [], error_code: null, artifacts: {} },
    [target]: { status: 'failed', cache_key: null, output_paths: [], error_code: `${target}_failed`, artifacts: {} },
  },
  workflow: {
    motion_scale: 1, preview: null, target_camera: null,
    foot_point: { image: [1, 1], world: [0, 0, 0], preview_artifact_id: 'preview', camera_revision: 1, pick_buffer_revision: 1 },
  },
}) as unknown as ProjectDto

it.each([
  ['segment', 'subject_mask_invalid', '重新选择人物', ['降低预览分辨率', '返回机位']],
  ['render', 'gpu_out_of_memory', '降低预览分辨率', ['重新选择人物', '返回机位']],
  ['solve_camera', 'camera_authority_stale', '返回机位', ['重新选择人物', '降低预览分辨率']],
] as const)('shows only the targeted %s recovery action', (target, code, expected, absent) => {
  const task: TaskDto = {
    id: `task-${target}`, target_stage: target, status: 'failed', revision: 3, error: code,
  }
  render(
    <PreviewPage
      activeTask={task}
      backend={{} as BackendClient}
      busy={false}
      latestEvent={{
        type: 'task_event', task_id: task.id, revision: 3, stage: target,
        progress: 1, error: { code, category: 'task', retryable: false },
      }}
      onBackToCamera={vi.fn()}
      onError={vi.fn()}
      onProjectChange={vi.fn()}
      onReselectSubject={vi.fn()}
      onStartStage={vi.fn()}
      project={failedProject(target)}
    />,
  )
  expect(screen.getByRole('button', { name: expected })).toBeInTheDocument()
  for (const name of absent) expect(screen.queryByRole('button', { name })).toBeNull()
  expect(screen.queryByRole('button', { name: '重试' })).toBeNull()
})

it('offers retry only for a failed task whose event says it is retryable', () => {
  const task: TaskDto = {
    id: 'task-composite', target_stage: 'composite', status: 'failed', revision: 3,
    error: 'transient_render_failure',
  }
  render(
    <PreviewPage
      activeTask={task}
      backend={{} as BackendClient}
      busy={false}
      latestEvent={{
        type: 'task_event', task_id: task.id, revision: 3, stage: 'composite',
        progress: 1, error: { code: task.error, category: 'resource', retryable: true },
      }}
      onBackToCamera={vi.fn()}
      onError={vi.fn()}
      onProjectChange={vi.fn()}
      onReselectSubject={vi.fn()}
      onStartStage={vi.fn()}
      project={failedProject()}
    />,
  )
  expect(screen.getByRole('button', { name: '重试' })).toBeInTheDocument()
})

it('revokes and removes a displayed frame when preview authority becomes null', async () => {
  const createUrl = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:preview')
  const revokeUrl = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined)
  const backend = {
    fetchPreviewArtifact: vi.fn(async () => new Blob(['preview'])),
  } as unknown as BackendClient
  const initial = failedProject()
  initial.workflow.preview = {
    artifact_id: 'preview-1', generation: 1, width: 640, height: 360,
    camera_revision: 1, pick_buffer_revision: 1,
    artifact_size: 7, artifact_sha256: 'a'.repeat(64),
  }
  const props = {
    activeTask: null,
    backend,
    busy: false,
    latestEvent: null,
    onBackToCamera: vi.fn(),
    onError: vi.fn(),
    onProjectChange: vi.fn(),
    onReselectSubject: vi.fn(),
    onStartStage: vi.fn(),
  }
  const view = render(<PreviewPage {...props} project={initial} />)
  expect(await screen.findByRole('img', { name: '相机参考帧（非合成视频）' })).toHaveAttribute('src', 'blob:preview')

  const revoked = failedProject()
  revoked.workflow.preview = null
  view.rerender(<PreviewPage {...props} project={revoked} />)

  await waitFor(() => expect(screen.queryByRole('img', { name: '相机参考帧（非合成视频）' })).toBeNull())
  expect(createUrl).toHaveBeenCalledOnce()
  expect(revokeUrl).toHaveBeenCalledWith('blob:preview')
})

const compositeDescriptor = (artifactId: string) => ({
  artifact_id: artifactId,
  filename: 'composite-preview.mp4' as const,
  size: 12,
  sha256: artifactId.padEnd(64, 'a'),
  duration_seconds: 2,
  fps: '24/1',
  frame_count: 48,
})

function compositeProject(cacheKey: string | null, status: 'pending' | 'succeeded') {
  const current = failedProject()
  current.stages.composite = {
    status,
    cache_key: cacheKey,
    output_paths: [],
    error_code: null,
    artifacts: {},
  }
  current.workflow.preview = {
    artifact_id: 'camera-preview', generation: 1, width: 640, height: 360,
    camera_revision: 1, pick_buffer_revision: 1,
    artifact_size: 7, artifact_sha256: 'b'.repeat(64),
  }
  return current
}

function compositeProps(backend: BackendClient, project: ProjectDto) {
  return {
    activeTask: null,
    backend,
    busy: false,
    latestEvent: null,
    onBackToCamera: vi.fn(),
    onError: vi.fn(),
    onProjectChange: vi.fn(),
    onReselectSubject: vi.fn(),
    onStartStage: vi.fn(),
    project,
  }
}

it('shows only the composite video registered by the succeeded stage', async () => {
  vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:composite-preview')
  vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined)
  const descriptor = compositeDescriptor('composite-1')
  const backend = {
    getCompositePreview: vi.fn(async () => descriptor),
    fetchCompositePreviewArtifact: vi.fn(async () => new Blob(['mp4'], { type: 'video/mp4' })),
    fetchPreviewArtifact: vi.fn(async () => new Blob(['camera'], { type: 'image/png' })),
  } as unknown as BackendClient

  render(<PreviewPage {...compositeProps(backend, compositeProject('cache-1', 'succeeded'))} />)

  const video = await screen.findByLabelText('低分辨率合成预览')
  expect(video).toHaveAttribute('controls')
  expect(video).toHaveAttribute('playsinline')
  expect(video).toHaveAttribute('preload', 'metadata')
  expect(video).toHaveAttribute('src', 'blob:composite-preview')
  expect(screen.queryByRole('img')).toBeNull()
  expect(backend.getCompositePreview).toHaveBeenCalledWith(
    expect.any(AbortSignal),
  )
  expect(backend.fetchCompositePreviewArtifact).toHaveBeenCalledWith(
    descriptor.artifact_id,
    expect.any(AbortSignal),
  )
})

it('aborts a stale composite descriptor before requesting its artifact', async () => {
  vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:composite-2')
  vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined)
  const first = compositeDescriptor('composite-1')
  const second = compositeDescriptor('composite-2')
  let firstSignal: AbortSignal | undefined
  let resolveFirst!: (descriptor: ReturnType<typeof compositeDescriptor>) => void
  const getCompositePreview = vi.fn((signal?: AbortSignal) => {
    if (getCompositePreview.mock.calls.length === 1) {
      firstSignal = signal
      return new Promise<ReturnType<typeof compositeDescriptor>>((resolve) => {
        resolveFirst = resolve
      })
    }
    return Promise.resolve(second)
  })
  const backend = {
    getCompositePreview,
    fetchCompositePreviewArtifact: vi.fn(async () => new Blob(['new'], { type: 'video/mp4' })),
    fetchPreviewArtifact: vi.fn(async () => new Blob(['camera'], { type: 'image/png' })),
  } as unknown as BackendClient
  const initialProps = compositeProps(backend, compositeProject('cache-1', 'succeeded'))
  const view = render(<PreviewPage {...initialProps} />)
  await waitFor(() => expect(getCompositePreview).toHaveBeenCalledOnce())

  view.rerender(
    <PreviewPage {...initialProps} project={compositeProject('cache-2', 'succeeded')} />,
  )

  expect(await screen.findByLabelText('低分辨率合成预览')).toHaveAttribute(
    'src',
    'blob:composite-2',
  )
  expect(firstSignal?.aborted).toBe(true)
  expect(backend.fetchCompositePreviewArtifact).toHaveBeenCalledOnce()
  expect(backend.fetchCompositePreviewArtifact).toHaveBeenCalledWith(
    second.artifact_id,
    expect.any(AbortSignal),
  )

  resolveFirst(first)
  await Promise.resolve()
  expect(backend.fetchCompositePreviewArtifact).not.toHaveBeenCalledWith(
    first.artifact_id,
    expect.anything(),
  )
})

it('labels the static Gaussian frame as a camera reference only before composite success', async () => {
  vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:camera-reference')
  vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined)
  const backend = {
    fetchPreviewArtifact: vi.fn(async () => new Blob(['camera'], { type: 'image/png' })),
    getCompositePreview: vi.fn(),
    fetchCompositePreviewArtifact: vi.fn(),
  } as unknown as BackendClient

  render(<PreviewPage {...compositeProps(backend, compositeProject(null, 'pending'))} />)

  expect(await screen.findByRole('img', { name: '相机参考帧（非合成视频）' })).toBeVisible()
  expect(screen.getByText('相机参考 · 非合成视频')).toBeVisible()
  expect(screen.queryByLabelText('低分辨率合成预览')).toBeNull()
  expect(backend.getCompositePreview).not.toHaveBeenCalled()
})

it('aborts stale composite fetches and publishes only the latest descriptor authority', async () => {
  const createUrl = vi.spyOn(URL, 'createObjectURL')
    .mockReturnValueOnce('blob:composite-2')
  const revokeUrl = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined)
  const first = compositeDescriptor('composite-1')
  const second = compositeDescriptor('composite-2')
  let resolveFirst!: (blob: Blob) => void
  let firstSignal: AbortSignal | undefined
  const backend = {
    getCompositePreview: vi.fn()
      .mockResolvedValueOnce(first)
      .mockResolvedValueOnce(second),
    fetchCompositePreviewArtifact: vi.fn((id: string, signal?: AbortSignal) => {
      if (id === first.artifact_id) {
        firstSignal = signal
        return new Promise<Blob>((resolve) => { resolveFirst = resolve })
      }
      return Promise.resolve(new Blob(['new'], { type: 'video/mp4' }))
    }),
    fetchPreviewArtifact: vi.fn(async () => new Blob(['camera'], { type: 'image/png' })),
  } as unknown as BackendClient
  const firstProps = compositeProps(backend, compositeProject('cache-1', 'succeeded'))
  const view = render(<PreviewPage {...firstProps} />)
  await waitFor(() => expect(backend.fetchCompositePreviewArtifact).toHaveBeenCalledWith(
    first.artifact_id,
    expect.any(AbortSignal),
  ))

  view.rerender(
    <PreviewPage {...firstProps} project={compositeProject('cache-2', 'succeeded')} />,
  )
  expect(await screen.findByLabelText('低分辨率合成预览')).toHaveAttribute(
    'src',
    'blob:composite-2',
  )
  resolveFirst(new Blob(['old'], { type: 'video/mp4' }))
  await Promise.resolve()

  expect(firstSignal?.aborted).toBe(true)
  expect(createUrl).toHaveBeenCalledOnce()
  expect(screen.getByLabelText('低分辨率合成预览')).toHaveAttribute(
    'src',
    'blob:composite-2',
  )
  view.unmount()
  expect(revokeUrl).toHaveBeenCalledWith('blob:composite-2')
})

it('revokes the composite URL when authority becomes null or replacement loading fails', async () => {
  vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:composite-1')
  const revokeUrl = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined)
  const backend = {
    getCompositePreview: vi.fn()
      .mockResolvedValueOnce(compositeDescriptor('composite-1'))
      .mockRejectedValueOnce(new Error('descriptor unavailable')),
    fetchCompositePreviewArtifact: vi.fn(async () => new Blob(['first'], { type: 'video/mp4' })),
    fetchPreviewArtifact: vi.fn(async () => new Blob(['camera'], { type: 'image/png' })),
  } as unknown as BackendClient
  const initialProps = compositeProps(backend, compositeProject('cache-1', 'succeeded'))
  const view = render(<PreviewPage {...initialProps} />)
  expect(await screen.findByLabelText('低分辨率合成预览')).toBeVisible()

  view.rerender(
    <PreviewPage {...initialProps} project={compositeProject('cache-2', 'succeeded')} />,
  )
  await waitFor(() => expect(initialProps.onError).toHaveBeenCalledWith(
    expect.objectContaining({ message: 'descriptor unavailable' }),
  ))
  expect(screen.queryByLabelText('低分辨率合成预览')).toBeNull()
  expect(revokeUrl).toHaveBeenCalledWith('blob:composite-1')

  view.rerender(
    <PreviewPage {...initialProps} project={compositeProject(null, 'pending')} />,
  )
  expect(screen.queryByLabelText('低分辨率合成预览')).toBeNull()
})
