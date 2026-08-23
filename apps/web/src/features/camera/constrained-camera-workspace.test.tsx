import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, expect, it, vi } from 'vitest'

import type { BackendClient } from '../../api/backend-client'
import type { PreviewFrameDto, ProjectDto, TargetGroundDto } from '../../api/types'
import { ConstrainedCameraWorkspace } from './constrained-camera-workspace'

afterEach(() => {
  vi.useRealTimers()
  vi.restoreAllMocks()
})

const identity = [
  [1, 0, 0, 0],
  [0, 1, 0, 0],
  [0, 0, 1, -4],
  [0, 0, 0, 1],
] as const

function project(
  targetGround: TargetGroundDto | null = null,
  preview: PreviewFrameDto | null = null,
  projectId = 'project-1',
): ProjectDto {
  return {
    project_id: projectId,
    scene_ply: 'opaque:scene',
    workflow: {
      exploration_camera: null,
      preview,
      target_ground: targetGround,
      gs_scale: 1,
      scene_azimuth: 0,
    },
  } as unknown as ProjectDto
}

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })
  return { promise, reject, resolve }
}

const restoredPreview: PreviewFrameDto = {
  artifact_id: 'preview-restored',
  generation: 4,
  width: 960,
  height: 540,
  camera_revision: 7,
  pick_buffer_revision: 9,
}

it('restores an existing authoritative GS preview when the scene page opens', async () => {
  vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:restored-preview')
  vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined)
  const initial = project(null, restoredPreview)
  const backend = {
    fetchPreviewArtifact: vi.fn(async () => new Blob(['png'], { type: 'image/png' })),
    renderLivePreview: vi.fn(() => new Promise<Blob>(() => undefined)),
    renderPreview: vi.fn(() => new Promise<PreviewFrameDto>(() => undefined)),
  } as unknown as BackendClient
  const props = {
    backend,
    busy: false,
    onError: vi.fn(),
    onProjectChange: vi.fn(),
    onRefresh: vi.fn(async () => initial),
    panel: 'scene' as const,
  }
  const view = render(<ConstrainedCameraWorkspace {...props} project={initial} />)

  expect(await screen.findByAltText('Gaussian 地面选择视图')).toHaveAttribute(
    'src',
    'blob:restored-preview',
  )
  expect(backend.fetchPreviewArtifact).toHaveBeenCalledWith(
    'preview-restored',
    expect.any(AbortSignal),
  )

  fireEvent.change(screen.getByLabelText('探索相机 X'), { target: { value: '1' } })
  expect(screen.getByAltText('Gaussian 地面选择视图')).toHaveAttribute(
    'src',
    'blob:restored-preview',
  )
  view.rerender(<ConstrainedCameraWorkspace {...props} onError={vi.fn()} project={initial} />)
  expect(backend.fetchPreviewArtifact).toHaveBeenCalledTimes(1)
  expect(screen.getByAltText('Gaussian 地面选择视图')).toBeInTheDocument()

  view.unmount()
  expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:restored-preview')
})

it('drops stale preview successes and errors across an A to B to A authority change', async () => {
  const firstA = deferred<Blob>()
  const staleB = deferred<Blob>()
  const latestA = deferred<Blob>()
  const latestBlob = new Blob(['latest'], { type: 'image/png' })
  const createObjectUrl = vi.spyOn(URL, 'createObjectURL').mockImplementation((blob) => (
    blob === latestBlob ? 'blob:latest-a' : 'blob:stale'
  ))
  vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined)
  const backend = {
    fetchPreviewArtifact: vi.fn()
      .mockReturnValueOnce(firstA.promise)
      .mockReturnValueOnce(staleB.promise)
      .mockReturnValueOnce(latestA.promise),
  } as unknown as BackendClient
  const onError = vi.fn()
  const firstProjectA = project(null, restoredPreview, 'project-a')
  const projectB = project(null, { ...restoredPreview, artifact_id: 'preview-b' }, 'project-b')
  const latestProjectA = project(null, restoredPreview, 'project-a')
  const props = {
    backend,
    busy: false,
    onError,
    onProjectChange: vi.fn(),
    onRefresh: vi.fn(async () => latestProjectA),
    panel: 'scene' as const,
  }
  const view = render(<ConstrainedCameraWorkspace {...props} project={firstProjectA} />)
  await waitFor(() => expect(backend.fetchPreviewArtifact).toHaveBeenCalledTimes(1))

  view.rerender(<ConstrainedCameraWorkspace {...props} project={projectB} />)
  await waitFor(() => expect(backend.fetchPreviewArtifact).toHaveBeenCalledTimes(2))
  view.rerender(<ConstrainedCameraWorkspace {...props} project={latestProjectA} />)
  await waitFor(() => expect(backend.fetchPreviewArtifact).toHaveBeenCalledTimes(3))

  await act(async () => {
    firstA.resolve(new Blob(['stale-a'], { type: 'image/png' }))
    staleB.reject(new Error('stale B failed'))
    await Promise.resolve()
  })
  expect(screen.queryByAltText('Gaussian 地面选择视图')).not.toBeInTheDocument()
  expect(onError).not.toHaveBeenCalled()

  await act(async () => {
    latestA.resolve(latestBlob)
    await Promise.resolve()
  })
  expect(await screen.findByAltText('Gaussian 地面选择视图')).toHaveAttribute(
    'src',
    'blob:latest-a',
  )
  expect(createObjectUrl).toHaveBeenCalledTimes(1)
  expect(onError).not.toHaveBeenCalled()
})

it('turns three viewport hints into one automatic ground candidate and one confirmation', async () => {
  vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:ground-view')
  vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined)
  const initial = project()
  const candidate: TargetGroundDto = {
    scene_asset_id: 'scene-1',
    hint_pixels: [[100, 100], [400, 100], [250, 300]],
    p0_world: [0, 0, 0],
    p1_world: [1, 0, 0],
    p2_world: [0, 0, 1],
    plane_normal: [0, 1, 0],
    plane_offset: 0,
    exploration_camera_to_world: identity.map((row) => [...row]) as TargetGroundDto['exploration_camera_to_world'],
    camera_fingerprint: 'f'.repeat(64),
    preview_artifact_id: 'preview-1',
    camera_revision: 1,
    pick_buffer_revision: 1,
    support_counts: [20, 21, 22],
    weighted_inlier_ratio: 0.9,
    rms_residual: 0.01,
    confidence: 0.92,
    revision: 1,
    confirmed: false,
  }
  const candidateProject = project(candidate)
  const confirmedProject = project({ ...candidate, confirmed: true })
  const backend = {
    renderLivePreview: vi.fn(async () => new Blob(['live'], { type: 'image/jpeg' })),
    renderPreview: vi.fn(async () => ({
      artifact_id: 'preview-1', generation: 1, width: 960, height: 540,
      camera_revision: 1, pick_buffer_revision: 1,
    })),
    fetchPreviewArtifact: vi.fn(async () => new Blob(['png'], { type: 'image/png' })),
    fitTargetGround: vi.fn(async () => candidateProject),
    confirmTargetGround: vi.fn(async () => confirmedProject),
    updateProject: vi.fn(),
  } as unknown as BackendClient
  const onProjectChange = vi.fn()
  const props = {
    backend,
    busy: false,
    onError: vi.fn(),
    onProjectChange,
    onRefresh: vi.fn(async () => initial),
    panel: 'scene' as const,
  }
  const user = userEvent.setup()
  const view = render(<ConstrainedCameraWorkspace {...props} project={initial} />)

  await waitFor(() => expect(backend.renderPreview).toHaveBeenCalledOnce())
  const viewport = await screen.findByLabelText('Gaussian 地面提示视口')
  vi.spyOn(viewport, 'getBoundingClientRect').mockReturnValue({
    x: 0, y: 0, left: 0, top: 0, right: 960, bottom: 540,
    width: 960, height: 540, toJSON: () => ({}),
  })
  fireEvent.pointerDown(viewport, { clientX: 100, clientY: 100 })
  fireEvent.pointerDown(viewport, { clientX: 400, clientY: 100 })
  fireEvent.pointerDown(viewport, { clientX: 250, clientY: 300 })

  await user.click(screen.getByRole('button', { name: '自动寻找真正地面' }))
  await waitFor(() => expect(backend.fitTargetGround).toHaveBeenCalledWith({
    expected_project_id: 'project-1',
    preview_artifact_id: 'preview-1',
    camera_revision: 1,
    pick_buffer_revision: 1,
    hints: [[100, 100], [400, 100], [250, 300]],
  }))

  view.rerender(<ConstrainedCameraWorkspace {...props} project={candidateProject} />)
  await user.click(screen.getByRole('button', { name: '确认地面' }))
  expect(backend.confirmTargetGround).toHaveBeenCalledWith('project-1', 1)
})

it('keeps the GS frame visible while pose edits request live and settled pick frames', async () => {
  vi.useFakeTimers()
  const initialBlob = new Blob(['initial'], { type: 'image/png' })
  const liveBlob = new Blob(['live'], { type: 'image/jpeg' })
  const settledBlob = new Blob(['settled'], { type: 'image/png' })
  vi.spyOn(URL, 'createObjectURL').mockImplementation((blob) => {
    if (blob === initialBlob) return 'blob:initial'
    if (blob === liveBlob) return 'blob:live'
    return 'blob:settled'
  })
  vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined)
  const initial = project(null, restoredPreview)
  const settled: PreviewFrameDto = {
    ...restoredPreview,
    artifact_id: 'preview-settled',
    generation: 5,
    camera_revision: 8,
    pick_buffer_revision: 10,
  }
  const refreshed = project(null, settled)
  const backend = {
    fetchPreviewArtifact: vi.fn(async (artifactId: string) => (
      artifactId === settled.artifact_id ? settledBlob : initialBlob
    )),
    renderLivePreview: vi.fn(async () => liveBlob),
    renderPreview: vi.fn(async () => settled),
  } as unknown as BackendClient
  const props = {
    backend,
    busy: false,
    onError: vi.fn(),
    onProjectChange: vi.fn(),
    onRefresh: vi.fn(async () => refreshed),
    panel: 'scene' as const,
  }
  render(<ConstrainedCameraWorkspace {...props} project={initial} />)
  await act(async () => { await Promise.resolve() })

  fireEvent.change(screen.getByLabelText('探索相机 X'), { target: { value: '1' } })
  expect(screen.getByAltText('Gaussian 地面选择视图')).toHaveAttribute('src', 'blob:initial')

  await act(async () => {
    await vi.advanceTimersByTimeAsync(200)
  })
  expect(backend.renderLivePreview).toHaveBeenCalled()
  expect(backend.renderPreview).toHaveBeenCalledWith(expect.objectContaining({
    expected_project_id: 'project-1',
    generation: 5,
  }), expect.any(AbortSignal))
  expect(screen.getByAltText('Gaussian 地面选择视图')).toHaveAttribute(
    'src',
    'blob:settled',
  )
})

it('uses middle-button dragging to roam the exploration camera', async () => {
  vi.useFakeTimers()
  vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:frame')
  vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined)
  const initial = project(null, restoredPreview)
  const backend = {
    fetchPreviewArtifact: vi.fn(async () => new Blob(['initial'], { type: 'image/png' })),
    renderLivePreview: vi.fn(async () => new Blob(['live'], { type: 'image/jpeg' })),
    renderPreview: vi.fn(() => new Promise<PreviewFrameDto>(() => undefined)),
  } as unknown as BackendClient
  render(
    <ConstrainedCameraWorkspace
      backend={backend}
      busy={false}
      onError={vi.fn()}
      onProjectChange={vi.fn()}
      onRefresh={vi.fn(async () => initial)}
      panel="scene"
      project={initial}
    />,
  )
  await act(async () => { await Promise.resolve() })
  const viewport = screen.getByLabelText('Gaussian 地面提示视口')

  fireEvent.pointerDown(viewport, {
    button: 1, buttons: 4, clientX: 100, clientY: 100, pointerId: 4,
  })
  fireEvent.pointerMove(viewport, {
    button: 1, buttons: 4, clientX: 140, clientY: 120, pointerId: 4,
  })
  fireEvent.pointerUp(viewport, {
    button: 1, buttons: 0, clientX: 140, clientY: 120, pointerId: 4,
  })
  await act(async () => { await vi.advanceTimersByTimeAsync(100) })

  expect(backend.renderLivePreview).toHaveBeenCalledWith(
    expect.objectContaining({
      expected_project_id: 'project-1',
      camera: expect.objectContaining({
        camera_to_world: expect.not.objectContaining({ 0: identity[0] }),
      }),
    }),
    expect.any(AbortSignal),
  )
})
