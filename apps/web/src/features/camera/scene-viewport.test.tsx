import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import type { BackendClient } from '../../api/backend-client'
import type { PreviewFrameDto, PreviewRequest } from '../../api/types'
import { SceneViewport, toImagePoint } from './scene-viewport'

describe('toImagePoint', () => {
  it('maps letterboxed CSS coordinates to integer image pixels', () => {
    expect(toImagePoint(
      { clientX: 250, clientY: 150 },
      { left: 0, top: 0, width: 500, height: 300 },
      { width: 400, height: 400 },
    )).toEqual({ x: 200, y: 200 })
  })

  it('rejects a click in the letterbox bars', () => {
    expect(toImagePoint(
      { clientX: 50, clientY: 150 },
      { left: 0, top: 0, width: 500, height: 300 },
      { width: 400, height: 400 },
    )).toBeNull()
  })
})

describe('SceneViewport', () => {
  it('invalidates confirm and pick immediately when the camera no longer matches the frame', async () => {
    vi.useFakeTimers()
    const backend = {
      fetchPreviewArtifact: vi.fn(async () => new Blob(['preview'])),
      renderPreview: vi.fn(() => new Promise(() => undefined)),
      confirmCamera: vi.fn(),
      pickFootPoint: vi.fn(),
    } as unknown as BackendClient
    render(
      <SceneViewport
        backend={backend}
        camera={{ target: [0, 0, 0], distance: 4, yaw: 0, pitch: 0, fov_y_degrees: 50 }}
        initialPreview={{ artifact_id: 'preview-1', artifact_size: 1, artifact_sha256: 'a', generation: 1, width: 960, height: 540, camera_revision: 1, pick_buffer_revision: 1 }}
        onError={vi.fn()}
        onPreview={vi.fn()}
      />,
    )
    await act(async () => { await Promise.resolve() })
    expect(screen.getByRole('button', { name: '确认初始机位' })).toBeEnabled()
    expect(screen.getByRole('button', { name: '确认场景落脚点' })).toBeEnabled()

    fireEvent.wheel(screen.getByLabelText('Gaussian 场景视口'), { deltaY: 100 })
    expect(screen.getByRole('button', { name: '确认初始机位' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '确认场景落脚点' })).toBeDisabled()
    fireEvent.click(screen.getByRole('button', { name: '确认初始机位' }))
    fireEvent.click(screen.getByRole('button', { name: '确认场景落脚点' }))
    expect(backend.confirmCamera).not.toHaveBeenCalled()
    expect(backend.pickFootPoint).not.toHaveBeenCalled()
    vi.useRealTimers()
  })

  it('suppresses picking after cumulative sub-threshold drag moves', async () => {
    const backend = {
      fetchPreviewArtifact: vi.fn(async () => new Blob(['preview'])),
      pickFootPoint: vi.fn(),
    } as unknown as BackendClient
    render(
      <SceneViewport
        backend={backend}
        camera={{ target: [0, 0, 0], distance: 4, yaw: 0, pitch: 0, fov_y_degrees: 50 }}
        initialPreview={{ artifact_id: 'preview-1', artifact_size: 1, artifact_sha256: 'a', generation: 1, width: 960, height: 540, camera_revision: 1, pick_buffer_revision: 1 }}
        onError={vi.fn()}
        onPreview={vi.fn()}
      />,
    )
    const viewport = screen.getByLabelText('Gaussian 场景视口')
    vi.spyOn(viewport, 'getBoundingClientRect').mockReturnValue({
      x: 0, y: 0, left: 0, top: 0, right: 960, bottom: 540,
      width: 960, height: 540, toJSON: () => ({}),
    })
    fireEvent.pointerDown(viewport, { clientX: 100, clientY: 100, pointerId: 1 })
    fireEvent.pointerMove(viewport, { clientX: 102, clientY: 100, pointerId: 1 })
    fireEvent.pointerMove(viewport, { clientX: 104, clientY: 100, pointerId: 1 })
    fireEvent.pointerUp(viewport, { clientX: 104, clientY: 100, pointerId: 1 })
    fireEvent.click(viewport, { clientX: 104, clientY: 100 })
    expect(backend.pickFootPoint).not.toHaveBeenCalled()
  })

  it('clears and revokes the frame when authoritative preview becomes null', async () => {
    const createUrl = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:preview')
    const revokeUrl = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined)
    const backend = { fetchPreviewArtifact: vi.fn(async () => new Blob(['preview'])) } as unknown as BackendClient
    const props = {
      backend,
      camera: { target: [0, 0, 0] as [number, number, number], distance: 4, yaw: 0, pitch: 0, fov_y_degrees: 50 },
      onError: vi.fn(),
      onPreview: vi.fn(),
    }
    const view = render(
      <SceneViewport {...props} initialPreview={{ artifact_id: 'preview-1', artifact_size: 1, artifact_sha256: 'a', generation: 1, width: 960, height: 540, camera_revision: 1, pick_buffer_revision: 1 }} />,
    )
    await waitFor(() => expect(createUrl).toHaveBeenCalledOnce())
    expect(screen.getByAltText('最新 Gaussian 场景后端预览')).toBeInTheDocument()

    view.rerender(<SceneViewport {...props} initialPreview={null} />)
    expect(revokeUrl).toHaveBeenCalledWith('blob:preview')
    expect(screen.queryByAltText('最新 Gaussian 场景后端预览')).toBeNull()
    expect(screen.getByRole('button', { name: '确认初始机位' })).toBeDisabled()
  })

  it('does not turn a camera drag into a foot-point pick', async () => {
    const backend = {
      fetchPreviewArtifact: vi.fn(async () => new Blob(['preview'])),
      pickFootPoint: vi.fn(),
    } as unknown as BackendClient
    render(
      <SceneViewport
        backend={backend}
        camera={{ target: [0, 0, 0], distance: 4, yaw: 0, pitch: 0, fov_y_degrees: 50 }}
        initialPreview={{
          artifact_id: 'preview-4', artifact_size: 8, artifact_sha256: 'sha',
          generation: 4, width: 960, height: 540, camera_revision: 4,
          pick_buffer_revision: 4,
        }}
        onError={vi.fn()}
        onPreview={vi.fn()}
      />,
    )
    const viewport = screen.getByLabelText('Gaussian 场景视口')
    vi.spyOn(viewport, 'getBoundingClientRect').mockReturnValue({
      x: 0, y: 0, left: 0, top: 0, right: 960, bottom: 540,
      width: 960, height: 540, toJSON: () => ({}),
    })
    fireEvent.pointerDown(viewport, { clientX: 100, clientY: 100, pointerId: 1 })
    fireEvent.pointerMove(viewport, { clientX: 140, clientY: 120, pointerId: 1 })
    fireEvent.pointerUp(viewport, { clientX: 140, clientY: 120, pointerId: 1 })
    fireEvent.click(viewport, { clientX: 140, clientY: 120 })
    expect(backend.pickFootPoint).not.toHaveBeenCalled()
  })

  it('synchronizes a newer authoritative camera and preview without retaining the stale frame', async () => {
    const backend = {
      fetchPreviewArtifact: vi.fn(async (id: string) => new Blob([id])),
    } as unknown as BackendClient
    const props = {
      backend,
      onError: vi.fn(),
      onPreview: vi.fn(),
    }
    const view = render(
      <SceneViewport
        {...props}
        camera={{ target: [0, 0, 0], distance: 4, yaw: 0, pitch: 0, fov_y_degrees: 50 }}
        initialPreview={{
          artifact_id: 'preview-4', artifact_size: 8, artifact_sha256: 'a',
          generation: 4, width: 960, height: 540, camera_revision: 4,
          pick_buffer_revision: 4,
        }}
      />,
    )
    view.rerender(
      <SceneViewport
        {...props}
        camera={{ target: [1, 0, 0], distance: 8, yaw: 20, pitch: 5, fov_y_degrees: 62 }}
        initialPreview={{
          artifact_id: 'preview-9', artifact_size: 8, artifact_sha256: 'b',
          generation: 9, width: 960, height: 540, camera_revision: 9,
          pick_buffer_revision: 9,
        }}
      />,
    )
    expect(screen.getByRole('slider', { name: '垂直视场角' })).toHaveValue('62')
    expect(backend.fetchPreviewArtifact).toHaveBeenCalledWith('preview-9', expect.any(AbortSignal))
  })

  it('debounces preview requests and ignores an out-of-order stale frame', async () => {
    vi.useFakeTimers()
    const requests: Array<{ signal?: AbortSignal; resolve: (value: never) => void }> = []
    const backend = {
      renderPreview: vi.fn((_input, signal) => new Promise((resolve) => {
        requests.push({ signal, resolve })
      })),
      fetchPreviewArtifact: vi.fn(),
    } as unknown as BackendClient
    const onPreview = vi.fn()
    render(
      <SceneViewport
        backend={backend}
        camera={{ target: [0, 0, 0], distance: 4, yaw: 0, pitch: 0, fov_y_degrees: 50 }}
        onError={vi.fn()}
        onPreview={onPreview}
      />,
    )

    fireEvent.wheel(screen.getByLabelText('Gaussian 场景视口'), { deltaY: 100 })
    await vi.advanceTimersByTimeAsync(150)
    fireEvent.wheel(screen.getByLabelText('Gaussian 场景视口'), { deltaY: 100 })
    await vi.advanceTimersByTimeAsync(150)

    expect(backend.renderPreview).toHaveBeenCalledTimes(2)
    expect(requests[0]?.signal?.aborted).toBe(true)
    vi.useRealTimers()
  })

  it('publishes only the newest frame when an aborted renderer resolves late', async () => {
    vi.useFakeTimers()
    const pending: Array<{
      input: PreviewRequest
      resolve(value: PreviewFrameDto): void
    }> = []
    const backend = {
      renderPreview: vi.fn((input: PreviewRequest) => new Promise<PreviewFrameDto>((resolve) => pending.push({ input, resolve }))),
      fetchPreviewArtifact: vi.fn(async (id: string) => new Blob([id])),
    } as unknown as BackendClient
    const onPreview = vi.fn()
    render(
      <SceneViewport
        backend={backend}
        camera={{ target: [0, 0, 0], distance: 4, yaw: 0, pitch: 0, fov_y_degrees: 50 }}
        onError={vi.fn()}
        onPreview={onPreview}
      />,
    )
    await vi.advanceTimersByTimeAsync(150)
    fireEvent.wheel(screen.getByLabelText('Gaussian 场景视口'), { deltaY: 100 })
    await vi.advanceTimersByTimeAsync(150)
    expect(pending).toHaveLength(2)
    const second = pending[1]!
    await act(async () => second.resolve({
      artifact_id: 'new', generation: second.input.generation, width: 960, height: 540,
      camera_revision: 2, pick_buffer_revision: 2,
    }))
    const first = pending[0]!
    await act(async () => first.resolve({
      artifact_id: 'old', generation: first.input.generation, width: 960, height: 540,
      camera_revision: 1, pick_buffer_revision: 1,
    }))
    expect(onPreview).toHaveBeenCalledTimes(1)
    expect(onPreview.mock.calls[0]?.[0]).toMatchObject({ artifact_id: 'new' })
    vi.useRealTimers()
  })

  it('binds a text-input pick to the newest authoritative preview artifact', async () => {
    const user = userEvent.setup()
    const backend = {
      fetchPreviewArtifact: vi.fn(async () => new Blob(['preview'])),
      pickFootPoint: vi.fn(async (input) => ({
        image: [input.x, input.y], world: [0, 0, 0],
        preview_artifact_id: input.preview_artifact_id,
        camera_revision: input.camera_revision,
        pick_buffer_revision: input.pick_buffer_revision,
      })),
    } as unknown as BackendClient
    const props = { backend, onError: vi.fn(), onPreview: vi.fn() }
    const view = render(
      <SceneViewport
        {...props}
        camera={{ target: [0, 0, 0], distance: 4, yaw: 0, pitch: 0, fov_y_degrees: 50 }}
        initialPreview={{ artifact_id: 'old', artifact_size: 1, artifact_sha256: 'a', generation: 1, width: 960, height: 540, camera_revision: 1, pick_buffer_revision: 1 }}
      />,
    )
    view.rerender(
      <SceneViewport
        {...props}
        camera={{ target: [0, 0, 0], distance: 4, yaw: 0, pitch: 0, fov_y_degrees: 50 }}
        initialPreview={{ artifact_id: 'new', artifact_size: 1, artifact_sha256: 'b', generation: 2, width: 960, height: 540, camera_revision: 2, pick_buffer_revision: 3 }}
      />,
    )
    await waitFor(() => expect(screen.getByRole('button', { name: '确认场景落脚点' })).toBeEnabled())
    await user.type(screen.getByLabelText('落脚点 X 坐标'), '320')
    await user.type(screen.getByLabelText('落脚点 Y 坐标'), '180')
    await user.click(screen.getByRole('button', { name: '确认场景落脚点' }))
    expect(backend.pickFootPoint).toHaveBeenCalledWith({
      x: 320, y: 180, preview_artifact_id: 'new', camera_revision: 2,
      pick_buffer_revision: 3,
    })
  })

  it('revokes old frame authority while a replacement artifact is still downloading', async () => {
    let resolveReplacement: ((value: Blob) => void) | undefined
    const createUrl = vi.spyOn(URL, 'createObjectURL')
      .mockReturnValueOnce('blob:old')
      .mockReturnValueOnce('blob:new')
    const revokeUrl = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined)
    const backend = {
      fetchPreviewArtifact: vi.fn((id: string) => id === 'old'
        ? Promise.resolve(new Blob(['old']))
        : new Promise<Blob>((resolve) => { resolveReplacement = resolve })),
      pickFootPoint: vi.fn(),
      confirmCamera: vi.fn(),
    } as unknown as BackendClient
    const props = { backend, onError: vi.fn(), onPreview: vi.fn() }
    const camera = { target: [0, 0, 0] as [number, number, number], distance: 4, yaw: 0, pitch: 0, fov_y_degrees: 50 }
    const view = render(
      <SceneViewport
        {...props}
        camera={camera}
        initialPreview={{ artifact_id: 'old', artifact_size: 1, artifact_sha256: 'a', generation: 1, width: 960, height: 540, camera_revision: 1, pick_buffer_revision: 1 }}
      />,
    )
    expect(await screen.findByRole('img', { name: /Gaussian 场景.*预览/ })).toHaveAttribute('src', 'blob:old')

    view.rerender(
      <SceneViewport
        {...props}
        camera={camera}
        initialPreview={{ artifact_id: 'new', artifact_size: 1, artifact_sha256: 'b', generation: 2, width: 960, height: 540, camera_revision: 2, pick_buffer_revision: 2 }}
      />,
    )

    expect(screen.queryByRole('img', { name: /Gaussian 场景.*预览/ })).toBeNull()
    expect(screen.getByRole('button', { name: '确认初始机位' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '确认场景落脚点' })).toBeDisabled()
    expect(revokeUrl).toHaveBeenCalledWith('blob:old')
    expect(backend.pickFootPoint).not.toHaveBeenCalled()

    resolveReplacement?.(new Blob(['new']))
    await waitFor(() => expect(screen.getByRole('button', { name: '确认初始机位' })).toBeEnabled())
    expect(screen.getByRole('img', { name: /Gaussian 场景.*预览/ })).toHaveAttribute('src', 'blob:new')
    expect(createUrl).toHaveBeenCalledTimes(2)
  })

  it('revokes replaced and unmounted preview object URLs', async () => {
    const createUrl = vi.spyOn(URL, 'createObjectURL')
      .mockReturnValueOnce('blob:first')
      .mockReturnValueOnce('blob:second')
    const revokeUrl = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined)
    const backend = {
      fetchPreviewArtifact: vi.fn(async () => new Blob(['preview'])),
    } as unknown as BackendClient
    const props = { backend, onError: vi.fn(), onPreview: vi.fn() }
    const view = render(
      <SceneViewport
        {...props}
        camera={{ target: [0, 0, 0], distance: 4, yaw: 0, pitch: 0, fov_y_degrees: 50 }}
        initialPreview={{ artifact_id: 'first', artifact_size: 1, artifact_sha256: 'a', generation: 1, width: 10, height: 10, camera_revision: 1, pick_buffer_revision: 1 }}
      />,
    )
    await waitFor(() => expect(createUrl).toHaveBeenCalledTimes(1))
    view.rerender(
      <SceneViewport
        {...props}
        camera={{ target: [0, 0, 0], distance: 4, yaw: 0, pitch: 0, fov_y_degrees: 50 }}
        initialPreview={{ artifact_id: 'second', artifact_size: 1, artifact_sha256: 'b', generation: 2, width: 10, height: 10, camera_revision: 2, pick_buffer_revision: 2 }}
      />,
    )
    await waitFor(() => expect(revokeUrl).toHaveBeenCalledWith('blob:first'))
    view.unmount()
    expect(revokeUrl).toHaveBeenCalledWith('blob:second')
  })

  it('provides keyboard-accessible FOV controls and coordinate inputs', async () => {
    const user = userEvent.setup()
    const backend = {
      renderPreview: vi.fn(async () => ({
        artifact_id: 'preview-1', generation: 1, width: 960, height: 540,
        camera_revision: 1, pick_buffer_revision: 1,
      })),
      fetchPreviewArtifact: vi.fn(async () => new Blob(['preview'])),
    } as unknown as BackendClient
    render(
      <SceneViewport
        backend={backend}
        camera={{ target: [0, 0, 0], distance: 4, yaw: 0, pitch: 0, fov_y_degrees: 50 }}
        onError={vi.fn()}
        onPreview={vi.fn()}
      />,
    )
    const fov = screen.getByRole('slider', { name: '垂直视场角' })
    await user.click(fov)
    await user.keyboard('{ArrowRight}')
    expect(fov).toHaveValue('51')
    expect(screen.getByLabelText('落脚点 X 坐标')).toBeInTheDocument()
    expect(screen.getByLabelText('落脚点 Y 坐标')).toBeInTheDocument()
  })
})
