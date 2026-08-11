import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { StrictMode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { BackendClient } from '../../api/backend-client'
import { BackendClientError } from '../../api/http-backend-client'
import type { CameraDto, LivePreviewRequest, PreviewFrameDto, PreviewRequest } from '../../api/types'
import { toImagePoint } from '../coordinates/image-point'
import { SceneViewport } from './scene-viewport'

afterEach(() => vi.useRealTimers())

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
  it('survives StrictMode effect replay and releases only after the final unmount', async () => {
    const backend = {
      renderLivePreview: vi.fn((_input, signal?: AbortSignal) => new Promise<Blob>((_resolve, reject) => {
        signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')), {
          once: true,
        })
      })),
      closeLivePreview: vi.fn().mockResolvedValue(undefined),
      renderPreview: vi.fn(() => new Promise<PreviewFrameDto>(() => undefined)),
    } as unknown as BackendClient

    const view = render(
      <StrictMode>
        <SceneViewport
          backend={backend}
          camera={{ target: [0, 0, 0], distance: 4, yaw: 0, pitch: 0, fov_y_degrees: 50 }}
          onError={vi.fn()}
          onPreview={vi.fn()}
        />
      </StrictMode>,
    )

    await waitFor(() => expect(backend.renderLivePreview).toHaveBeenCalledTimes(2))
    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(backend.closeLivePreview).not.toHaveBeenCalled()

    view.unmount()
    await waitFor(() => expect(backend.closeLivePreview).toHaveBeenCalledOnce())
  })

  it('keeps one realtime request in flight and then renders only the latest camera', async () => {
    const createUrl = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:live')
    const liveRequests: LivePreviewRequest[] = []
    const pending: Array<{ resolve(value: Blob): void }> = []
    const backend = {
      renderLivePreview: vi.fn((input) => {
        liveRequests.push(input)
        return new Promise<Blob>((resolve) => pending.push({ resolve }))
      }),
      renderPreview: vi.fn(() => new Promise<PreviewFrameDto>(() => undefined)),
    } as unknown as BackendClient
    render(
      <SceneViewport
        backend={backend}
        camera={{ target: [0, 0, 0], distance: 4, yaw: 0, pitch: 0, fov_y_degrees: 50 }}
        onError={vi.fn()}
        onPreview={vi.fn()}
      />,
    )

    await waitFor(() => expect(backend.renderLivePreview).toHaveBeenCalledTimes(1))
    const viewport = screen.getByLabelText('Gaussian 场景视口')
    fireEvent.pointerDown(viewport, { clientX: 10, clientY: 10, pointerId: 1 })
    fireEvent.pointerMove(viewport, { clientX: 30, clientY: 10, pointerId: 1 })
    fireEvent.pointerMove(viewport, { clientX: 70, clientY: 30, pointerId: 1 })
    expect(backend.renderLivePreview).toHaveBeenCalledTimes(1)

    await act(async () => pending[0]!.resolve(new Blob(['live'], { type: 'image/jpeg' })))

    await waitFor(() => expect(backend.renderLivePreview).toHaveBeenCalledTimes(2))
    expect(createUrl).not.toHaveBeenCalled()
    expect(liveRequests[1]!.camera).toMatchObject({ yaw: 15, pitch: 5 })
    expect(screen.getByRole('button', { name: '确认初始机位' })).toBeDisabled()
  })

  it('does not display a stale realtime frame after an A-B-A camera change', async () => {
    const createUrl = vi.spyOn(URL, 'createObjectURL')
      .mockReturnValueOnce('blob:live-a')
      .mockReturnValueOnce('blob:latest-a')
    const pending: Array<{ resolve(value: Blob): void }> = []
    const backend = {
      renderLivePreview: vi.fn(() => new Promise<Blob>((resolve) => pending.push({ resolve }))),
      renderPreview: vi.fn(() => new Promise<PreviewFrameDto>(() => undefined)),
    } as unknown as BackendClient
    render(
      <SceneViewport
        backend={backend}
        camera={{ target: [0, 0, 0], distance: 4, yaw: 0, pitch: 0, fov_y_degrees: 50 }}
        onError={vi.fn()}
        onPreview={vi.fn()}
      />,
    )

    await waitFor(() => expect(backend.renderLivePreview).toHaveBeenCalledOnce())
    await act(async () => pending[0]!.resolve(new Blob(['a'])))
    expect(await screen.findByRole('img', { name: /Gaussian 场景.*预览/ })).toHaveAttribute('src', 'blob:live-a')

    const viewport = screen.getByLabelText('Gaussian 场景视口')
    fireEvent.pointerDown(viewport, { clientX: 0, clientY: 0, pointerId: 1 })
    fireEvent.pointerMove(viewport, { clientX: 40, clientY: 0, pointerId: 1 })
    fireEvent.pointerUp(viewport, { clientX: 40, clientY: 0, pointerId: 1 })
    await waitFor(() => expect(backend.renderLivePreview).toHaveBeenCalledTimes(2))

    fireEvent.pointerDown(viewport, { clientX: 40, clientY: 0, pointerId: 2 })
    fireEvent.pointerMove(viewport, { clientX: 0, clientY: 0, pointerId: 2 })
    fireEvent.pointerUp(viewport, { clientX: 0, clientY: 0, pointerId: 2 })
    await act(async () => pending[1]!.resolve(new Blob(['stale-b'])))

    await waitFor(() => expect(backend.renderLivePreview).toHaveBeenCalledTimes(3))
    expect(createUrl).toHaveBeenCalledOnce()
    expect(screen.getByRole('img', { name: /Gaussian 场景.*预览/ })).toHaveAttribute('src', 'blob:live-a')
  })

  it('ignores a stale realtime failure after an A-B-A camera change', async () => {
    const pending: Array<{ resolve(value: Blob): void; reject(reason: unknown): void }> = []
    const onError = vi.fn()
    const backend = {
      renderLivePreview: vi.fn(() => new Promise<Blob>((resolve, reject) => pending.push({ resolve, reject }))),
      renderPreview: vi.fn(() => new Promise<PreviewFrameDto>(() => undefined)),
    } as unknown as BackendClient
    render(
      <SceneViewport
        backend={backend}
        camera={{ target: [0, 0, 0], distance: 4, yaw: 0, pitch: 0, fov_y_degrees: 50 }}
        onError={onError}
        onPreview={vi.fn()}
      />,
    )

    await waitFor(() => expect(backend.renderLivePreview).toHaveBeenCalledOnce())
    await act(async () => pending[0]!.resolve(new Blob(['a'])))
    const viewport = screen.getByLabelText('Gaussian 场景视口')
    fireEvent.pointerDown(viewport, { clientX: 0, clientY: 0, pointerId: 1 })
    fireEvent.pointerMove(viewport, { clientX: 40, clientY: 0, pointerId: 1 })
    fireEvent.pointerUp(viewport, { clientX: 40, clientY: 0, pointerId: 1 })
    await waitFor(() => expect(backend.renderLivePreview).toHaveBeenCalledTimes(2))
    fireEvent.pointerDown(viewport, { clientX: 40, clientY: 0, pointerId: 2 })
    fireEvent.pointerMove(viewport, { clientX: 0, clientY: 0, pointerId: 2 })
    fireEvent.pointerUp(viewport, { clientX: 0, clientY: 0, pointerId: 2 })

    await act(async () => pending[1]!.reject(new Error('stale B failed')))

    await waitFor(() => expect(backend.renderLivePreview).toHaveBeenCalledTimes(3))
    expect(onError).not.toHaveBeenCalled()
  })

  it('stops realtime pumping and requests authority immediately after live failure', async () => {
    vi.useFakeTimers()
    const onError = vi.fn()
    const backend = {
      renderLivePreview: vi.fn().mockRejectedValue(new Error('live failed')),
      closeLivePreview: vi.fn().mockResolvedValue(undefined),
      renderPreview: vi.fn(() => new Promise<PreviewFrameDto>(() => undefined)),
    } as unknown as BackendClient
    render(
      <SceneViewport
        backend={backend}
        camera={{ target: [0, 0, 0], distance: 4, yaw: 0, pitch: 0, fov_y_degrees: 50 }}
        onError={onError}
        onPreview={vi.fn()}
      />,
    )

    await act(async () => { await Promise.resolve() })
    await vi.advanceTimersByTimeAsync(0)

    expect(backend.renderLivePreview).toHaveBeenCalledOnce()
    expect(backend.renderPreview).toHaveBeenCalledOnce()
    expect(onError).toHaveBeenCalledWith('实时预览暂不可用，已切换到高质量预览。')
  })

  it('commits precise Yaw and Pitch inputs with camera bounds', async () => {
    vi.useFakeTimers()
    const requests: PreviewRequest[] = []
    const backend = {
      renderPreview: vi.fn((input: PreviewRequest) => {
        requests.push(input)
        return new Promise<PreviewFrameDto>(() => undefined)
      }),
    } as unknown as BackendClient
    render(
      <SceneViewport
        backend={backend}
        camera={{ target: [0, 0, 0], distance: 4, yaw: 10, pitch: 5, fov_y_degrees: 50 }}
        onError={vi.fn()}
        onPreview={vi.fn()}
      />,
    )

    const yaw = screen.getByRole('spinbutton', { name: 'Yaw 角度' })
    const pitch = screen.getByRole('spinbutton', { name: 'Pitch 角度' })
    expect(yaw).toHaveValue(10)
    expect(pitch).toHaveValue(5)

    fireEvent.change(yaw, { target: { value: '190' } })
    fireEvent.keyDown(yaw, { key: 'Enter' })
    expect(yaw).toHaveValue(-170)

    fireEvent.change(pitch, { target: { value: '-100' } })
    fireEvent.blur(pitch)
    expect(pitch).toHaveValue(-89)

    await vi.advanceTimersByTimeAsync(150)
    expect(requests).toHaveLength(1)
    expect(requests[0]!.camera).toMatchObject({ yaw: -170, pitch: -89 })

    const viewport = screen.getByLabelText('Gaussian 场景视口')
    fireEvent.pointerDown(viewport, { clientX: 10, clientY: 10, pointerId: 1 })
    fireEvent.pointerMove(viewport, { clientX: 50, clientY: 30, pointerId: 1 })
    fireEvent.pointerUp(viewport, { clientX: 50, clientY: 30, pointerId: 1 })
    expect(yaw).toHaveValue(-160)
    expect(pitch).toHaveValue(-84)
  })

  it('keeps a cancellable wheel gesture inside the Gaussian viewport', () => {
    const backend = {
      renderPreview: vi.fn(() => new Promise<PreviewFrameDto>(() => undefined)),
    } as unknown as BackendClient
    const view = render(
      <SceneViewport
        backend={backend}
        camera={{ target: [0, 0, 0], distance: 4, yaw: 0, pitch: 0, fov_y_degrees: 50 }}
        onError={vi.fn()}
        onPreview={vi.fn()}
      />,
    )
    const viewport = screen.getByLabelText('Gaussian 场景视口')
    const wheel = new WheelEvent('wheel', { bubbles: true, cancelable: true, deltaY: 100 })

    let propagated = true
    act(() => { propagated = viewport.dispatchEvent(wheel) })

    expect(propagated).toBe(false)
    expect(wheel.defaultPrevented).toBe(true)
    expect(screen.getByText('距离 4.32')).toBeVisible()

    view.unmount()
    const afterUnmount = new WheelEvent('wheel', { bubbles: true, cancelable: true, deltaY: 100 })
    expect(viewport.dispatchEvent(afterUnmount)).toBe(true)
    expect(afterUnmount.defaultPrevented).toBe(false)
  })

  it('does not retain the React change event across batched FOV updates', async () => {
    const backend = {
      fetchPreviewArtifact: vi.fn(async () => new Blob(['preview'])),
      renderPreview: vi.fn(() => new Promise<PreviewFrameDto>(() => undefined)),
    } as unknown as BackendClient
    render(
      <SceneViewport
        backend={backend}
        camera={{ target: [0, 0, 0], distance: 4, yaw: 0, pitch: 0, fov_y_degrees: 50 }}
        initialPreview={{
          artifact_id: 'preview-1', artifact_size: 1, artifact_sha256: 'a',
          generation: 1, width: 960, height: 540, camera_revision: 1,
          pick_buffer_revision: 1,
        }}
        onError={vi.fn()}
        onPreview={vi.fn()}
      />,
    )
    await act(async () => { await Promise.resolve() })
    const slider = screen.getByRole('slider', { name: '垂直视场角' })

    expect(() => {
      act(() => {
        fireEvent.change(slider, { target: { value: '51' } })
        fireEvent.change(slider, { target: { value: '52' } })
      })
    }).not.toThrow()
    expect(slider).toHaveValue('52')
  })

  it('projects a restored camera to the strict preview request contract', async () => {
    vi.useFakeTimers()
    const requests: PreviewRequest[] = []
    const backend = {
      renderPreview: vi.fn((input: PreviewRequest) => {
        requests.push(input)
        return new Promise<PreviewFrameDto>(() => undefined)
      }),
    } as unknown as BackendClient
    const restoredCamera: CameraDto = {
      target: [0, 0, 0], distance: 4, yaw: 15, pitch: 5,
      fov_y_degrees: 50, revision: 7,
    }

    render(
      <SceneViewport
        backend={backend}
        camera={restoredCamera}
        onError={vi.fn()}
        onPreview={vi.fn()}
      />,
    )
    await vi.advanceTimersByTimeAsync(150)

    expect(requests).toHaveLength(1)
    expect(requests[0]!.camera).toEqual({
      target: [0, 0, 0], distance: 4, yaw: 15, pitch: 5, fov_y_degrees: 50,
    })
    expect(requests[0]!.camera).not.toHaveProperty('revision')
  })

  it('allocates a newer generation after remount while the old request is pending', async () => {
    vi.useFakeTimers()
    const requests: PreviewRequest[] = []
    const backend = {
      renderPreview: vi.fn((input: PreviewRequest) => {
        requests.push(input)
        return new Promise<PreviewFrameDto>(() => undefined)
      }),
    } as unknown as BackendClient
    const props = {
      backend,
      onError: vi.fn(),
      onPreview: vi.fn(),
    }

    const first = render(
      <SceneViewport
        {...props}
        camera={{ target: [0, 0, 0], distance: 4, yaw: 0, pitch: 0, fov_y_degrees: 50 }}
      />,
    )
    await vi.advanceTimersByTimeAsync(150)
    first.unmount()

    render(
      <SceneViewport
        {...props}
        camera={{ target: [0, 0, 0], distance: 4, yaw: 15, pitch: 0, fov_y_degrees: 50 }}
      />,
    )
    await vi.advanceTimersByTimeAsync(150)

    expect(requests).toHaveLength(2)
    expect(requests[1]!.generation).toBeGreaterThan(requests[0]!.generation)
  })

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
        confirmedCameraRevision={1}
        confirmedPreviewArtifactId="preview-1"
        initialPreview={{ artifact_id: 'preview-1', artifact_size: 1, artifact_sha256: 'a', generation: 1, width: 960, height: 540, camera_revision: 1, pick_buffer_revision: 1 }}
        onError={vi.fn()}
        onPreview={vi.fn()}
      />,
    )
    await act(async () => { await Promise.resolve() })
    expect(screen.getByRole('button', { name: '确认初始机位' })).toBeEnabled()
    fireEvent.change(screen.getByLabelText('落脚点 X 坐标'), { target: { value: '320' } })
    fireEvent.change(screen.getByLabelText('落脚点 Y 坐标'), { target: { value: '180' } })
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
        confirmedCameraRevision={1}
        confirmedPreviewArtifactId="old"
        initialPreview={{ artifact_id: 'old', artifact_size: 1, artifact_sha256: 'a', generation: 1, width: 960, height: 540, camera_revision: 1, pick_buffer_revision: 1 }}
      />,
    )
    view.rerender(
      <SceneViewport
        {...props}
        camera={{ target: [0, 0, 0], distance: 4, yaw: 0, pitch: 0, fov_y_degrees: 50 }}
        confirmedCameraRevision={2}
        confirmedPreviewArtifactId="new"
        initialPreview={{ artifact_id: 'new', artifact_size: 1, artifact_sha256: 'b', generation: 2, width: 960, height: 540, camera_revision: 2, pick_buffer_revision: 3 }}
      />,
    )
    await waitFor(() => expect(screen.getByRole('button', { name: '确认初始机位' })).toBeEnabled())
    await user.type(screen.getByLabelText('落脚点 X 坐标'), '320')
    await user.type(screen.getByLabelText('落脚点 Y 坐标'), '180')
    await waitFor(() => expect(screen.getByRole('button', { name: '确认场景落脚点' })).toBeEnabled())
    await user.click(screen.getByRole('button', { name: '确认场景落脚点' }))
    expect(backend.pickFootPoint).toHaveBeenCalledWith({
      x: 320, y: 180, preview_artifact_id: 'new', camera_revision: 2,
      pick_buffer_revision: 3,
    })
  })

  it('keeps a viewport click local until the user explicitly confirms the foot point', async () => {
    const backend = {
      fetchPreviewArtifact: vi.fn(async () => new Blob(['preview'])),
      pickFootPoint: vi.fn(async (input) => ({
        image: [input.x, input.y], world: [0, 0, 0],
        preview_artifact_id: input.preview_artifact_id,
        camera_revision: input.camera_revision,
        pick_buffer_revision: input.pick_buffer_revision,
      })),
    } as unknown as BackendClient
    render(
      <SceneViewport
        backend={backend}
        camera={{ target: [0, 0, 0], distance: 4, yaw: 0, pitch: 0, fov_y_degrees: 50 }}
        confirmedCameraRevision={4}
        confirmedPreviewArtifactId="preview-4"
        initialPreview={{ artifact_id: 'preview-4', artifact_size: 8, artifact_sha256: 'sha', generation: 4, width: 960, height: 540, camera_revision: 4, pick_buffer_revision: 4 }}
        onError={vi.fn()}
        onPreview={vi.fn()}
      />,
    )
    const viewport = screen.getByLabelText('Gaussian 场景视口')
    vi.spyOn(viewport, 'getBoundingClientRect').mockReturnValue({
      x: 0, y: 0, left: 0, top: 0, right: 962, bottom: 542,
      width: 962, height: 542, toJSON: () => ({}),
    })
    vi.spyOn(await screen.findByRole('img', { name: /Gaussian 场景.*预览/ }), 'getBoundingClientRect').mockReturnValue({
      x: 1, y: 1, left: 1, top: 1, right: 961, bottom: 541,
      width: 960, height: 540, toJSON: () => ({}),
    })
    await waitFor(() => expect(screen.getByRole('button', { name: '确认初始机位' })).toBeEnabled())

    fireEvent.click(viewport, { clientX: 321, clientY: 181 })

    expect(screen.getByLabelText('落脚点 X 坐标')).toHaveValue(320)
    expect(screen.getByLabelText('落脚点 Y 坐标')).toHaveValue(180)
    expect(screen.getByText('X 320 · Y 180')).toBeInTheDocument()
    expect(backend.pickFootPoint).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: '确认场景落脚点' }))
    await waitFor(() => expect(backend.pickFootPoint).toHaveBeenCalledOnce())
  })

  it('restores only a foot point bound to the current authoritative preview', async () => {
    const backend = {
      fetchPreviewArtifact: vi.fn(async (id: string) => new Blob([id])),
    } as unknown as BackendClient
    const camera = { target: [0, 0, 0] as [number, number, number], distance: 4, yaw: 0, pitch: 0, fov_y_degrees: 50 }
    const props = { backend, camera, onError: vi.fn(), onPreview: vi.fn() }
    const footPoint = {
      image: [300, 200] as [number, number], world: [0, 0, 0] as [number, number, number],
      preview_artifact_id: 'old', camera_revision: 1, pick_buffer_revision: 1,
    }
    const view = render(
      <SceneViewport
        {...props}
        confirmedCameraRevision={1}
        confirmedPreviewArtifactId="old"
        initialFootPoint={footPoint}
        initialPreview={{ artifact_id: 'old', artifact_size: 1, artifact_sha256: 'a', generation: 1, width: 960, height: 540, camera_revision: 1, pick_buffer_revision: 1 }}
      />,
    )

    await waitFor(() => expect(screen.getByLabelText('落脚点 X 坐标')).toHaveValue(300))
    expect(screen.getByLabelText('落脚点 Y 坐标')).toHaveValue(200)
    expect(screen.getByText('场景落脚点已验证')).toBeVisible()

    view.rerender(
      <SceneViewport
        {...props}
        confirmedCameraRevision={2}
        confirmedPreviewArtifactId="new"
        initialFootPoint={footPoint}
        initialPreview={{ artifact_id: 'new', artifact_size: 1, artifact_sha256: 'b', generation: 2, width: 960, height: 540, camera_revision: 2, pick_buffer_revision: 2 }}
      />,
    )

    await waitFor(() => expect(screen.getByRole('button', { name: '确认初始机位' })).toBeEnabled())
    expect(screen.getByLabelText('落脚点 X 坐标')).toHaveValue(null)
    expect(screen.getByLabelText('落脚点 Y 坐标')).toHaveValue(null)
    expect(screen.getByText('尚未选择场景落脚点')).toBeVisible()
  })

  it('does not restore a foot point when camera confirmation disagrees with the preview', async () => {
    const backend = {
      fetchPreviewArtifact: vi.fn(async () => new Blob(['preview'])),
    } as unknown as BackendClient
    render(
      <SceneViewport
        backend={backend}
        camera={{ target: [0, 0, 0], distance: 4, yaw: 0, pitch: 0, fov_y_degrees: 50 }}
        confirmedCameraRevision={1}
        confirmedPreviewArtifactId="different-preview"
        initialFootPoint={{
          image: [300, 200], world: [0, 0, 0], preview_artifact_id: 'preview-1',
          camera_revision: 1, pick_buffer_revision: 1,
        }}
        initialPreview={{ artifact_id: 'preview-1', artifact_size: 1, artifact_sha256: 'a', generation: 1, width: 960, height: 540, camera_revision: 1, pick_buffer_revision: 1 }}
        onError={vi.fn()}
        onPreview={vi.fn()}
      />,
    )

    await screen.findByRole('img', { name: /Gaussian 场景.*预览/ })
    expect(screen.getByLabelText('落脚点 X 坐标')).toHaveValue(null)
    expect(screen.getByLabelText('落脚点 Y 坐标')).toHaveValue(null)
    expect(screen.getByText('尚未选择场景落脚点')).toBeVisible()
    expect(screen.getByRole('button', { name: '确认场景落脚点' })).toBeDisabled()
  })

  it('keeps initial preview authority unavailable until its artifact has loaded', async () => {
    let resolveArtifact: ((value: Blob) => void) | undefined
    const backend = {
      fetchPreviewArtifact: vi.fn(() => new Promise<Blob>((resolve) => { resolveArtifact = resolve })),
      pickFootPoint: vi.fn(),
    } as unknown as BackendClient
    render(
      <SceneViewport
        backend={backend}
        camera={{ target: [0, 0, 0], distance: 4, yaw: 0, pitch: 0, fov_y_degrees: 50 }}
        confirmedCameraRevision={1}
        confirmedPreviewArtifactId="preview-1"
        initialFootPoint={{
          image: [300, 200], world: [0, 0, 0], preview_artifact_id: 'preview-1',
          camera_revision: 1, pick_buffer_revision: 1,
        }}
        initialPreview={{ artifact_id: 'preview-1', artifact_size: 1, artifact_sha256: 'a', generation: 1, width: 960, height: 540, camera_revision: 1, pick_buffer_revision: 1 }}
        onError={vi.fn()}
        onPreview={vi.fn()}
      />,
    )

    expect(screen.getByLabelText('落脚点 X 坐标')).toBeDisabled()
    expect(screen.getByRole('button', { name: '确认初始机位' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '确认场景落脚点' })).toBeDisabled()
    expect(backend.pickFootPoint).not.toHaveBeenCalled()

    resolveArtifact?.(new Blob(['preview']))
    await waitFor(() => expect(screen.getByLabelText('落脚点 X 坐标')).toHaveValue(300))
    expect(screen.getByRole('button', { name: '确认场景落脚点' })).toBeEnabled()
  })

  it('ignores a successful pick response after preview authority changes', async () => {
    let resolvePick: ((value: {
      image: [number, number]
      world: [number, number, number]
      preview_artifact_id: string
      camera_revision: number
      pick_buffer_revision: number
    }) => void) | undefined
    const onFootPoint = vi.fn()
    const backend = {
      fetchPreviewArtifact: vi.fn(async (id: string) => new Blob([id])),
      pickFootPoint: vi.fn(() => new Promise((resolve) => { resolvePick = resolve })),
    } as unknown as BackendClient
    const props = { backend, onError: vi.fn(), onFootPoint, onPreview: vi.fn() }
    const view = render(
      <SceneViewport
        {...props}
        camera={{ target: [0, 0, 0], distance: 4, yaw: 0, pitch: 0, fov_y_degrees: 50 }}
        confirmedCameraRevision={1}
        confirmedPreviewArtifactId="old"
        initialPreview={{ artifact_id: 'old', artifact_size: 1, artifact_sha256: 'a', generation: 1, width: 960, height: 540, camera_revision: 1, pick_buffer_revision: 1 }}
      />,
    )
    await waitFor(() => expect(screen.getByLabelText('落脚点 X 坐标')).toBeEnabled())
    fireEvent.change(screen.getByLabelText('落脚点 X 坐标'), { target: { value: '320' } })
    fireEvent.change(screen.getByLabelText('落脚点 Y 坐标'), { target: { value: '180' } })
    await waitFor(() => expect(screen.getByRole('button', { name: '确认场景落脚点' })).toBeEnabled())
    fireEvent.click(screen.getByRole('button', { name: '确认场景落脚点' }))
    await waitFor(() => expect(backend.pickFootPoint).toHaveBeenCalledOnce())

    view.rerender(
      <SceneViewport
        {...props}
        camera={{ target: [0, 0, 0], distance: 4, yaw: 0, pitch: 0, fov_y_degrees: 50 }}
        confirmedCameraRevision={2}
        confirmedPreviewArtifactId="new"
        initialPreview={{ artifact_id: 'new', artifact_size: 1, artifact_sha256: 'b', generation: 2, width: 960, height: 540, camera_revision: 2, pick_buffer_revision: 2 }}
      />,
    )
    await screen.findByRole('img', { name: /Gaussian 场景.*预览/ })
    resolvePick?.({ image: [320, 180], world: [0, 0, 0], preview_artifact_id: 'old', camera_revision: 1, pick_buffer_revision: 1 })

    await act(async () => { await Promise.resolve() })
    expect(onFootPoint).not.toHaveBeenCalled()
    expect(screen.getByLabelText('落脚点 X 坐标')).toHaveValue(null)
    expect(screen.getByLabelText('落脚点 Y 坐标')).toHaveValue(null)
  })

  it.each(['stale_pick_buffer', 'camera_not_confirmed'])('locks rejected %s authority until refresh completes', async (code) => {
    let rejectPick: ((reason: unknown) => void) | undefined
    let resolveRefresh: (() => void) | undefined
    const onAuthorityStale = vi.fn(() => new Promise<void>((resolve) => { resolveRefresh = resolve }))
    const backend = {
      fetchPreviewArtifact: vi.fn(async () => new Blob(['preview'])),
      pickFootPoint: vi.fn(() => new Promise((_resolve, reject) => { rejectPick = reject })),
      confirmCamera: vi.fn(async () => ({})),
    } as unknown as BackendClient
    render(
      <SceneViewport
        backend={backend}
        camera={{ target: [0, 0, 0], distance: 4, yaw: 0, pitch: 0, fov_y_degrees: 50 }}
        confirmedCameraRevision={1}
        confirmedPreviewArtifactId="preview-1"
        initialPreview={{ artifact_id: 'preview-1', artifact_size: 1, artifact_sha256: 'a', generation: 1, width: 960, height: 540, camera_revision: 1, pick_buffer_revision: 1 }}
        onAuthorityStale={onAuthorityStale}
        onError={vi.fn()}
        onPreview={vi.fn()}
      />,
    )
    await waitFor(() => expect(screen.getByLabelText('落脚点 X 坐标')).toBeEnabled())
    fireEvent.change(screen.getByLabelText('落脚点 X 坐标'), { target: { value: '320' } })
    fireEvent.change(screen.getByLabelText('落脚点 Y 坐标'), { target: { value: '180' } })
    await waitFor(() => expect(screen.getByRole('button', { name: '确认场景落脚点' })).toBeEnabled())
    fireEvent.click(screen.getByRole('button', { name: '确认场景落脚点' }))
    expect(screen.getByLabelText('Yaw 角度')).toBeDisabled()
    expect(screen.getByLabelText('落脚点 X 坐标')).toBeDisabled()

    rejectPick?.(new BackendClientError(409, {
      code, category: 'conflict', message: 'stale authority', retryable: true,
    }))
    await waitFor(() => expect(onAuthorityStale).toHaveBeenCalledOnce())
    expect(screen.getByLabelText('落脚点 X 坐标')).toHaveValue(null)
    expect(screen.getByRole('button', { name: '验证中…' })).toBeDisabled()

    resolveRefresh?.()
    await waitFor(() => expect(screen.getByLabelText('Yaw 角度')).toBeEnabled())
    if (code === 'stale_pick_buffer') {
      expect(screen.getByLabelText('落脚点 X 坐标')).toBeDisabled()
      expect(screen.getByRole('button', { name: '确认初始机位' })).toBeDisabled()
    } else {
      expect(screen.getByLabelText('落脚点 X 坐标')).toBeEnabled()
      expect(screen.getByRole('button', { name: '确认初始机位' })).toBeEnabled()
      fireEvent.click(screen.getByRole('button', { name: '确认初始机位' }))
      await waitFor(() => expect(backend.confirmCamera).toHaveBeenCalledWith(1))
    }
    expect(screen.getByRole('button', { name: '确认场景落脚点' })).toBeDisabled()
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
