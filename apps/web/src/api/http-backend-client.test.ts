import { act, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { BackendClient } from './backend-client'
import {
  BackendClientError,
  HttpBackendClient,
} from './http-backend-client'
import {
  WebSocketTaskEventSource,
  createTaskStore,
  useTaskEventSource,
} from './task-events'
import type { TaskEventSource, TaskEventSubscription } from './task-events'
import type { PickRequest, TaskDto } from './types'

const task = (revision = 9): TaskDto => ({
  id: 't1',
  target_stage: 'segment',
  revision,
  status: 'running',
  error: null,
})

const fakeBackendClient = (currentTask = task()): BackendClient => ({
  bootstrap: vi.fn(),
  getEnvironmentRepair: vi.fn(),
  startEnvironmentRepair: vi.fn(),
  cancelEnvironmentRepair: vi.fn(),
  importLocalPath: vi.fn(),
  createUpload: vi.fn(),
  putUploadChunk: vi.fn(),
  getUpload: vi.fn(),
  completeUpload: vi.fn(),
  cancelUpload: vi.fn(),
  getProject: vi.fn(),
  updateProject: vi.fn(),
  renderPreview: vi.fn(),
  fetchPreviewArtifact: vi.fn(),
  pickFootPoint: vi.fn(),
  confirmCamera: vi.fn(),
  getVerifiedExport: vi.fn(),
  fetchExportArtifact: vi.fn(),
  getCompositePreview: vi.fn(),
  fetchCompositePreviewArtifact: vi.fn(),
  copyVerifiedExport: vi.fn(),
  getSubjectMedia: vi.fn(),
  fetchSubjectMediaArtifact: vi.fn(),
  startTask: vi.fn(),
  getTask: vi.fn().mockResolvedValue(currentTask),
  cancelTask: vi.fn(),
})

class FakeWebSocket {
  readonly sent: string[] = []
  readonly url: string
  close = vi.fn()
  private readonly listeners = new Map<string, Set<(event: Event) => void>>()

  constructor(url: string) {
    this.url = url
  }

  addEventListener(type: string, listener: (event: Event) => void): void {
    const listeners = this.listeners.get(type) ?? new Set()
    listeners.add(listener)
    this.listeners.set(type, listeners)
  }

  removeEventListener(type: string, listener: (event: Event) => void): void {
    this.listeners.get(type)?.delete(listener)
  }

  send(payload: string): void {
    this.sent.push(payload)
  }

  emit(type: 'open' | 'close' | 'error'): void {
    for (const listener of this.listeners.get(type) ?? []) {
      listener(new Event(type))
    }
  }

  message(payload: unknown): void {
    const event = new MessageEvent('message', { data: JSON.stringify(payload) })
    for (const listener of this.listeners.get('message') ?? []) {
      listener(event)
    }
  }
}

afterEach(() => {
  vi.useRealTimers()
})

describe('HttpBackendClient', () => {
  it('uses the versioned endpoint and keeps the bearer token out of the URL', async () => {
    const fetchImpl = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(JSON.stringify(task()), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    )
    const client = new HttpBackendClient({
      origin: 'http://127.0.0.1:49152',
      token: 'memory-only-secret',
      fetchImpl,
    })

    await client.getTask('t1')

    expect(fetchImpl).toHaveBeenCalledOnce()
    const [url, init] = fetchImpl.mock.calls[0] ?? []
    expect(url).toBe('http://127.0.0.1:49152/api/v1/tasks/t1')
    expect(String(url)).not.toContain('memory-only-secret')
    expect(new Headers(init?.headers).get('Authorization')).toBe(
      'Bearer memory-only-secret',
    )
  })

  it('parses the stable backend error envelope', async () => {
    const fetchImpl = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(
        JSON.stringify({
          code: 'task_conflict',
          category: 'task',
          message: 'The task cannot be changed.',
          retryable: true,
        }),
        { status: 409, headers: { 'Content-Type': 'application/json' } },
      ),
    )
    const client = new HttpBackendClient({
      origin: 'http://127.0.0.1:49152',
      token: 'secret',
      fetchImpl,
    })

    const error = await client.getTask('t1').catch((caught: unknown) => caught)

    expect(error).toBeInstanceOf(BackendClientError)
    expect(error).toMatchObject({
      status: 409,
      code: 'task_conflict',
      category: 'task',
      message: 'The task cannot be changed.',
      retryable: true,
    })
  })

  it('uses a stable fallback when an error body is not the API envelope', async () => {
    const fetchImpl = vi
      .fn<typeof fetch>()
      .mockResolvedValue(new Response('<h1>proxy error</h1>', { status: 502 }))
    const client = new HttpBackendClient({
      origin: 'http://127.0.0.1:49152',
      token: 'secret',
      fetchImpl,
    })

    await expect(client.bootstrap()).rejects.toMatchObject({
      status: 502,
      code: 'http_error',
      category: 'transport',
      message: 'The local service request failed.',
      retryable: true,
    })
  })

  it('turns an elapsed request timeout into a stable client error', async () => {
    vi.useFakeTimers()
    const fetchImpl = vi.fn<typeof fetch>().mockImplementation((_url, init) => {
      return new Promise((_resolve, reject) => {
        init?.signal?.addEventListener('abort', () => reject(init.signal?.reason), {
          once: true,
        })
      })
    })
    const client = new HttpBackendClient({
      origin: 'http://127.0.0.1:49152',
      token: 'secret',
      timeoutMs: 25,
      fetchImpl,
    })

    const request = client.bootstrap()
    const rejection = expect(request).rejects.toMatchObject({
      status: 0,
      code: 'request_timeout',
      category: 'transport',
      retryable: true,
    })
    await vi.advanceTimersByTimeAsync(25)

    await rejection
  })

  it('honors an external AbortSignal without reporting a timeout', async () => {
    const fetchImpl = vi.fn<typeof fetch>().mockImplementation((_url, init) => {
      return new Promise((_resolve, reject) => {
        init?.signal?.addEventListener('abort', () => reject(init.signal?.reason), {
          once: true,
        })
      })
    })
    const client = new HttpBackendClient({
      origin: 'http://127.0.0.1:49152',
      token: 'secret',
      fetchImpl,
    })
    const controller = new AbortController()

    const request = client.bootstrap(controller.signal)
    controller.abort()

    await expect(request).rejects.toMatchObject({
      status: 0,
      code: 'request_aborted',
      category: 'transport',
      retryable: false,
    })
  })

  it('completes and can cancel the browser upload protocol', async () => {
    const fetchImpl = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            id: 'upload-1',
            filename: 'clip.mp4',
            total_size: 4,
            chunk_size: 2,
            uploaded_chunks: [0, 1],
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        ),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ path: 'source/clip.mp4' }), {
          status: 201,
          headers: { 'Content-Type': 'application/json' },
        }),
      )
      .mockResolvedValueOnce(new Response(null, { status: 204 }))
    const client = new HttpBackendClient({
      origin: 'http://127.0.0.1:49152',
      token: 'secret',
      fetchImpl,
    })

    await expect(client.getUpload('upload-1')).resolves.toMatchObject({
      uploaded_chunks: [0, 1],
    })
    await expect(client.completeUpload('upload-1')).resolves.toEqual({
      path: 'source/clip.mp4',
    })
    await expect(client.cancelUpload('upload-1')).resolves.toBeUndefined()

    expect(
      fetchImpl.mock.calls.map(([url, init]) => [
        url,
        init?.method ?? 'GET',
      ]),
    ).toEqual([
      ['http://127.0.0.1:49152/api/v1/uploads/upload-1', 'GET'],
      ['http://127.0.0.1:49152/api/v1/uploads/upload-1/complete', 'POST'],
      ['http://127.0.0.1:49152/api/v1/uploads/upload-1', 'DELETE'],
    ])
  })

  it('renders and fetches a preview through authenticated client methods', async () => {
    const descriptor = {
      artifact_id: 'preview-1',
      generation: 3,
      width: 960,
      height: 540,
      camera_revision: 2,
      pick_buffer_revision: 2,
    }
    const fetchImpl = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(
        new Response(JSON.stringify(descriptor), {
          status: 201,
          headers: { 'Content-Type': 'application/json' },
        }),
      )
      .mockResolvedValueOnce(
        new Response(new Blob(['png'], { type: 'image/png' }), {
          status: 200,
          headers: { 'Content-Type': 'image/png' },
        }),
      )
    const client = new HttpBackendClient({
      origin: 'http://127.0.0.1:49152',
      token: 'secret',
      fetchImpl,
    })
    const controller = new AbortController()

    await expect(
      client.renderPreview(
        {
          generation: 3,
          width: 960,
          height: 540,
          camera: {
            target: [0, 0, 0],
            distance: 4,
            yaw: 0,
            pitch: 0,
            fov_y_degrees: 55,
          },
        },
        controller.signal,
      ),
    ).resolves.toEqual(descriptor)
    const blob = await client.fetchPreviewArtifact('preview-1')
    expect(blob.type).toBe('image/png')
    const authorization = fetchImpl.mock.calls.map(([, init]) =>
      new Headers(init?.headers).get('Authorization'),
    )
    expect(authorization).toEqual(['Bearer secret', 'Bearer secret'])
    expect(String(fetchImpl.mock.calls[1]?.[0])).not.toContain('secret')
  })

  it('binds a foot-point pick to the exact opaque preview artifact', async () => {
    const input: PickRequest = {
      x: 8,
      y: 4,
      preview_artifact_id: 'preview-1',
      camera_revision: 2,
      pick_buffer_revision: 2,
    }
    const fetchImpl = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(
        JSON.stringify({
          image: [8, 4],
          world: [0, 0, 0],
          preview_artifact_id: 'preview-1',
          camera_revision: 2,
          pick_buffer_revision: 2,
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      ),
    )
    const client = new HttpBackendClient({
      origin: 'http://127.0.0.1:49152',
      token: 'secret',
      fetchImpl,
    })

    await client.pickFootPoint(input)

    expect(fetchImpl).toHaveBeenCalledWith(
      'http://127.0.0.1:49152/api/v1/projects/current/pick',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify(input),
      }),
    )
  })

  it('retrieves only a verified opaque export and its authenticated Blob', async () => {
    const result = {
      artifact_id: 'export-1',
      filename: 'final.mp4',
      size: 14,
      duration_seconds: 10,
      fps: '30',
      frame_count: 300,
      has_audio: true,
      verified: true,
    }
    const fetchImpl = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(
        new Response(JSON.stringify(result), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
      )
      .mockResolvedValueOnce(
        new Response(new Blob(['video'], { type: 'video/mp4' }), {
          status: 200,
          headers: { 'Content-Type': 'video/mp4' },
        }),
      )
    const client = new HttpBackendClient({
      origin: 'http://127.0.0.1:49152',
      token: 'secret',
      fetchImpl,
    })

    await expect(client.getVerifiedExport()).resolves.toEqual(result)
    await expect(client.fetchExportArtifact('export-1')).resolves.toBeInstanceOf(
      Blob,
    )
    expect(String(fetchImpl.mock.calls[1]?.[0])).not.toContain('secret')
  })

  it('fetches the current opaque composite preview', async () => {
    const descriptor = {
      artifact_id: 'composite-preview-1',
      filename: 'composite-preview.mp4',
      size: 14,
      sha256: 'a'.repeat(64),
      duration_seconds: 1,
      fps: '30',
      frame_count: 30,
    }
    const fetchImpl = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(
        new Response(JSON.stringify(descriptor), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
      )
      .mockResolvedValueOnce(
        new Response(new Blob(['video'], { type: 'video/mp4' }), {
          status: 200,
          headers: { 'Content-Type': 'video/mp4' },
        }),
      )
    const client = new HttpBackendClient({
      origin: 'http://127.0.0.1:49152',
      token: 'secret',
      fetchImpl,
    })

    await expect(client.getCompositePreview()).resolves.toEqual(descriptor)
    await expect(
      client.fetchCompositePreviewArtifact(descriptor.artifact_id),
    ).resolves.toBeInstanceOf(Blob)
    expect(fetchImpl).toHaveBeenNthCalledWith(
      2,
      expect.stringContaining(
        `/artifacts/composite-previews/${descriptor.artifact_id}`,
      ),
      expect.objectContaining({ headers: expect.any(Headers) }),
    )
  })

  it('propagates caller cancellation to the composite descriptor request', async () => {
    let settleFetch:
      | ((response: Response) => void)
      | undefined
    let requestSignal: AbortSignal | null | undefined
    const fetchImpl = vi.fn<typeof fetch>().mockImplementation((_url, init) => {
      requestSignal = init?.signal
      return new Promise<Response>((resolve, reject) => {
        settleFetch = resolve
        init?.signal?.addEventListener(
          'abort',
          () => reject(init.signal?.reason),
          { once: true },
        )
      })
    })
    const client = new HttpBackendClient({
      origin: 'http://127.0.0.1:49152',
      token: 'secret',
      fetchImpl,
    })
    const controller = new AbortController()

    const request = client.getCompositePreview(controller.signal)
    controller.abort()

    expect(requestSignal?.aborted).toBe(true)
    settleFetch?.(
      new Response('{}', {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    )
    await request.catch(() => undefined)
  })

  it('requests a local copy only for the current opaque export id', async () => {
    const fetchImpl = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(null, { status: 204 }),
    )
    const client = new HttpBackendClient({
      origin: 'http://127.0.0.1:49152',
      token: 'secret',
      fetchImpl,
    })

    await expect(
      client.copyVerifiedExport('export-1', 'E:\\chosen\\result.mp4'),
    ).resolves.toBeUndefined()

    expect(fetchImpl).toHaveBeenCalledWith(
      'http://127.0.0.1:49152/api/v1/projects/current/exports/export-1/copy',
      expect.objectContaining({
        method: 'POST',
        body: '{"destination":"E:\\\\chosen\\\\result.mp4"}',
      }),
    )
    expect(String(fetchImpl.mock.calls[0]?.[0])).not.toContain('secret')
  })

  it('confirms exactly the rendered camera revision through the project API', async () => {
    const project = {
      schema_version: 3,
      project_id: 'p1',
      name: 'demo',
      created_at: '2026-07-16T00:00:00Z',
      source_video: null,
      scene_ply: null,
      stages: {},
      workflow: { confirmed_camera_revision: 4 },
    }
    const fetchImpl = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(JSON.stringify(project), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    )
    const client = new HttpBackendClient({
      origin: 'http://127.0.0.1:49152',
      token: 'secret',
      fetchImpl,
    })

    await client.confirmCamera(4)

    expect(fetchImpl).toHaveBeenCalledWith(
      'http://127.0.0.1:49152/api/v1/projects/current/camera/confirm',
      expect.objectContaining({ method: 'POST', body: '{"camera_revision":4}' }),
    )
  })

  it('fetches opaque subject proxy and Alpha artifacts with authentication', async () => {
    const descriptor = {
      role: 'proxy',
      artifact_id: 'proxy-1',
      frame_index: 4,
      width: 960,
      height: 540,
      size: 42,
      mime_type: 'image/jpeg',
    }
    const fetchImpl = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(
        new Response(JSON.stringify(descriptor), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
      )
      .mockResolvedValueOnce(
        new Response(new Blob(['jpg'], { type: 'image/jpeg' }), {
          status: 200,
          headers: { 'Content-Type': 'image/jpeg' },
        }),
      )
    const client = new HttpBackendClient({
      origin: 'http://127.0.0.1:49152',
      token: 'secret',
      fetchImpl,
    })

    await expect(client.getSubjectMedia('proxy')).resolves.toEqual(descriptor)
    await expect(
      client.fetchSubjectMediaArtifact('proxy', 'proxy-1'),
    ).resolves.toBeInstanceOf(Blob)
    expect(String(fetchImpl.mock.calls[1]?.[0])).not.toContain('secret')
  })
})

describe('WebSocketTaskEventSource', () => {
  it('refuses to send the session token to a non-loopback WebSocket origin', () => {
    const createWebSocket = vi.fn()

    expect(
      () =>
        new WebSocketTaskEventSource({
          origin: 'https://example.com',
          token: 'memory-only-secret',
          createWebSocket,
        }),
    ).toThrow('loopback origin')
    expect(createWebSocket).not.toHaveBeenCalled()
  })

  it('authenticates before resuming and never puts the token in the URL', () => {
    let socket: FakeWebSocket | undefined
    const source = new WebSocketTaskEventSource({
      origin: 'http://127.0.0.1:49152',
      token: 'memory-only-secret',
      createWebSocket: (url) => {
        socket = new FakeWebSocket(url)
        return socket
      },
    })

    const unsubscribe = source.subscribe({
      afterRevision: 7,
      onEvent: vi.fn(),
      onConnectionChange: vi.fn(),
    })
    socket?.emit('open')

    expect(socket?.url).toBe('ws://127.0.0.1:49152/api/v1/events')
    expect(socket?.url).not.toContain('memory-only-secret')
    expect(socket?.sent.map((payload) => JSON.parse(payload))).toEqual([
      { type: 'authenticate', token: 'memory-only-secret' },
    ])

    socket?.message({ type: 'authenticated', revision: 12 })
    expect(socket?.sent.map((payload) => JSON.parse(payload))).toEqual([
      { type: 'authenticate', token: 'memory-only-secret' },
      { type: 'resume', after_revision: 7 },
    ])

    unsubscribe()
    expect(socket?.close).toHaveBeenCalledOnce()
  })

  it('reconnects from the latest accepted revision and stops after cleanup', async () => {
    vi.useFakeTimers()
    const sockets: FakeWebSocket[] = []
    const source = new WebSocketTaskEventSource({
      origin: 'http://127.0.0.1:49152',
      token: 'memory-only-secret',
      reconnectDelayMs: 50,
      createWebSocket: (url) => {
        const socket = new FakeWebSocket(url)
        sockets.push(socket)
        return socket
      },
    })
    const onEvent = vi.fn()
    const unsubscribe = source.subscribe({
      afterRevision: 3,
      onEvent,
      onConnectionChange: vi.fn(),
    })

    sockets[0]?.emit('open')
    sockets[0]?.message({ type: 'authenticated', revision: 4 })
    sockets[0]?.message({
      type: 'task_event',
      task_id: 't1',
      revision: 4,
      stage: 'segment',
      progress: 0.5,
      error: null,
    })
    sockets[0]?.emit('close')
    await vi.advanceTimersByTimeAsync(50)

    expect(sockets).toHaveLength(2)
    sockets[1]?.emit('open')
    sockets[1]?.message({ type: 'authenticated', revision: 4 })
    expect(sockets[1]?.sent.map((payload) => JSON.parse(payload))).toEqual([
      { type: 'authenticate', token: 'memory-only-secret' },
      { type: 'resume', after_revision: 4 },
    ])

    unsubscribe()
    await vi.advanceTimersByTimeAsync(50)
    expect(sockets).toHaveLength(2)
  })
})

describe('recoverable task store', () => {
  it('restores determinate progress from REST without fabricating an ETA', () => {
    const store = createTaskStore(fakeBackendClient())
    const restored: TaskDto = {
      ...task(12),
      progress: 0.4,
      current: 12,
      total: 30,
      message: '渲染背景 12/30',
      elapsed_seconds: 4.5,
      eta_seconds: null,
    }

    store.replaceFromRest(restored)

    expect(store.snapshot().latestEvent).toEqual({
      type: 'task_event',
      task_id: restored.id,
      revision: 12,
      stage: 'segment',
      progress: 0.4,
      current: 12,
      total: 30,
      message: '渲染背景 12/30',
      elapsed_seconds: 4.5,
      eta_seconds: null,
      error: null,
    })
  })

  it('does not let stale REST progress overwrite a newer websocket event', () => {
    const store = createTaskStore(fakeBackendClient())
    const initial: TaskDto = {
      ...task(10),
      progress: 0.2,
      current: 2,
      total: 10,
      message: 'render 2/10',
      elapsed_seconds: 2,
      eta_seconds: 8,
    }
    store.replaceFromRest(initial)
    store.onEvent({
      type: 'task_event',
      task_id: initial.id,
      revision: 12,
      stage: 'segment',
      progress: 0.6,
      current: 6,
      total: 10,
      message: 'render 6/10',
      elapsed_seconds: 6,
      eta_seconds: 4,
      error: null,
    })

    store.replaceFromRest({
      ...initial,
      revision: 11,
      progress: 0.4,
      current: 4,
      message: 'render 4/10',
      elapsed_seconds: 4,
      eta_seconds: 6,
    })

    expect(store.snapshot().latestEvent).toMatchObject({
      revision: 12,
      progress: 0.6,
      current: 6,
      message: 'render 6/10',
    })
  })

  it('converges through REST when the live event source observes a revision jump', async () => {
    vi.useFakeTimers()
    const sockets: FakeWebSocket[] = []
    const newerTask = { ...task(3), id: 'other-task' }
    const client = fakeBackendClient(newerTask)
    const store = createTaskStore(client, { retryDelayMs: 50 })
    const source = new WebSocketTaskEventSource({
      origin: 'http://127.0.0.1:49152',
      token: 'memory-only-secret',
      reconnectDelayMs: 100,
      createWebSocket: (url) => {
        const socket = new FakeWebSocket(url)
        sockets.push(socket)
        return socket
      },
    })
    const unsubscribe = source.subscribe({
      afterRevision: 0,
      getResumeRevision: () => store.snapshot().revision,
      onEvent: store.onEvent,
      onConnectionChange: store.onConnectionChange,
    })

    sockets[0]?.emit('open')
    sockets[0]?.message({ type: 'authenticated', revision: 3 })
    sockets[0]?.message({
      type: 'task_event',
      task_id: 't1',
      revision: 1,
      stage: 'segment',
      progress: 0.1,
      error: null,
    })
    sockets[0]?.message({
      type: 'task_event',
      task_id: 'other-task',
      revision: 3,
      stage: 'segment',
      progress: 0.5,
      error: null,
    })
    await store.whenIdle()

    expect(client.getTask).toHaveBeenCalledOnce()
    expect(client.getTask).toHaveBeenCalledWith('other-task')
    expect(store.snapshot()).toMatchObject({
      revision: 3,
      task: newerTask,
      pendingResyncRevision: null,
      latestEvent: {
        type: 'resync_required',
        task_id: 'other-task',
        revision: 3,
      },
    })
    expect(sockets[0]?.close).toHaveBeenCalledOnce()

    unsubscribe()
    expect(vi.getTimerCount()).toBe(0)
  })

  it('does not guess an old owner for an identifier-free backend resync', async () => {
    const client = fakeBackendClient(task(9))
    const store = createTaskStore(client)
    store.replaceFromRest(task(1))

    store.onEvent({ type: 'resync_required', revision: 9 })
    await store.whenIdle()

    expect(client.getTask).not.toHaveBeenCalled()
    expect(store.snapshot()).toMatchObject({ revision: 1, pendingResyncRevision: 9 })

    store.acknowledgeResync(9, task(9))
    expect(store.snapshot()).toMatchObject({
      revision: 9,
      task: task(9),
      pendingResyncRevision: null,
    })

    store.onEvent({ type: 'resync_required', revision: 10 })
    store.acknowledgeResync(10, null)
    expect(store.snapshot()).toMatchObject({
      revision: 10,
      task: null,
      pendingResyncRevision: null,
    })
  })

  it('never publishes an old owner whose REST response loses an in-flight owner race', async () => {
    let resolveOld: ((value: TaskDto) => void) | undefined
    const nextTask = { ...task(10), id: 'task-new', target_stage: 'render' as const }
    const client = fakeBackendClient(nextTask)
    vi.mocked(client.getTask).mockImplementation((id) => id === 'task-old'
      ? new Promise((resolve) => { resolveOld = resolve })
      : Promise.resolve(nextTask))
    const store = createTaskStore(client)

    store.onEvent({ type: 'resync_required', taskId: 'task-old', revision: 9 })
    await vi.waitFor(() => expect(client.getTask).toHaveBeenCalledWith('task-old'))
    store.onEvent({ type: 'resync_required', taskId: 'task-new', revision: 10 })
    resolveOld?.({ ...task(9), id: 'task-old' })
    await store.whenIdle()

    expect(client.getTask).toHaveBeenCalledWith('task-new')
    expect(store.snapshot()).toMatchObject({ task: nextTask, pendingResyncRevision: null })
  })

  it('never lets an older same-owner recovery overwrite a newer REST acknowledgement', async () => {
    let resolveRecovery: ((value: TaskDto) => void) | undefined
    const client = fakeBackendClient()
    vi.mocked(client.getTask).mockImplementation(
      () => new Promise((resolve) => { resolveRecovery = resolve }),
    )
    const store = createTaskStore(client)
    const succeeded: TaskDto = {
      id: 'task-new', target_stage: 'render', status: 'succeeded', revision: 11, error: null,
    }

    store.onEvent({ type: 'resync_required', taskId: 'task-new', revision: 10 })
    await vi.waitFor(() => expect(client.getTask).toHaveBeenCalledWith('task-new'))
    store.acknowledgeResync(11, succeeded)
    resolveRecovery?.({ ...succeeded, status: 'queued', revision: 10 })
    await store.whenIdle()

    expect(store.snapshot()).toMatchObject({ task: succeeded, revision: 11, pendingResyncRevision: null })
  })

  it('resyncs authoritative task state after an event gap', async () => {
    const client = fakeBackendClient(task(9))
    const store = createTaskStore(client)

    store.onEvent({ type: 'resync_required', taskId: 't1', revision: 9 })
    await store.whenIdle()

    expect(client.getTask).toHaveBeenCalledWith('t1')
    expect(store.snapshot().revision).toBe(9)
    expect(store.snapshot().task).toEqual(task(9))
  })

  it('deduplicates events by revision without replacing REST task state', () => {
    const store = createTaskStore(fakeBackendClient())
    const listener = vi.fn()
    store.subscribe(listener)

    store.onEvent({
      type: 'task_event',
      task_id: 't1',
      revision: 5,
      stage: 'segment',
      progress: 0.25,
      error: null,
    })
    store.onEvent({
      type: 'task_event',
      task_id: 't1',
      revision: 5,
      stage: 'segment',
      progress: 0.25,
      error: null,
    })
    store.onEvent({
      type: 'task_event',
      task_id: 't1',
      revision: 4,
      stage: 'segment',
      progress: 0.1,
      error: null,
    })

    expect(listener).toHaveBeenCalledOnce()
    expect(store.snapshot()).toMatchObject({ revision: 5, task: null })
  })

  it('changes only realtime state when the connection drops', () => {
    const store = createTaskStore(fakeBackendClient())
    store.replaceFromRest(task(9))

    store.onConnectionChange('disconnected')

    expect(store.snapshot()).toMatchObject({
      revision: 9,
      task: task(9),
      connection: 'disconnected',
    })
  })

  it('can retry REST convergence after a transient resync failure', async () => {
    vi.useFakeTimers()
    const client = fakeBackendClient(task(10))
    vi.mocked(client.getTask)
      .mockRejectedValueOnce(new Error('offline'))
      .mockResolvedValueOnce(task(10))
    const store = createTaskStore(client, { retryDelayMs: 25 })

    store.onEvent({ type: 'resync_required', taskId: 't1', revision: 9 })
    await vi.advanceTimersByTimeAsync(25)
    await store.whenIdle()

    expect(client.getTask).toHaveBeenCalledTimes(2)
    expect(store.snapshot().task).toEqual(task(10))
  })

  it('stops persistent REST recovery when its owner disposes the store', async () => {
    vi.useFakeTimers()
    const client = fakeBackendClient()
    vi.mocked(client.getTask).mockRejectedValue(new Error('offline'))
    const store = createTaskStore(client, { retryDelayMs: 25 })

    store.onEvent({ type: 'resync_required', taskId: 't1', revision: 9 })
    await vi.advanceTimersByTimeAsync(0)
    expect(client.getTask).toHaveBeenCalledOnce()
    expect(vi.getTimerCount()).toBe(1)
    const pendingRecovery = store.whenIdle()

    store.dispose()
    store.dispose()
    expect(vi.getTimerCount()).toBe(0)
    await pendingRecovery
    await store.whenIdle()

    store.onEvent({ type: 'resync_required', taskId: 't1', revision: 10 })
    await vi.advanceTimersByTimeAsync(250)
    expect(client.getTask).toHaveBeenCalledOnce()
    expect(vi.getTimerCount()).toBe(0)
  })

  it('ignores a late in-flight REST result after disposal', async () => {
    let resolveTask: ((value: TaskDto) => void) | undefined
    const client = fakeBackendClient()
    vi.mocked(client.getTask).mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveTask = resolve
        }),
    )
    const store = createTaskStore(client)

    store.onEvent({ type: 'resync_required', taskId: 't1', revision: 9 })
    const snapshotAtDisposal = store.snapshot()
    const pendingRecovery = store.whenIdle()
    store.dispose()
    await pendingRecovery

    resolveTask?.(task(9))
    await Promise.resolve()
    expect(store.snapshot()).toBe(snapshotAtDisposal)
    expect(store.snapshot()).toMatchObject({
      revision: 0,
      task: null,
      pendingResyncRevision: 9,
    })
  })

  it('closes the event subscription from React effect cleanup', () => {
    const close = vi.fn()
    const subscribe = vi.fn((_subscription: TaskEventSubscription) => close)
    const source: TaskEventSource = { subscribe }
    const store = createTaskStore(fakeBackendClient())
    const dispose = vi.spyOn(store, 'dispose')
    store.replaceFromRest(task(9))

    const { unmount } = renderHook(() => useTaskEventSource(source, store))
    expect(subscribe).toHaveBeenCalledOnce()
    expect(subscribe.mock.calls[0]?.[0].getResumeRevision?.()).toBe(9)

    act(() => unmount())
    expect(close).toHaveBeenCalledOnce()
    expect(dispose).not.toHaveBeenCalled()
  })
})
