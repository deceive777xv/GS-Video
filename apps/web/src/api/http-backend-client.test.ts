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
import type { TaskDto } from './types'

const task = (revision = 9): TaskDto => ({
  id: 't1',
  target_stage: 'segment',
  revision,
  status: 'running',
  error: null,
})

const fakeBackendClient = (currentTask = task()): BackendClient => ({
  bootstrap: vi.fn(),
  importLocalPath: vi.fn(),
  createUpload: vi.fn(),
  putUploadChunk: vi.fn(),
  getProject: vi.fn(),
  updateProject: vi.fn(),
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
  it('converges through REST when the live event source observes a revision jump', async () => {
    vi.useFakeTimers()
    const sockets: FakeWebSocket[] = []
    const client = fakeBackendClient(task(3))
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
    expect(client.getTask).toHaveBeenCalledWith('t1')
    expect(store.snapshot()).toMatchObject({
      revision: 3,
      task: task(3),
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

  it('retries one backend-shaped resync without acknowledging it before REST succeeds', async () => {
    vi.useFakeTimers()
    const sockets: FakeWebSocket[] = []
    const client = fakeBackendClient(task(9))
    vi.mocked(client.getTask)
      .mockRejectedValueOnce(new Error('offline'))
      .mockResolvedValueOnce(task(9))
    const store = createTaskStore(client, { retryDelayMs: 50 })
    const source = new WebSocketTaskEventSource({
      origin: 'http://127.0.0.1:49152',
      token: 'memory-only-secret',
      reconnectDelayMs: 10,
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
    sockets[0]?.message({ type: 'authenticated', revision: 9 })
    sockets[0]?.message({
      type: 'task_event',
      task_id: 't1',
      revision: 1,
      stage: 'segment',
      progress: 0.1,
      error: null,
    })
    sockets[0]?.message({ type: 'resync_required', revision: 9 })
    await vi.waitFor(() => expect(client.getTask).toHaveBeenCalledOnce())

    expect(store.snapshot()).toMatchObject({
      revision: 1,
      pendingResyncRevision: 9,
    })
    sockets[0]?.emit('close')
    await vi.advanceTimersByTimeAsync(10)
    sockets[1]?.emit('open')
    sockets[1]?.message({ type: 'authenticated', revision: 9 })
    expect(sockets[1]?.sent.map((payload) => JSON.parse(payload))).toEqual([
      { type: 'authenticate', token: 'memory-only-secret' },
      { type: 'resume', after_revision: 1 },
    ])

    await vi.advanceTimersByTimeAsync(40)
    await store.whenIdle()
    expect(client.getTask).toHaveBeenCalledTimes(2)
    expect(store.snapshot()).toMatchObject({
      revision: 9,
      task: task(9),
      pendingResyncRevision: null,
    })

    unsubscribe()
    expect(vi.getTimerCount()).toBe(0)
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

  it('closes the event subscription from React effect cleanup', () => {
    const close = vi.fn()
    const subscribe = vi.fn((_subscription: TaskEventSubscription) => close)
    const source: TaskEventSource = { subscribe }
    const store = createTaskStore(fakeBackendClient())
    store.replaceFromRest(task(9))

    const { unmount } = renderHook(() => useTaskEventSource(source, store))
    expect(subscribe).toHaveBeenCalledOnce()
    expect(subscribe.mock.calls[0]?.[0].getResumeRevision?.()).toBe(9)

    act(() => unmount())
    expect(close).toHaveBeenCalledOnce()
  })
})
