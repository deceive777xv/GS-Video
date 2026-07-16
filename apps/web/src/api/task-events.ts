import { useEffect, useSyncExternalStore } from 'react'

import type { BackendClient } from './backend-client'
import { normalizeLocalApiOrigin } from './http-backend-client'
import type { SessionConfig, TaskDto, TaskEvent } from './types'

export type TaskEventConnection =
  | 'connecting'
  | 'authenticating'
  | 'connected'
  | 'disconnected'

export interface TaskEventSubscription {
  afterRevision: number
  onEvent(event: TaskEvent): void
  onConnectionChange(state: TaskEventConnection): void
}

export interface TaskEventSource {
  subscribe(subscription: TaskEventSubscription): () => void
}

interface WebSocketLike {
  addEventListener(type: string, listener: (event: Event) => void): void
  removeEventListener(type: string, listener: (event: Event) => void): void
  send(payload: string): void
  close(): void
}

export interface WebSocketTaskEventSourceOptions extends SessionConfig {
  createWebSocket?: (url: string) => WebSocketLike
  reconnectDelayMs?: number
}

function websocketUrl(origin: string): string {
  const url = new URL('/api/v1/events', origin)
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
  return url.toString()
}

function taskEvent(value: unknown): TaskEvent | null {
  if (typeof value !== 'object' || value === null) return null
  const candidate = value as Record<string, unknown>
  if (
    candidate.type === 'resync_required' &&
    Number.isSafeInteger(candidate.revision) &&
    (candidate.revision as number) >= 0
  ) {
    return { type: 'resync_required', revision: candidate.revision as number }
  }
  if (
    candidate.type === 'task_event' &&
    typeof candidate.task_id === 'string' &&
    Number.isSafeInteger(candidate.revision) &&
    typeof candidate.stage === 'string' &&
    typeof candidate.progress === 'number'
  ) {
    return candidate as unknown as TaskEvent
  }
  return null
}

export class WebSocketTaskEventSource implements TaskEventSource {
  readonly #origin: string
  readonly #token: string
  readonly #createWebSocket: (url: string) => WebSocketLike
  readonly #reconnectDelayMs: number

  constructor(options: WebSocketTaskEventSourceOptions) {
    this.#origin = normalizeLocalApiOrigin(options.origin)
    this.#token = options.token
    this.#createWebSocket =
      options.createWebSocket ?? ((url) => new WebSocket(url))
    this.#reconnectDelayMs = options.reconnectDelayMs ?? 1_000
  }

  subscribe(subscription: TaskEventSubscription): () => void {
    let afterRevision = subscription.afterRevision
    let stopped = false
    let socket: WebSocketLike | null = null
    let removeListeners: (() => void) | null = null
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null

    const connect = (): void => {
      if (stopped) return
      subscription.onConnectionChange('connecting')
      const current = this.#createWebSocket(websocketUrl(this.#origin))
      socket = current
      let authenticated = false
      let disconnected = false

      const onOpen = (): void => {
        subscription.onConnectionChange('authenticating')
        current.send(
          JSON.stringify({ type: 'authenticate', token: this.#token }),
        )
      }
      const onMessage = (event: Event): void => {
        const data = (event as MessageEvent<unknown>).data
        if (typeof data !== 'string') return
        let message: unknown
        try {
          message = JSON.parse(data)
        } catch {
          return
        }
        if (
          !authenticated &&
          typeof message === 'object' &&
          message !== null &&
          (message as Record<string, unknown>).type === 'authenticated'
        ) {
          authenticated = true
          current.send(
            JSON.stringify({ type: 'resume', after_revision: afterRevision }),
          )
          subscription.onConnectionChange('connected')
          return
        }
        if (!authenticated) return
        const parsed = taskEvent(message)
        if (parsed !== null) {
          afterRevision = Math.max(afterRevision, parsed.revision)
          subscription.onEvent(parsed)
        }
      }
      const onDisconnected = (): void => {
        if (disconnected || stopped) return
        disconnected = true
        removeListeners?.()
        if (socket === current) socket = null
        subscription.onConnectionChange('disconnected')
        reconnectTimer = setTimeout(connect, this.#reconnectDelayMs)
      }
      const onError = (): void => {
        onDisconnected()
        current.close()
      }

      removeListeners = () => {
        current.removeEventListener('open', onOpen)
        current.removeEventListener('message', onMessage)
        current.removeEventListener('close', onDisconnected)
        current.removeEventListener('error', onError)
      }
      current.addEventListener('open', onOpen)
      current.addEventListener('message', onMessage)
      current.addEventListener('close', onDisconnected)
      current.addEventListener('error', onError)
    }

    connect()
    return () => {
      stopped = true
      if (reconnectTimer !== null) clearTimeout(reconnectTimer)
      removeListeners?.()
      socket?.close()
      socket = null
    }
  }
}

export interface TaskStoreSnapshot {
  readonly task: TaskDto | null
  readonly revision: number
  readonly connection: TaskEventConnection
  readonly latestEvent: TaskEvent | null
}

export interface TaskStore {
  readonly subscribe: (listener: () => void) => () => void
  readonly snapshot: () => TaskStoreSnapshot
  readonly onEvent: (event: TaskEvent) => void
  readonly onConnectionChange: (state: TaskEventConnection) => void
  readonly replaceFromRest: (task: TaskDto) => void
  readonly whenIdle: () => Promise<void>
}

export function createTaskStore(client: BackendClient): TaskStore {
  const listeners = new Set<() => void>()
  let state: TaskStoreSnapshot = Object.freeze({
    task: null,
    revision: 0,
    connection: 'disconnected',
    latestEvent: null,
  })
  let idle = Promise.resolve()

  const publish = (next: TaskStoreSnapshot): void => {
    state = Object.freeze(next)
    for (const listener of listeners) listener()
  }
  const replaceFromRest = (task: TaskDto): void => {
    publish({
      ...state,
      task,
      revision: Math.max(state.revision, task.revision),
    })
  }
  const onEvent = (event: TaskEvent): void => {
    if (event.type === 'resync_required') {
      const taskId = event.taskId ?? event.task_id ?? state.task?.id
      if (taskId === undefined) return
      if (event.revision > state.revision) {
        publish({ ...state, revision: event.revision, latestEvent: event })
      }
      idle = idle
        .then(async () => {
          replaceFromRest(await client.getTask(taskId))
        })
        .catch(() => undefined)
      return
    }
    if (event.revision <= state.revision) return
    publish({ ...state, revision: event.revision, latestEvent: event })
  }

  return {
    subscribe: (listener) => {
      listeners.add(listener)
      return () => listeners.delete(listener)
    },
    snapshot: () => state,
    onEvent,
    onConnectionChange: (connection) => {
      if (connection !== state.connection) publish({ ...state, connection })
    },
    replaceFromRest,
    whenIdle: () => idle,
  }
}

export function useTaskStore(store: TaskStore): TaskStoreSnapshot {
  return useSyncExternalStore(store.subscribe, store.snapshot, store.snapshot)
}

export function useTaskEventSource(
  source: TaskEventSource,
  store: TaskStore,
): void {
  useEffect(
    () =>
      source.subscribe({
        afterRevision: store.snapshot().revision,
        onEvent: store.onEvent,
        onConnectionChange: store.onConnectionChange,
      }),
    [source, store],
  )
}
