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
  getResumeRevision?: () => number
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
    const identifier =
      typeof candidate.taskId === 'string'
        ? { taskId: candidate.taskId }
        : typeof candidate.task_id === 'string'
          ? { task_id: candidate.task_id }
          : {}
    return {
      type: 'resync_required',
      revision: candidate.revision as number,
      ...identifier,
    }
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
      let observedRevision =
        subscription.getResumeRevision?.() ?? afterRevision

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
          observedRevision =
            subscription.getResumeRevision?.() ?? afterRevision
          current.send(
            JSON.stringify({
              type: 'resume',
              after_revision: observedRevision,
            }),
          )
          subscription.onConnectionChange('connected')
          return
        }
        if (!authenticated) return
        const parsed = taskEvent(message)
        if (parsed !== null) {
          if (parsed.type === 'resync_required') {
            subscription.onEvent(parsed)
            return
          }
          if (parsed.revision <= observedRevision) return
          if (parsed.revision !== observedRevision + 1) {
            subscription.onEvent({
              type: 'resync_required',
              revision: parsed.revision,
              task_id: parsed.task_id,
            })
            onDisconnected()
            current.close()
            return
          }
          observedRevision = parsed.revision
          if (subscription.getResumeRevision === undefined) {
            afterRevision = parsed.revision
          }
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
  readonly pendingResyncRevision: number | null
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

export interface TaskStoreOptions {
  retryDelayMs?: number
}

export function createTaskStore(
  client: BackendClient,
  options: TaskStoreOptions = {},
): TaskStore {
  const listeners = new Set<() => void>()
  const retryDelayMs = options.retryDelayMs ?? 1_000
  let state: TaskStoreSnapshot = Object.freeze({
    task: null,
    revision: 0,
    pendingResyncRevision: null,
    connection: 'disconnected',
    latestEvent: null,
  })
  let trackedTaskId: string | undefined
  let recovery: Promise<void> | null = null

  const publish = (next: TaskStoreSnapshot): void => {
    state = Object.freeze(next)
    for (const listener of listeners) listener()
  }
  const replaceFromRest = (task: TaskDto): void => {
    trackedTaskId = task.id
    publish({
      ...state,
      task,
      revision: Math.max(state.revision, task.revision),
    })
  }
  const retryDelay = (): Promise<void> =>
    new Promise((resolve) => setTimeout(resolve, retryDelayMs))
  const recover = async (): Promise<void> => {
    while (
      state.pendingResyncRevision !== null &&
      trackedTaskId !== undefined
    ) {
      const recoveringRevision = state.pendingResyncRevision
      try {
        const task = await client.getTask(trackedTaskId)
        const stillPending =
          state.pendingResyncRevision > recoveringRevision
            ? state.pendingResyncRevision
            : null
        publish({
          ...state,
          task,
          revision: Math.max(
            state.revision,
            task.revision,
            recoveringRevision,
          ),
          pendingResyncRevision: stillPending,
        })
      } catch {
        await retryDelay()
      }
    }
  }
  const startRecovery = (): void => {
    if (recovery !== null || trackedTaskId === undefined) return
    recovery = recover().finally(() => {
      recovery = null
      if (state.pendingResyncRevision !== null) startRecovery()
    })
  }
  const onEvent = (event: TaskEvent): void => {
    if (event.type === 'resync_required') {
      trackedTaskId =
        state.task?.id ??
        trackedTaskId ??
        event.taskId ??
        event.task_id
      const pendingResyncRevision = Math.max(
        state.pendingResyncRevision ?? 0,
        event.revision,
      )
      publish({ ...state, pendingResyncRevision, latestEvent: event })
      startRecovery()
      return
    }
    if (event.revision <= state.revision) return
    trackedTaskId ??= event.task_id
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
    whenIdle: () => recovery ?? Promise.resolve(),
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
        getResumeRevision: () => store.snapshot().revision,
        onEvent: store.onEvent,
        onConnectionChange: store.onConnectionChange,
      }),
    [source, store],
  )
}
