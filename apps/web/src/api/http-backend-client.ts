import type { BackendClient } from './backend-client'
import type {
  AssetDto,
  AssetKind,
  BootstrapDto,
  ErrorEnvelopeDto,
  ProjectDto,
  ProjectPatch,
  PreviewFrameDto,
  LivePreviewRequest,
  PreviewRequest,
  PickRequest,
  FootPointDto,
  SessionConfig,
  StageName,
  TaskDto,
  UploadInit,
  UploadSessionDto,
  UploadStatusDto,
  UploadCompleteDto,
  VerifiedExportDto,
  CompositePreviewDto,
  SubjectMediaDto,
  SubjectMediaRole,
  EnvironmentRepairSnapshotDto,
  VramBudgetDto,
  VramBudgetUpdate,
} from './types'

const API_PREFIX = '/api/v1'
const DEFAULT_TIMEOUT_MS = 15_000

export class BackendClientError extends Error {
  readonly status: number
  readonly code: string
  readonly category: string
  readonly retryable: boolean

  constructor(status: number, envelope: ErrorEnvelopeDto) {
    super(envelope.message)
    this.name = 'BackendClientError'
    this.status = status
    this.code = envelope.code
    this.category = envelope.category
    this.retryable = envelope.retryable
  }
}

export interface HttpBackendClientOptions extends SessionConfig {
  timeoutMs?: number
  fetchImpl?: typeof fetch
}

interface RequestOptions {
  method?: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE'
  json?: unknown
  body?: BodyInit
  signal?: AbortSignal
  response?: 'json' | 'blob'
}

export function normalizeLocalApiOrigin(value: string): string {
  const url = new URL(value)
  const hostname = url.hostname.replace(/^\[|\]$/g, '')
  if (
    (url.protocol !== 'http:' && url.protocol !== 'https:') ||
    !['127.0.0.1', 'localhost', '::1'].includes(hostname) ||
    url.username !== '' ||
    url.password !== '' ||
    (url.pathname !== '' && url.pathname !== '/') ||
    url.search !== '' ||
    url.hash !== ''
  ) {
    throw new TypeError('The API origin must be an HTTP loopback origin')
  }
  return url.origin
}

function isErrorEnvelope(value: unknown): value is ErrorEnvelopeDto {
  if (typeof value !== 'object' || value === null) return false
  const candidate = value as Record<string, unknown>
  return (
    typeof candidate.code === 'string' &&
    typeof candidate.category === 'string' &&
    typeof candidate.message === 'string' &&
    typeof candidate.retryable === 'boolean'
  )
}

function transportError(
  code: 'request_aborted' | 'request_timeout' | 'network_error',
): BackendClientError {
  const messages = {
    request_aborted: 'The local service request was cancelled.',
    request_timeout: 'The local service request timed out.',
    network_error: 'The local service could not be reached.',
  } as const
  return new BackendClientError(0, {
    code,
    category: 'transport',
    message: messages[code],
    retryable: code !== 'request_aborted',
  })
}

export class HttpBackendClient implements BackendClient {
  readonly #origin: string
  readonly #token: string
  readonly #timeoutMs: number
  readonly #fetch: typeof fetch

  constructor(options: HttpBackendClientOptions) {
    this.#origin = normalizeLocalApiOrigin(options.origin)
    this.#token = options.token
    this.#timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS
    this.#fetch = options.fetchImpl ?? globalThis.fetch.bind(globalThis)
    if (this.#token.length === 0) throw new TypeError('A session token is required')
    if (!Number.isFinite(this.#timeoutMs) || this.#timeoutMs <= 0) {
      throw new TypeError('timeoutMs must be greater than zero')
    }
  }

  async #request<T>(path: string, options: RequestOptions = {}): Promise<T> {
    const controller = new AbortController()
    let timedOut = false
    const abortFromCaller = (): void => controller.abort(options.signal?.reason)
    if (options.signal?.aborted === true) abortFromCaller()
    else options.signal?.addEventListener('abort', abortFromCaller, { once: true })
    const timeout = setTimeout(() => {
      timedOut = true
      controller.abort()
    }, this.#timeoutMs)

    const headers = new Headers({
      Accept: 'application/json',
      Authorization: `Bearer ${this.#token}`,
    })
    let body = options.body
    if (options.json !== undefined) {
      headers.set('Content-Type', 'application/json')
      body = JSON.stringify(options.json)
    } else if (body instanceof Blob) {
      headers.set('Content-Type', 'application/octet-stream')
    }

    try {
      const requestInit: RequestInit = {
        method: options.method ?? 'GET',
        headers,
        signal: controller.signal,
      }
      if (body !== undefined) requestInit.body = body
      const response = await this.#fetch(
        `${this.#origin}${API_PREFIX}${path}`,
        requestInit,
      )
      if (!response.ok) {
        let payload: unknown
        try {
          payload = await response.json()
        } catch {
          payload = undefined
        }
        if (isErrorEnvelope(payload)) {
          throw new BackendClientError(response.status, payload)
        }
        throw new BackendClientError(response.status, {
          code: 'http_error',
          category: 'transport',
          message: 'The local service request failed.',
          retryable: response.status >= 500,
        })
      }
      if (response.status === 204) return undefined as T
      if (options.response === 'blob') return (await response.blob()) as T
      return (await response.json()) as T
    } catch (error) {
      if (error instanceof BackendClientError) throw error
      if (timedOut) throw transportError('request_timeout')
      if (options.signal?.aborted === true) throw transportError('request_aborted')
      throw transportError('network_error')
    } finally {
      clearTimeout(timeout)
      options.signal?.removeEventListener('abort', abortFromCaller)
    }
  }

  bootstrap(signal?: AbortSignal): Promise<BootstrapDto> {
    return signal === undefined
      ? this.#request('/bootstrap')
      : this.#request('/bootstrap', { signal })
  }

  getVramBudget(): Promise<VramBudgetDto> {
    return this.#request('/runtime/vram-budget')
  }

  updateVramBudget(input: VramBudgetUpdate): Promise<VramBudgetDto> {
    return this.#request('/runtime/vram-budget', { method: 'PATCH', json: input })
  }

  getEnvironmentRepair(): Promise<EnvironmentRepairSnapshotDto> {
    return this.#request('/environment/repair')
  }

  startEnvironmentRepair(): Promise<EnvironmentRepairSnapshotDto> {
    return this.#request('/environment/repair', { method: 'POST' })
  }

  cancelEnvironmentRepair(): Promise<EnvironmentRepairSnapshotDto> {
    return this.#request('/environment/repair', { method: 'DELETE' })
  }

  importLocalPath(kind: AssetKind, path: string): Promise<AssetDto> {
    return this.#request('/assets/import', {
      method: 'POST',
      json: { kind, path },
    })
  }

  createUpload(input: UploadInit): Promise<UploadSessionDto> {
    return this.#request('/uploads', { method: 'POST', json: input })
  }

  putUploadChunk(
    id: string,
    index: number,
    data: Blob,
    signal?: AbortSignal,
  ): Promise<void> {
    const options: RequestOptions = { method: 'PUT', body: data }
    if (signal !== undefined) options.signal = signal
    return this.#request(
      `/uploads/${encodeURIComponent(id)}/chunks/${String(index)}`,
      options,
    )
  }

  getUpload(id: string): Promise<UploadStatusDto> {
    return this.#request(`/uploads/${encodeURIComponent(id)}`)
  }

  completeUpload(id: string): Promise<UploadCompleteDto> {
    return this.#request(`/uploads/${encodeURIComponent(id)}/complete`, {
      method: 'POST',
    })
  }

  cancelUpload(id: string): Promise<void> {
    return this.#request(`/uploads/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    })
  }

  getProject(): Promise<ProjectDto> {
    return this.#request('/projects/current')
  }

  updateProject(patch: ProjectPatch): Promise<ProjectDto> {
    return this.#request('/projects/current', { method: 'PATCH', json: patch })
  }

  renderPreview(
    input: PreviewRequest,
    signal?: AbortSignal,
  ): Promise<PreviewFrameDto> {
    const options: RequestOptions = { method: 'POST', json: input }
    if (signal !== undefined) options.signal = signal
    return this.#request('/projects/current/preview', options)
  }

  renderLivePreview(input: LivePreviewRequest, signal?: AbortSignal): Promise<Blob> {
    const options: RequestOptions = { method: 'POST', json: input, response: 'blob' }
    if (signal !== undefined) options.signal = signal
    return this.#request('/projects/current/preview/live', options)
  }

  closeLivePreview(): Promise<void> {
    return this.#request('/projects/current/preview/live', { method: 'DELETE' })
  }

  fetchPreviewArtifact(id: string, signal?: AbortSignal): Promise<Blob> {
    const options: RequestOptions = { response: 'blob' }
    if (signal !== undefined) options.signal = signal
    return this.#request(
      `/projects/current/previews/${encodeURIComponent(id)}`,
      options,
    )
  }

  pickFootPoint(input: PickRequest): Promise<FootPointDto> {
    return this.#request('/projects/current/pick', {
      method: 'POST',
      json: input,
    })
  }

  confirmCamera(cameraRevision: number): Promise<ProjectDto> {
    return this.#request('/projects/current/camera/confirm', {
      method: 'POST',
      json: { camera_revision: cameraRevision },
    })
  }

  getVerifiedExport(): Promise<VerifiedExportDto> {
    return this.#request('/projects/current/export')
  }

  fetchExportArtifact(id: string, signal?: AbortSignal): Promise<Blob> {
    const options: RequestOptions = { response: 'blob' }
    if (signal !== undefined) options.signal = signal
    return this.#request(
      `/projects/current/exports/${encodeURIComponent(id)}`,
      options,
    )
  }

  getCompositePreview(signal?: AbortSignal): Promise<CompositePreviewDto> {
    return signal === undefined
      ? this.#request('/projects/current/composite-preview')
      : this.#request('/projects/current/composite-preview', { signal })
  }

  fetchCompositePreviewArtifact(id: string, signal?: AbortSignal): Promise<Blob> {
    const options: RequestOptions = { response: 'blob' }
    if (signal !== undefined) options.signal = signal
    return this.#request(
      `/artifacts/composite-previews/${encodeURIComponent(id)}`,
      options,
    )
  }

  copyVerifiedExport(id: string, destination: string): Promise<void> {
    return this.#request(
      `/projects/current/exports/${encodeURIComponent(id)}/copy`,
      { method: 'POST', json: { destination } },
    )
  }

  getSubjectMedia(role: SubjectMediaRole): Promise<SubjectMediaDto> {
    return this.#request(
      `/projects/current/subject-media/${encodeURIComponent(role)}`,
    )
  }

  fetchSubjectMediaArtifact(
    role: SubjectMediaRole,
    id: string,
    signal?: AbortSignal,
  ): Promise<Blob> {
    const options: RequestOptions = { response: 'blob' }
    if (signal !== undefined) options.signal = signal
    return this.#request(
      `/projects/current/subject-media/${encodeURIComponent(role)}/${encodeURIComponent(id)}`,
      options,
    )
  }

  startTask(targetStage: StageName): Promise<TaskDto> {
    return this.#request('/tasks', {
      method: 'POST',
      json: { target_stage: targetStage },
    })
  }

  getTask(id: string): Promise<TaskDto> {
    return this.#request(`/tasks/${encodeURIComponent(id)}`)
  }

  cancelTask(id: string): Promise<TaskDto> {
    return this.#request(`/tasks/${encodeURIComponent(id)}`, { method: 'DELETE' })
  }
}
