export type AssetKind = 'source_video' | 'scene_ply'

export type StageName =
  | 'ingest'
  | 'segment'
  | 'solve_camera'
  | 'map_trajectory'
  | 'render'
  | 'composite'
  | 'export'

export type TaskStatus =
  | 'queued'
  | 'running'
  | 'succeeded'
  | 'failed'
  | 'cancelled'

export interface StageStateDto {
  status: 'pending' | 'running' | 'succeeded' | 'failed' | 'cancelled' | 'stale'
  cache_key: string | null
  output_paths: string[]
  error_code: string | null
}

export interface ProjectDto {
  schema_version: number
  project_id: string
  name: string
  created_at: string
  source_video: string | null
  scene_ply: string | null
  stages: Partial<Record<StageName, StageStateDto>>
}

export interface EnvironmentIssueDto {
  code: string
  message: string
}

export interface EnvironmentDto {
  ready: boolean
  vram_mb: number
  issues: EnvironmentIssueDto[]
  renderer_versions: Record<string, string> | null
}

export interface BootstrapDto {
  api_version: string
  capabilities: string[]
  project: ProjectDto
  environment: EnvironmentDto
}

export interface ProjectPatch {
  name?: string
}

export interface AssetDto {
  kind: AssetKind
  path: string
  size: number
  sha256: string
}

export interface UploadInit {
  filename: string
  mime_type: string
  total_size: number
  sha256: string
}

export interface UploadSessionDto {
  id: string
  chunk_size: number
}

export interface TaskDto {
  id: string
  target_stage: StageName
  status: TaskStatus
  revision: number
  error: string | null
}

export interface ErrorEnvelopeDto {
  code: string
  category: string
  message: string
  retryable: boolean
}

export interface TaskProgressEvent {
  type: 'task_event'
  task_id: string
  revision: number
  stage: StageName
  progress: number
  error: Record<string, unknown> | null
}

export interface ResyncRequiredEvent {
  type: 'resync_required'
  revision: number
  taskId?: string
  task_id?: string
}

export type TaskEvent = TaskProgressEvent | ResyncRequiredEvent

export interface SessionConfig {
  origin: string
  token: string
}
