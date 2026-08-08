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

export type ArtifactRole = 'proxy_frames' | 'subject_masks' | 'export_video'

export interface StageStateDto {
  status: 'pending' | 'running' | 'succeeded' | 'failed' | 'cancelled' | 'stale'
  cache_key: string | null
  output_paths: string[]
  error_code: string | null
  artifacts: Partial<Record<ArtifactRole, string>>
}

export interface ProjectDto {
  schema_version: number
  project_id: string
  name: string
  created_at: string
  source_video: string | null
  scene_ply: string | null
  stages: Partial<Record<StageName, StageStateDto>>
  workflow: WorkflowDto
}

export interface VideoSummaryDto {
  filename: string
  size: number
  sha256: string
  width: number
  height: number
  duration_seconds: number
  fps: string
  has_audio: boolean
  frame_count: number | null
}

export interface SceneSummaryDto {
  filename: string
  size: number
  sha256: string
  gaussian_count: number
  estimated_vram_mb: number
}

export interface SubjectPromptDto {
  frame_index: number
  x: number
  y: number
}

export interface CameraInput {
  target: [number, number, number]
  distance: number
  yaw: number
  pitch: number
  fov_y_degrees: number
}

export interface CameraDto extends CameraInput {
  revision: number
}

export interface FootPointDto {
  image: [number, number]
  world: [number, number, number]
  preview_artifact_id: string
  camera_revision: number
  pick_buffer_revision: number
}

export interface PreviewDto {
  artifact_id: string
  artifact_size: number
  artifact_sha256: string
  generation: number
  width: number
  height: number
  camera_revision: number
  pick_buffer_revision: number
}

export interface ExportResultDto {
  artifact_id: string
  filename: string
  size: number
  sha256: string
  duration_seconds: number
  fps: string
  frame_count: number
  has_audio: boolean
  verified: boolean
}

export interface WorkflowDto {
  source_summary: VideoSummaryDto | null
  scene_summary: SceneSummaryDto | null
  subject_prompt: SubjectPromptDto | null
  target_camera: CameraDto | null
  preview_epoch: number
  confirmed_camera_revision: number | null
  confirmed_preview_artifact_id: string | null
  foot_point: FootPointDto | null
  motion_scale: number
  preview_height: number
  active_task_id: string | null
  preview: PreviewDto | null
  export_result: ExportResultDto | null
}

export type VerifiedExportDto = Omit<
  ExportResultDto,
  'sha256'
>

export interface CompositePreviewDto {
  artifact_id: string
  filename: 'composite-preview.mp4'
  size: number
  sha256: string
  duration_seconds: number
  fps: string
  frame_count: number
}

export type SubjectMediaRole = 'proxy' | 'alpha'

export interface SubjectMediaDto {
  role: SubjectMediaRole
  artifact_id: string
  frame_index: number
  width: number
  height: number
  size: number
  mime_type: 'image/jpeg' | 'image/png'
}

export interface EnvironmentIssueDto {
  code: string
  message: string
}

export interface EnvironmentDto {
  ready: boolean
  vram_mb: number
  vram_limit_mb: number
  issues: EnvironmentIssueDto[]
  renderer_versions: Record<string, string> | null
}

export type VramBudgetMode = 'standard' | 'custom'

export interface VramBudgetDto {
  mode: VramBudgetMode
  minimum_vram_mb: number
  total_vram_mb: number
  selected_vram_mb: number
  editable: boolean
  blocked_reason: string | null
  recovered_from_invalid_preference: boolean
}

export type VramBudgetUpdate =
  | { mode: 'standard'; selected_vram_mb: null }
  | { mode: 'custom'; selected_vram_mb: number }

export interface BootstrapDto {
  api_version: string
  capabilities: string[]
  project: ProjectDto
  environment: EnvironmentDto
  vram_budget: VramBudgetDto
}

export type EnvironmentRepairState =
  | 'idle'
  | 'running'
  | 'cancelling'
  | 'cancelled'
  | 'succeeded'
  | 'failed'

export interface EnvironmentRepairErrorDto {
  code: string
  message: string
  retryable: boolean
}

export interface EnvironmentRepairSnapshotDto {
  state: EnvironmentRepairState
  job_id: string | null
  step: string | null
  resource_id: string | null
  resource_name: string | null
  progress: number
  downloaded_bytes: number
  total_bytes: number | null
  message: string | null
  resume_available: boolean
  restart_required: boolean
  error: EnvironmentRepairErrorDto | null
  environment: EnvironmentDto | null
}

export interface ProjectPatch {
  name?: string
  subject_prompt?: SubjectPromptDto | null
  motion_scale?: number
  preview_height?: number
}

export interface AssetDto {
  kind: AssetKind
  path: string
  size: number
  sha256: string
}

export interface UploadInit {
  kind: AssetKind
  filename: string
  mime_type: string
  total_size: number
  sha256: string
}

export interface UploadSessionDto {
  id: string
  chunk_size: number
}

export interface UploadStatusDto extends UploadSessionDto {
  kind: AssetKind
  filename: string
  total_size: number
  uploaded_chunks: number[]
}

export interface UploadCompleteDto {
  path: string
  kind: AssetKind
  size: number
  sha256: string
}

export interface PreviewRequest {
  generation: number
  width: number
  height: number
  camera: CameraInput
}

export interface LivePreviewRequest {
  request_id: number
  width: number
  height: number
  camera: CameraInput
}

export interface PreviewFrameDto {
  artifact_id: string
  generation: number
  width: number
  height: number
  camera_revision: number
  pick_buffer_revision: number
}

export interface PickRequest {
  x: number
  y: number
  preview_artifact_id: string
  camera_revision: number
  pick_buffer_revision: number
}

export interface TaskDto {
  id: string
  target_stage: StageName
  status: TaskStatus
  revision: number
  error: string | null
  progress?: number
  current?: number | null
  total?: number | null
  message?: string | null
  elapsed_seconds?: number
  eta_seconds?: number | null
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
  current?: number | null
  total?: number | null
  message?: string | null
  elapsed_seconds?: number
  eta_seconds?: number | null
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
