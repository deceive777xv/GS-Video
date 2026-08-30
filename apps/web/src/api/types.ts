export type AssetKind = 'source_video' | 'scene_ply' | 'lut_3d'

export type StageName =
  | 'ingest'
  | 'segment'
  | 'solve_camera'
  | 'map_trajectory'
  | 'render'
  | 'composite'
  | 'post_process'
  | 'export'

export type TaskStatus =
  | 'queued'
  | 'running'
  | 'succeeded'
  | 'failed'
  | 'cancelled'

export type ArtifactRole =
  | 'proxy_frames'
  | 'subject_masks'
  | 'composite_frames'
  | 'post_process_frames'
  | 'post_process_preview'
  | 'export_video'

export interface ArtifactRefDto {
  project_id: string
  category: 'frames' | 'proxies' | 'masks' | 'camera' | 'trajectories' | 'renders' | 'composites' | 'post_processes' | 'previews' | 'exports'
  cache_key: string
  member: string | null
}

export interface StageStateDto {
  status: 'pending' | 'running' | 'succeeded' | 'failed' | 'cancelled' | 'stale'
  cache_key: string | null
  output_paths: ArtifactRefDto[]
  error_code: string | null
  artifacts: Partial<Record<ArtifactRole, ArtifactRefDto>>
}

export interface ProjectDto {
  schema_version: number
  project_id: string
  name: string
  created_at: string
  updated_at?: string
  source_video_asset_id?: string | null
  scene_ply_asset_id?: string | null
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
  color_primaries?: string | null
  color_transfer?: string | null
  color_matrix?: string | null
  color_range?: string | null
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

export type Matrix4 = [
  [number, number, number, number],
  [number, number, number, number],
  [number, number, number, number],
  [number, number, number, number],
]

export interface MatrixCameraInput {
  camera_to_world: Matrix4
  fov_y_degrees: number
}

export interface ExplorationCameraDto extends MatrixCameraInput {
  revision: number
}

export interface TargetGroundDto {
  scene_asset_id: string
  hint_pixels: [[number, number], [number, number], [number, number]]
  p0_world: [number, number, number]
  p1_world: [number, number, number]
  p2_world: [number, number, number]
  plane_normal: [number, number, number]
  plane_offset: number
  exploration_camera_to_world: Matrix4
  camera_fingerprint: string
  preview_artifact_id: string
  camera_revision: number
  pick_buffer_revision: number
  support_counts: [number, number, number]
  weighted_inlier_ratio: number
  rms_residual: number
  confidence: number
  revision: number
  confirmed: boolean
}

export interface OutputCropDto {
  x: number
  y: number
  width: number
  height: number
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

export interface MatteRefinementSettings {
  enabled: boolean
  edge_offset: number
  feather_radius: number
  decontaminate_strength: number
  decontaminate_radius: number
}

export interface PrimaryCorrectionParameters {
  exposure: number
  contrast: number
  highlights: number
  shadows: number
  temperature: number
  tint: number
  saturation: number
  vibrance: number
}

export interface Lut3DParameters { asset_id: string }
export interface BloomParameters {
  threshold: number
  soft_knee: number
  radius: number
  intensity: number
  tint: [number, number, number]
}
export interface VignetteParameters {
  amount: number
  midpoint: number
  feather: number
  roundness: number
  center_x: number
  center_y: number
}
export interface SharpenParameters {
  amount: number
  radius: number
  threshold: number
}

interface EffectInstanceBase {
  instance_id: string
  params_version: 1
  display_name: string | null
  enabled: boolean
  mix: number
}

export type EffectInstance =
  | (EffectInstanceBase & { type: 'primary_correction'; parameters: PrimaryCorrectionParameters })
  | (EffectInstanceBase & { type: 'lut_3d'; parameters: Lut3DParameters })
  | (EffectInstanceBase & { type: 'bloom'; parameters: BloomParameters })
  | (EffectInstanceBase & { type: 'vignette'; parameters: VignetteParameters })
  | (EffectInstanceBase & { type: 'sharpen'; parameters: SharpenParameters })

export interface ExportEncodingSettings {
  codec: 'h264' | 'h265'
  rate_control: 'constant_quality' | 'two_pass_vbr'
  quality: number
  target_bitrate_mbps: number
  compression_preset: 'fast' | 'balanced' | 'high_compression'
}

export interface WorkflowDto {
  source_summary: VideoSummaryDto | null
  scene_summary: SceneSummaryDto | null
  subject_prompt: SubjectPromptDto | null
  target_camera: CameraDto | null
  exploration_camera: ExplorationCameraDto | null
  preview_epoch: number
  target_ground: TargetGroundDto | null
  gs_scale: number
  scene_azimuth: number
  output_crop: OutputCropDto | null
  preview_height: number
  source_color_interpretation: 'rec709_metadata' | 'assumed_rec709' | null
  matte_refinement: MatteRefinementSettings
  effect_chain: EffectInstance[]
  effect_chain_revision: number
  export_settings: ExportEncodingSettings
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
  filename: 'post-process-preview.mp4'
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

export interface StorageLayoutStatusDto {
  project_library_root: string
  project_library_id: string
  cache_root: string
  cache_id: string
  restart_required: boolean
  editable: boolean
  blocked_reason: string | null
}

export interface StorageLayoutDto extends StorageLayoutStatusDto {
  project_library_bytes: number
  project_library_free_bytes: number
  cache_bytes: number
  cache_free_bytes: number
}

export interface CacheCleanupResultDto {
  mode: 'safe' | 'deep'
  removed_entries: number
  freed_bytes: number
  storage: StorageLayoutDto
}

export interface CacheCleanupPlanDto {
  plan_token: string
  mode: 'safe' | 'deep'
  removable_entries: number
  reclaimable_bytes: number
  expires_in_seconds: number
}

export interface StorageLayoutUpdate {
  project_library_root: string
  cache_root: string
  project_action: 'migrate' | 'open_existing'
  cache_action: 'start_fresh' | 'migrate'
}

export interface BootstrapDto {
  api_version: string
  capabilities: string[]
  project: ProjectDto | null
  projects?: ProjectSummaryDto[]
  asset_counts?: Partial<Record<LibraryAssetKind, number>>
  environment: EnvironmentDto
  vram_budget: VramBudgetDto
  storage_layout?: StorageLayoutStatusDto | null
}

export type LibraryAssetKind = 'video' | 'ply' | 'lut'

export type WorkflowStepName = 'import' | 'subject' | 'camera' | 'preview' | 'postprocess' | 'export'

export interface ProjectSummaryDto {
  project_id: string
  name: string
  created_at: string
  updated_at: string
  workflow_step: WorkflowStepName
  active_task_id: string | null
}

export interface LibraryAssetRecordDto {
  asset_id: string
  kind: LibraryAssetKind
  original_filename: string
  stored_relative_path: string
  size: number
  sha256: string
  imported_at: string
  video_summary: VideoSummaryDto | null
  scene_summary: SceneSummaryDto | null
  lut_summary?: {
    filename: string
    size: number
    sha256: string
    lut_size: number
    domain_min: [number, number, number]
    domain_max: [number, number, number]
  } | null
}

export interface AssetListItemDto {
  asset: LibraryAssetRecordDto
  references: ProjectSummaryDto[]
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
  expected_project_id?: string
  expected_ingest_cache_key?: string | null
  name?: string
  subject_prompt?: SubjectPromptDto | null
  gs_scale?: number
  scene_azimuth?: number
  output_crop?: OutputCropDto | null
  preview_height?: number
  source_color_interpretation?: 'rec709_metadata' | 'assumed_rec709'
  matte_refinement?: MatteRefinementSettings
  effect_chain?: EffectInstance[]
  expected_effect_chain_revision?: number
  export_settings?: ExportEncodingSettings
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
  assign_to_current?: boolean
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
  filename?: string
  kind: AssetKind
  size: number
  sha256: string
  assign_to_current?: boolean
}

export interface PreviewRequest {
  expected_project_id: string
  generation: number
  width: number
  height: number
  camera: CameraInput | MatrixCameraInput
}

export interface LivePreviewRequest {
  expected_project_id: string
  request_id: number
  width: number
  height: number
  camera: CameraInput | MatrixCameraInput
}

export interface DraftCompositePreviewRequest {
  expected_project_id: string
  request_id: number
  frame_index?: number
  maximum_width: number
  maximum_height: number
  gs_scale: number
  scene_azimuth: number
  output_crop: OutputCropDto
  matte_refinement: MatteRefinementSettings
}

export interface DraftPostProcessPreviewRequest {
  expected_project_id: string
  request_id: number
  frame_index: number
  maximum_width: number
  maximum_height: number
  effect_chain: EffectInstance[]
  bypass: boolean
}

export interface TargetGroundCandidateInput {
  expected_project_id: string
  preview_artifact_id: string
  camera_revision: number
  pick_buffer_revision: number
  hints: [[number, number], [number, number], [number, number]]
}

export interface PreviewFrameDto {
  artifact_id: string
  generation: number
  width: number
  height: number
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
