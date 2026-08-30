import type {
  AssetDto,
  AssetListItemDto,
  AssetKind,
  BootstrapDto,
  ProjectDto,
  ProjectSummaryDto,
  ProjectPatch,
  PreviewFrameDto,
  LivePreviewRequest,
  DraftCompositePreviewRequest,
  DraftPostProcessPreviewRequest,
  PreviewRequest,
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
  EnvironmentDto,
  VramBudgetDto,
  VramBudgetUpdate,
  StorageLayoutDto,
  StorageLayoutUpdate,
  CacheCleanupResultDto,
  CacheCleanupPlanDto,
  TargetGroundCandidateInput,
} from './types'

export interface BackendClient {
  bootstrap(signal?: AbortSignal): Promise<BootstrapDto>
  refreshEnvironment(): Promise<EnvironmentDto>
  getVramBudget(): Promise<VramBudgetDto>
  updateVramBudget(input: VramBudgetUpdate): Promise<VramBudgetDto>
  getStorageLayout(): Promise<StorageLayoutDto>
  updateStorageLayout(input: StorageLayoutUpdate): Promise<StorageLayoutDto>
  planStorageCacheCleanup(mode: 'safe' | 'deep'): Promise<CacheCleanupPlanDto>
  cleanupStorageCache(mode: 'safe' | 'deep', planToken: string): Promise<CacheCleanupResultDto>
  getEnvironmentRepair(): Promise<EnvironmentRepairSnapshotDto>
  startEnvironmentRepair(): Promise<EnvironmentRepairSnapshotDto>
  cancelEnvironmentRepair(): Promise<EnvironmentRepairSnapshotDto>
  listProjects(): Promise<ProjectSummaryDto[]>
  createProject(name: string): Promise<ProjectDto>
  activateProject(id: string): Promise<ProjectDto>
  renameProject(id: string, name: string): Promise<ProjectSummaryDto>
  deleteProject(id: string): Promise<void>
  listAssets(kind: 'video' | 'ply' | 'lut'): Promise<AssetListItemDto[]>
  deleteAsset(id: string): Promise<void>
  selectProjectAsset(kind: AssetKind, assetId: string | null, expectedProjectId: string): Promise<ProjectDto>
  importLocalPath(kind: AssetKind, path: string, assignToCurrent?: boolean): Promise<AssetDto>
  createUpload(input: UploadInit): Promise<UploadSessionDto>
  putUploadChunk(
    id: string,
    index: number,
    data: Blob,
    signal?: AbortSignal,
  ): Promise<void>
  getUpload(id: string): Promise<UploadStatusDto>
  completeUpload(id: string): Promise<UploadCompleteDto>
  cancelUpload(id: string): Promise<void>
  getProject(): Promise<ProjectDto>
  updateProject(patch: ProjectPatch): Promise<ProjectDto>
  renderPreview(
    input: PreviewRequest,
    signal?: AbortSignal,
  ): Promise<PreviewFrameDto>
  renderLivePreview(input: LivePreviewRequest, signal?: AbortSignal): Promise<Blob>
  renderDraftCompositePreview(
    input: DraftCompositePreviewRequest,
    signal?: AbortSignal,
  ): Promise<Blob>
  renderDraftPostProcessPreview(
    input: DraftPostProcessPreviewRequest,
    signal?: AbortSignal,
  ): Promise<Blob>
  closePostProcessPreview(): Promise<void>
  closeLivePreview(): Promise<void>
  fetchPreviewArtifact(id: string, signal?: AbortSignal): Promise<Blob>
  fitTargetGround(input: TargetGroundCandidateInput): Promise<ProjectDto>
  confirmTargetGround(expectedProjectId: string, targetGroundRevision: number): Promise<ProjectDto>
  getVerifiedExport(): Promise<VerifiedExportDto>
  fetchExportArtifact(id: string, signal?: AbortSignal): Promise<Blob>
  getCompositePreview(signal?: AbortSignal): Promise<CompositePreviewDto>
  fetchCompositePreviewArtifact(id: string, signal?: AbortSignal): Promise<Blob>
  copyVerifiedExport(id: string, destination: string): Promise<void>
  getSubjectMedia(role: SubjectMediaRole, frameIndex?: number): Promise<SubjectMediaDto>
  fetchSubjectMediaArtifact(
    role: SubjectMediaRole,
    id: string,
    signal?: AbortSignal,
    frameIndex?: number,
  ): Promise<Blob>
  startTask(targetStage: StageName, expectedProjectId: string): Promise<TaskDto>
  getTask(id: string): Promise<TaskDto>
  cancelTask(id: string): Promise<TaskDto>
}
