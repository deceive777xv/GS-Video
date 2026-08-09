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
  PreviewRequest,
  PickRequest,
  FootPointDto,
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
  StorageLayoutDto,
  StorageLayoutUpdate,
  CacheCleanupResultDto,
  CacheCleanupPlanDto,
} from './types'

export interface BackendClient {
  bootstrap(signal?: AbortSignal): Promise<BootstrapDto>
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
  listAssets(kind: 'video' | 'ply'): Promise<AssetListItemDto[]>
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
  closeLivePreview(): Promise<void>
  fetchPreviewArtifact(id: string, signal?: AbortSignal): Promise<Blob>
  pickFootPoint(input: PickRequest): Promise<FootPointDto>
  confirmCamera(cameraRevision: number): Promise<ProjectDto>
  getVerifiedExport(): Promise<VerifiedExportDto>
  fetchExportArtifact(id: string, signal?: AbortSignal): Promise<Blob>
  getCompositePreview(signal?: AbortSignal): Promise<CompositePreviewDto>
  fetchCompositePreviewArtifact(id: string, signal?: AbortSignal): Promise<Blob>
  copyVerifiedExport(id: string, destination: string): Promise<void>
  getSubjectMedia(role: SubjectMediaRole): Promise<SubjectMediaDto>
  fetchSubjectMediaArtifact(
    role: SubjectMediaRole,
    id: string,
    signal?: AbortSignal,
  ): Promise<Blob>
  startTask(targetStage: StageName, expectedProjectId: string): Promise<TaskDto>
  getTask(id: string): Promise<TaskDto>
  cancelTask(id: string): Promise<TaskDto>
}
