import type {
  AssetDto,
  AssetKind,
  BootstrapDto,
  ProjectDto,
  ProjectPatch,
  PreviewFrameDto,
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
} from './types'

export interface BackendClient {
  bootstrap(signal?: AbortSignal): Promise<BootstrapDto>
  getEnvironmentRepair(): Promise<EnvironmentRepairSnapshotDto>
  startEnvironmentRepair(): Promise<EnvironmentRepairSnapshotDto>
  cancelEnvironmentRepair(): Promise<EnvironmentRepairSnapshotDto>
  importLocalPath(kind: AssetKind, path: string): Promise<AssetDto>
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
  startTask(targetStage: StageName): Promise<TaskDto>
  getTask(id: string): Promise<TaskDto>
  cancelTask(id: string): Promise<TaskDto>
}
