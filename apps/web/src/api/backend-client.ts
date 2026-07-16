import type {
  AssetDto,
  AssetKind,
  BootstrapDto,
  ProjectDto,
  ProjectPatch,
  StageName,
  TaskDto,
  UploadInit,
  UploadSessionDto,
} from './types'

export interface BackendClient {
  bootstrap(signal?: AbortSignal): Promise<BootstrapDto>
  importLocalPath(kind: AssetKind, path: string): Promise<AssetDto>
  createUpload(input: UploadInit): Promise<UploadSessionDto>
  putUploadChunk(
    id: string,
    index: number,
    data: Blob,
    signal?: AbortSignal,
  ): Promise<void>
  getProject(): Promise<ProjectDto>
  updateProject(patch: ProjectPatch): Promise<ProjectDto>
  startTask(targetStage: StageName): Promise<TaskDto>
  getTask(id: string): Promise<TaskDto>
  cancelTask(id: string): Promise<TaskDto>
}
