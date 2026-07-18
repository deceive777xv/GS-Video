import type { AssetKind, UploadStatusDto } from '../../api/types'

export interface UploadResumeRecord {
  version: 1
  projectId: string
  kind: AssetKind
  filename: string
  mimeType: string
  size: number
  sha256: string
  id: string
  chunkSize: number
}

function storageKey(projectId: string, kind: AssetKind): string {
  return `gs-video:upload-resume:${encodeURIComponent(projectId)}:${kind}`
}

function validRecord(value: unknown): value is UploadResumeRecord {
  if (typeof value !== 'object' || value === null) return false
  const candidate = value as Record<string, unknown>
  return candidate.version === 1
    && typeof candidate.projectId === 'string' && candidate.projectId.length > 0
    && (candidate.kind === 'source_video' || candidate.kind === 'scene_ply')
    && typeof candidate.filename === 'string' && candidate.filename.length > 0
    && typeof candidate.mimeType === 'string' && candidate.mimeType.length > 0
    && Number.isSafeInteger(candidate.size) && (candidate.size as number) >= 0
    && typeof candidate.sha256 === 'string' && /^[a-f0-9]{64}$/.test(candidate.sha256)
    && typeof candidate.id === 'string' && candidate.id.length > 0
    && Number.isSafeInteger(candidate.chunkSize) && (candidate.chunkSize as number) > 0
}

export function readUploadResume(
  projectId: string,
  kind: AssetKind,
): UploadResumeRecord | null {
  try {
    const raw = sessionStorage.getItem(storageKey(projectId, kind))
    if (raw === null) return null
    const value: unknown = JSON.parse(raw)
    if (!validRecord(value) || value.projectId !== projectId || value.kind !== kind) return null
    return value
  } catch {
    return null
  }
}

export function writeUploadResume(record: UploadResumeRecord): void {
  try {
    sessionStorage.setItem(storageKey(record.projectId, record.kind), JSON.stringify(record))
  } catch {
    // Session storage is a recovery optimization; upload still works without it.
  }
}

export function clearUploadResume(projectId: string, kind: AssetKind): void {
  try {
    sessionStorage.removeItem(storageKey(projectId, kind))
  } catch {
    // Storage may be unavailable in hardened browser contexts.
  }
}

export function matchesSelectedFile(
  record: UploadResumeRecord,
  kind: AssetKind,
  file: File,
  sha256: string,
): boolean {
  return record.kind === kind
    && record.filename === file.name
    && record.mimeType === (file.type || 'application/octet-stream')
    && record.size === file.size
    && record.sha256 === sha256
}

export function matchesServerStatus(
  record: UploadResumeRecord,
  status: UploadStatusDto,
): boolean {
  return status.id === record.id
    && status.kind === record.kind
    && status.filename === record.filename
    && status.total_size === record.size
    && status.chunk_size === record.chunkSize
}
