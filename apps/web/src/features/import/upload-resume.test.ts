import { afterEach, describe, expect, it } from 'vitest'

import {
  clearUploadResume,
  readUploadResume,
  writeUploadResume,
  type UploadResumeRecord,
} from './upload-resume'

const record: UploadResumeRecord = {
  version: 1,
  projectId: 'project-1',
  kind: 'source_video',
  filename: 'clip.mp4',
  mimeType: 'video/mp4',
  size: 123,
  sha256: 'a'.repeat(64),
  id: 'upload-1',
  chunkSize: 32,
}

afterEach(() => sessionStorage.clear())

describe('upload resume metadata', () => {
  it('persists and restores only the non-secret file/session fingerprint', () => {
    writeUploadResume(record)
    expect(readUploadResume('project-1', 'source_video')).toEqual(record)
    const serialized = sessionStorage.getItem(
      'gs-video:upload-resume:project-1:source_video',
    ) ?? ''
    expect(Object.keys(JSON.parse(serialized))).toEqual([
      'version', 'projectId', 'kind', 'filename', 'mimeType', 'size',
      'sha256', 'id', 'chunkSize',
    ])
    expect(serialized).not.toContain('token')
    expect(serialized).not.toContain('path')
    expect(serialized).not.toContain('bytes')
  })

  it('fails closed for malformed or cross-project metadata and clears explicitly', () => {
    sessionStorage.setItem('gs-video:upload-resume:project-1:source_video', '{"id":"upload-1"}')
    expect(readUploadResume('project-1', 'source_video')).toBeNull()
    writeUploadResume(record)
    expect(readUploadResume('project-2', 'source_video')).toBeNull()
    clearUploadResume('project-1', 'source_video')
    expect(readUploadResume('project-1', 'source_video')).toBeNull()
  })
})
