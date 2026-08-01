import { act, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { BackendClient } from '../../api/backend-client'
import { BackendClientError } from '../../api/http-backend-client'
import type { ProjectDto, StageStateDto } from '../../api/types'
import { SubjectPage } from './subject-page'

afterEach(() => vi.useRealTimers())

function readyProject(): ProjectDto {
  const stage = (status: 'pending' | 'succeeded'): StageStateDto => ({
    status, cache_key: status === 'succeeded' ? 'ingest-cache' : null,
    output_paths: [], error_code: null, artifacts: {},
  })
  return {
    schema_version: 3,
    project_id: 'project-1',
    name: 'Portrait import',
    created_at: '2026-08-01T00:00:00Z',
    source_video: 'opaque:source',
    scene_ply: 'opaque:scene',
    stages: {
      ingest: stage('succeeded'), segment: stage('pending'), solve_camera: stage('pending'),
      map_trajectory: stage('pending'), render: stage('pending'),
      composite: stage('pending'), export: stage('pending'),
    },
    workflow: {
      source_summary: {
        filename: 'portrait.mp4', size: 10, sha256: 'source', width: 480, height: 852,
        duration_seconds: 12, fps: '30', has_audio: false, frame_count: 360,
      },
      scene_summary: {
        filename: 'scene.ply', size: 20, sha256: 'scene', gaussian_count: 100,
        estimated_vram_mb: 128,
      },
      subject_prompt: null, target_camera: null, preview_epoch: 0,
      confirmed_camera_revision: null, confirmed_preview_artifact_id: null,
      foot_point: null, motion_scale: 1, preview_height: 540,
      active_task_id: null, preview: null, export_result: null,
    },
  }
}

describe('subject media readiness', () => {
  it('retries a transient not-ready response without requiring remount', async () => {
    vi.useFakeTimers()
    const notReady = new BackendClientError(409, {
      code: 'subject_media_not_ready', category: 'project',
      message: 'The requested subject media is not ready.', retryable: false,
    })
    const getSubjectMedia = vi.fn()
      .mockRejectedValueOnce(notReady)
      .mockResolvedValue({
        role: 'proxy', artifact_id: 'proxy-1', frame_index: 0,
        width: 480, height: 852, size: 8, mime_type: 'image/jpeg',
      })
    const backend = {
      getSubjectMedia,
      fetchSubjectMediaArtifact: vi.fn(async () => new Blob(['proxy'], { type: 'image/jpeg' })),
    } as unknown as BackendClient
    const onError = vi.fn()

    render(
      <SubjectPage
        backend={backend}
        busy={false}
        onError={onError}
        onProjectChange={vi.fn()}
        onStartStage={vi.fn()}
        project={readyProject()}
      />,
    )

    await act(async () => { await Promise.resolve() })
    expect(getSubjectMedia).toHaveBeenCalledTimes(1)
    expect(screen.getByText('载入代表帧…')).toBeVisible()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(250)
    })

    expect(getSubjectMedia).toHaveBeenCalledTimes(2)
    expect(screen.getByRole('img', { name: '人物代表帧' })).toBeVisible()
    expect(onError).not.toHaveBeenCalled()
  })
})
