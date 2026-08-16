import { describe, expect, it } from 'vitest'

import type { ProjectDto, StageStateDto } from '../api/types'
import { canVisitStep, workflowStepForProject } from './project-store'

function stage(status: StageStateDto['status']): StageStateDto {
  return {
    status,
    cache_key: null,
    output_paths: [],
    error_code: null,
    artifacts: {},
  }
}

function projectWithImport(status: StageStateDto['status']): ProjectDto {
  return {
    schema_version: 3,
    project_id: 'project-1',
    name: 'Portrait import',
    created_at: '2026-08-01T00:00:00Z',
    source_video: 'opaque:source',
    scene_ply: 'opaque:scene',
    stages: {
      ingest: stage(status),
      segment: stage('pending'),
      solve_camera: stage('pending'),
      map_trajectory: stage('pending'),
      render: stage('pending'),
      composite: stage('pending'),
      export: stage('pending'),
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
      subject_prompt: null,
      target_camera: null,
      exploration_camera: null,
      preview_epoch: 0,
      confirmed_camera_revision: null,
      confirmed_preview_artifact_id: null,
      foot_point: null,
      subject_visibility_audit: null,
      source_perspective_calibration: null,
      local_ground_anchor: null,
      subject_contact_constraint: null,
      synthesis_placement: null,
      confirmed_synthesis_placement_revision: null,
      motion_scale: 1,
      preview_height: 540,
      active_task_id: null,
      preview: null,
      export_result: null,
    },
  }
}

describe('workflow import authority', () => {
  it('keeps the subject step unreachable until ingest succeeds', () => {
    const running = projectWithImport('running')

    expect(canVisitStep(running, 'subject')).toBe(false)
    expect(workflowStepForProject(running)).toBe('import')

    const succeeded = projectWithImport('succeeded')
    expect(canVisitStep(succeeded, 'subject')).toBe(true)
    expect(workflowStepForProject(succeeded)).toBe('subject')
  })
})
