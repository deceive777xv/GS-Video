import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, expect, it, vi } from 'vitest'

import type { BackendClient } from '../../api/backend-client'
import type { ProjectDto, TargetGroundDto } from '../../api/types'
import { ConstrainedCameraWorkspace } from './constrained-camera-workspace'

afterEach(() => vi.restoreAllMocks())

const identity = [
  [1, 0, 0, 0],
  [0, 1, 0, 0],
  [0, 0, 1, -4],
  [0, 0, 0, 1],
] as const

function project(targetGround: TargetGroundDto | null = null): ProjectDto {
  return {
    project_id: 'project-1',
    scene_ply: 'opaque:scene',
    workflow: {
      exploration_camera: null,
      preview: null,
      target_ground: targetGround,
      gs_scale: 1,
      scene_azimuth: 0,
    },
  } as unknown as ProjectDto
}

it('turns three viewport hints into one automatic ground candidate and one confirmation', async () => {
  vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:ground-view')
  vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined)
  const initial = project()
  const candidate: TargetGroundDto = {
    scene_asset_id: 'scene-1',
    hint_pixels: [[100, 100], [400, 100], [250, 300]],
    p0_world: [0, 0, 0],
    p1_world: [1, 0, 0],
    p2_world: [0, 0, 1],
    plane_normal: [0, 1, 0],
    plane_offset: 0,
    exploration_camera_to_world: identity.map((row) => [...row]) as TargetGroundDto['exploration_camera_to_world'],
    camera_fingerprint: 'f'.repeat(64),
    preview_artifact_id: 'preview-1',
    camera_revision: 1,
    pick_buffer_revision: 1,
    support_counts: [20, 21, 22],
    weighted_inlier_ratio: 0.9,
    rms_residual: 0.01,
    confidence: 0.92,
    revision: 1,
    confirmed: false,
  }
  const candidateProject = project(candidate)
  const confirmedProject = project({ ...candidate, confirmed: true })
  const backend = {
    renderPreview: vi.fn(async () => ({
      artifact_id: 'preview-1', generation: 1, width: 960, height: 540,
      camera_revision: 1, pick_buffer_revision: 1,
    })),
    fetchPreviewArtifact: vi.fn(async () => new Blob(['png'], { type: 'image/png' })),
    fitTargetGround: vi.fn(async () => candidateProject),
    confirmTargetGround: vi.fn(async () => confirmedProject),
    updateProject: vi.fn(),
  } as unknown as BackendClient
  const onProjectChange = vi.fn()
  const props = {
    backend,
    busy: false,
    onError: vi.fn(),
    onProjectChange,
    onRefresh: vi.fn(async () => initial),
    panel: 'scene' as const,
  }
  const user = userEvent.setup()
  const view = render(<ConstrainedCameraWorkspace {...props} project={initial} />)

  await user.click(screen.getByRole('button', { name: '更新地面选择视图' }))
  const viewport = await screen.findByLabelText('Gaussian 地面提示视口')
  vi.spyOn(viewport, 'getBoundingClientRect').mockReturnValue({
    x: 0, y: 0, left: 0, top: 0, right: 960, bottom: 540,
    width: 960, height: 540, toJSON: () => ({}),
  })
  fireEvent.pointerDown(viewport, { clientX: 100, clientY: 100 })
  fireEvent.pointerDown(viewport, { clientX: 400, clientY: 100 })
  fireEvent.pointerDown(viewport, { clientX: 250, clientY: 300 })

  await user.click(screen.getByRole('button', { name: '自动寻找真正地面' }))
  await waitFor(() => expect(backend.fitTargetGround).toHaveBeenCalledWith({
    expected_project_id: 'project-1',
    preview_artifact_id: 'preview-1',
    camera_revision: 1,
    pick_buffer_revision: 1,
    hints: [[100, 100], [400, 100], [250, 300]],
  }))

  view.rerender(<ConstrainedCameraWorkspace {...props} project={candidateProject} />)
  await user.click(screen.getByRole('button', { name: '确认地面' }))
  expect(backend.confirmTargetGround).toHaveBeenCalledWith('project-1', 1)
})
