import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { BackendClient } from '../../api/backend-client'
import type { EffectInstance, ProjectDto } from '../../api/types'
import { PostProcessPage } from './postprocess-page'

function project(): ProjectDto {
  return {
    schema_version: 9,
    project_id: 'project-1',
    name: 'Post process',
    created_at: '2026-08-28T00:00:00Z',
    source_video: 'opaque:source',
    scene_ply: 'opaque:scene',
    stages: {
      composite: { status: 'succeeded', cache_key: 'composite', output_paths: [], error_code: null, artifacts: {} },
      post_process: { status: 'pending', cache_key: null, output_paths: [], error_code: null, artifacts: {} },
    },
    workflow: {
      source_summary: {
        filename: 'source.mp4', size: 1, sha256: 'a'.repeat(64), width: 640,
        height: 360, duration_seconds: 2, fps: '24/1', has_audio: false,
        frame_count: 48,
      },
      scene_summary: null,
      subject_prompt: { frame_index: 0, x: 1, y: 1 },
      target_camera: null,
      exploration_camera: null,
      preview_epoch: 0,
      target_ground: null,
      gs_scale: 1,
      scene_azimuth: 0,
      output_crop: { x: 0, y: 0, width: 640, height: 360 },
      preview_height: 540,
      source_color_interpretation: 'rec709_metadata',
      matte_refinement: {
        enabled: true, edge_offset: -1, feather_radius: 1,
        decontaminate_strength: 0, decontaminate_radius: 3,
      },
      effect_chain: [],
      effect_chain_revision: 0,
      export_settings: {
        codec: 'h264', rate_control: 'constant_quality', quality: 70,
        target_bitrate_mbps: 12, compression_preset: 'balanced',
      },
      active_task_id: null,
      preview: null,
      export_result: null,
    },
  }
}

function backend(current: ProjectDto, save?: (chain: EffectInstance[]) => Promise<ProjectDto>): BackendClient {
  return {
    listAssets: vi.fn(async () => []),
    renderDraftPostProcessPreview: vi.fn(async () => new Blob(['png'], { type: 'image/png' })),
    closePostProcessPreview: vi.fn(async () => undefined),
    updateProject: vi.fn(async (patch) => {
      if (patch.effect_chain === undefined) return current
      if (save !== undefined) return save(patch.effect_chain)
      current = structuredClone(current)
      current.workflow.effect_chain = structuredClone(patch.effect_chain)
      current.workflow.effect_chain_revision += 1
      return current
    }),
  } as unknown as BackendClient
}

afterEach(() => vi.restoreAllMocks())

describe('PostProcessPage', () => {
  it('keeps edits in a draft and saves them with the expected revision', async () => {
    vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:preview')
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined)
    const current = project()
    const client = backend(current)
    const dirty = vi.fn()
    const user = userEvent.setup()

    render(<PostProcessPage activeTask={null} backend={client} busy={false} onDirtyChange={dirty} onError={vi.fn()} onProjectChange={vi.fn()} onStartStage={vi.fn()} project={current} />)
    await user.click(screen.getByRole('button', { name: '+ 基础校色' }))
    expect(screen.getByPlaceholderText('基础校色')).toBeVisible()
    expect(client.updateProject).not.toHaveBeenCalled()
    await user.click(screen.getByRole('button', { name: '保存效果链' }))

    await waitFor(() => expect(client.updateProject).toHaveBeenCalledWith(expect.objectContaining({
      expected_effect_chain_revision: 0,
      effect_chain: [expect.objectContaining({ type: 'primary_correction', mix: 100 })],
    })))
    expect(await screen.findByText('已保存 · revision 1')).toBeVisible()
    expect(dirty).toHaveBeenCalledWith(true)
  })

  it('preserves the local draft when revision-conflict saving fails', async () => {
    vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:preview')
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined)
    const current = project()
    const failure = new Error('effect_chain_revision_conflict')
    const client = backend(current, async () => { throw failure })
    const onError = vi.fn()
    const user = userEvent.setup()

    render(<PostProcessPage activeTask={null} backend={client} busy={false} onDirtyChange={vi.fn()} onError={onError} onProjectChange={vi.fn()} onStartStage={vi.fn()} project={current} />)
    await user.click(screen.getByRole('button', { name: '+ 锐化' }))
    await user.click(screen.getByRole('button', { name: '保存效果链' }))

    await waitFor(() => expect(onError).toHaveBeenCalledWith(failure))
    expect(screen.getByPlaceholderText('锐化')).toBeVisible()
    expect(screen.getByText('有未保存修改')).toBeVisible()
  })
})
