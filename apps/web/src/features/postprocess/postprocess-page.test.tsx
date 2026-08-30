import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
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

afterEach(() => {
  vi.useRealTimers()
  vi.restoreAllMocks()
})

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

  it('waits until an effect slider is released before requesting another image', async () => {
    vi.useFakeTimers()
    vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:preview')
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined)
    const current = project()
    current.workflow.effect_chain = [{
      instance_id: 'sharpen-1',
      type: 'sharpen',
      params_version: 1,
      display_name: null,
      enabled: true,
      mix: 100,
      parameters: { amount: 50, radius: 1, threshold: 1 },
    }]
    const client = backend(current)

    render(<PostProcessPage activeTask={null} backend={client} busy={false} onDirtyChange={vi.fn()} onError={vi.fn()} onProjectChange={vi.fn()} onStartStage={vi.fn()} project={current} />)
    await act(async () => { await vi.advanceTimersByTimeAsync(150) })
    const renderPreview = vi.mocked(client.renderDraftPostProcessPreview)
    expect(renderPreview).toHaveBeenCalledTimes(2)
    renderPreview.mockClear()

    const amount = screen.getByText('强度').closest('label')?.querySelector('input[type="range"]')
    expect(amount).toBeInstanceOf(HTMLInputElement)
    fireEvent.pointerDown(amount as HTMLInputElement, { pointerId: 1 })
    fireEvent.input(amount as HTMLInputElement, { target: { value: '70' } })
    fireEvent.input(amount as HTMLInputElement, { target: { value: '85' } })
    expect(screen.getByText('85%')).toBeVisible()
    expect(screen.getByText('已保存 · revision 0')).toBeVisible()
    await act(async () => { await vi.advanceTimersByTimeAsync(500) })
    expect(renderPreview).not.toHaveBeenCalled()

    fireEvent.pointerUp(amount as HTMLInputElement, { pointerId: 1 })
    await act(async () => { await vi.advanceTimersByTimeAsync(150) })
    expect(renderPreview).toHaveBeenCalledTimes(2)
    expect(renderPreview).toHaveBeenLastCalledWith(expect.objectContaining({
      effect_chain: [expect.objectContaining({ parameters: expect.objectContaining({ amount: 85 }) })],
    }), expect.any(AbortSignal))
  })

  it('uses the comparison line itself as the before-after control', async () => {
    vi.useFakeTimers()
    vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:preview')
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined)
    const current = project()
    const client = backend(current)

    render(<PostProcessPage activeTask={null} backend={client} busy={false} onDirtyChange={vi.fn()} onError={vi.fn()} onProjectChange={vi.fn()} onStartStage={vi.fn()} project={current} />)
    await act(async () => { await vi.advanceTimersByTimeAsync(150) })
    const divider = screen.getByRole('slider', { name: '前后分割线' })
    expect(divider.tagName).toBe('DIV')
    const frame = divider.parentElement
    expect(frame).not.toBeNull()
    vi.spyOn(frame as HTMLElement, 'getBoundingClientRect').mockReturnValue({
      bottom: 562.5, height: 562.5, left: 0, right: 1000, top: 0, width: 1000,
      x: 0, y: 0, toJSON: () => ({}),
    })

    fireEvent.pointerDown(divider, { button: 0, clientX: 500, pointerId: 7 })
    fireEvent.pointerMove(divider, { clientX: 760, pointerId: 7 })
    expect(divider).toHaveAttribute('aria-valuenow', '76')
    fireEvent.pointerUp(divider, { clientX: 760, pointerId: 7 })

    fireEvent.keyDown(divider, { key: 'ArrowLeft', shiftKey: true })
    expect(divider).toHaveAttribute('aria-valuenow', '71')
    expect(screen.queryByRole('slider', { name: /前后分割$/ })).not.toBeInTheDocument()
  })

  it('limits effect reordering drag behavior to the dedicated handle', () => {
    vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:preview')
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined)
    const current = project()
    current.workflow.effect_chain = [{
      instance_id: 'sharpen-1',
      type: 'sharpen',
      params_version: 1,
      display_name: null,
      enabled: true,
      mix: 100,
      parameters: { amount: 50, radius: 1, threshold: 1 },
    }]

    render(<PostProcessPage activeTask={null} backend={backend(current)} busy={false} onDirtyChange={vi.fn()} onError={vi.fn()} onProjectChange={vi.fn()} onStartStage={vi.fn()} project={current} />)
    const card = screen.getByPlaceholderText('锐化').closest('.effect-card')
    expect(card).not.toBeNull()
    expect(card).not.toHaveAttribute('draggable')
    expect(card?.querySelector('.drag-handle')).toHaveAttribute('draggable', 'true')
  })
})
