import { render, screen, waitFor } from '@testing-library/react'
import { expect, it, vi } from 'vitest'

import type { BackendClient } from '../../api/backend-client'
import type { ProjectDto, TaskDto } from '../../api/types'
import { PreviewPage } from './preview-page'

it('does not admit a second composite task while the authoritative owner is active', () => {
  const project = {
    stages: { composite: { status: 'running' } },
    workflow: {
      motion_scale: 1,
      preview: null,
      target_camera: null,
      foot_point: {
        image: [1, 1], world: [0, 0, 0], preview_artifact_id: 'preview',
        camera_revision: 1, pick_buffer_revision: 1,
      },
    },
  } as unknown as ProjectDto
  const task: TaskDto = {
    id: 'task-composite', target_stage: 'composite', status: 'running',
    revision: 1, error: null,
  }
  const onStartStage = vi.fn()
  render(
    <PreviewPage
      activeTask={task}
      backend={{} as BackendClient}
      busy
      latestEvent={null}
      onBackToCamera={vi.fn()}
      onError={vi.fn()}
      onProjectChange={vi.fn()}
      onReselectSubject={vi.fn()}
      onStartStage={onStartStage}
      project={project}
    />,
  )

  expect(screen.getByRole('button', { name: '生成中…' })).toBeDisabled()
  expect(onStartStage).not.toHaveBeenCalled()
})

const failedProject = (target: TaskDto['target_stage'] = 'composite') => ({
  stages: {
    composite: target === 'composite'
      ? { status: 'failed', cache_key: null, output_paths: [], error_code: 'composite_failed', artifacts: {} }
      : { status: 'pending', cache_key: null, output_paths: [], error_code: null, artifacts: {} },
    [target]: { status: 'failed', cache_key: null, output_paths: [], error_code: `${target}_failed`, artifacts: {} },
  },
  workflow: {
    motion_scale: 1, preview: null, target_camera: null,
    foot_point: { image: [1, 1], world: [0, 0, 0], preview_artifact_id: 'preview', camera_revision: 1, pick_buffer_revision: 1 },
  },
}) as unknown as ProjectDto

it.each([
  ['segment', 'subject_mask_invalid', '重新选择人物', ['降低预览分辨率', '返回机位']],
  ['render', 'gpu_out_of_memory', '降低预览分辨率', ['重新选择人物', '返回机位']],
  ['solve_camera', 'camera_authority_stale', '返回机位', ['重新选择人物', '降低预览分辨率']],
] as const)('shows only the targeted %s recovery action', (target, code, expected, absent) => {
  const task: TaskDto = {
    id: `task-${target}`, target_stage: target, status: 'failed', revision: 3, error: code,
  }
  render(
    <PreviewPage
      activeTask={task}
      backend={{} as BackendClient}
      busy={false}
      latestEvent={{
        type: 'task_event', task_id: task.id, revision: 3, stage: target,
        progress: 1, error: { code, category: 'task', retryable: false },
      }}
      onBackToCamera={vi.fn()}
      onError={vi.fn()}
      onProjectChange={vi.fn()}
      onReselectSubject={vi.fn()}
      onStartStage={vi.fn()}
      project={failedProject(target)}
    />,
  )
  expect(screen.getByRole('button', { name: expected })).toBeInTheDocument()
  for (const name of absent) expect(screen.queryByRole('button', { name })).toBeNull()
  expect(screen.queryByRole('button', { name: '重试' })).toBeNull()
})

it('offers retry only for a failed task whose event says it is retryable', () => {
  const task: TaskDto = {
    id: 'task-composite', target_stage: 'composite', status: 'failed', revision: 3,
    error: 'transient_render_failure',
  }
  render(
    <PreviewPage
      activeTask={task}
      backend={{} as BackendClient}
      busy={false}
      latestEvent={{
        type: 'task_event', task_id: task.id, revision: 3, stage: 'composite',
        progress: 1, error: { code: task.error, category: 'resource', retryable: true },
      }}
      onBackToCamera={vi.fn()}
      onError={vi.fn()}
      onProjectChange={vi.fn()}
      onReselectSubject={vi.fn()}
      onStartStage={vi.fn()}
      project={failedProject()}
    />,
  )
  expect(screen.getByRole('button', { name: '重试' })).toBeInTheDocument()
})

it('revokes and removes a displayed frame when preview authority becomes null', async () => {
  const createUrl = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:preview')
  const revokeUrl = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined)
  const backend = {
    fetchPreviewArtifact: vi.fn(async () => new Blob(['preview'])),
  } as unknown as BackendClient
  const initial = failedProject()
  initial.workflow.preview = {
    artifact_id: 'preview-1', generation: 1, width: 640, height: 360,
    camera_revision: 1, pick_buffer_revision: 1,
    artifact_size: 7, artifact_sha256: 'a'.repeat(64),
  }
  const props = {
    activeTask: null,
    backend,
    busy: false,
    latestEvent: null,
    onBackToCamera: vi.fn(),
    onError: vi.fn(),
    onProjectChange: vi.fn(),
    onReselectSubject: vi.fn(),
    onStartStage: vi.fn(),
  }
  const view = render(<PreviewPage {...props} project={initial} />)
  expect(await screen.findByRole('img', { name: '最近验证的合成参考帧' })).toHaveAttribute('src', 'blob:preview')

  const revoked = failedProject()
  revoked.workflow.preview = null
  view.rerender(<PreviewPage {...props} project={revoked} />)

  await waitFor(() => expect(screen.queryByRole('img', { name: '最近验证的合成参考帧' })).toBeNull())
  expect(createUrl).toHaveBeenCalledOnce()
  expect(revokeUrl).toHaveBeenCalledWith('blob:preview')
})
