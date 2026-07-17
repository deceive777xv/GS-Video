import { render, screen } from '@testing-library/react'
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
