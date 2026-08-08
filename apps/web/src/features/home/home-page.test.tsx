import { fireEvent, render, screen, within } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import type { ProjectSummaryDto } from '../../api/types'
import { HomePage } from './home-page'

const project: ProjectSummaryDto = {
  project_id: 'project-1',
  name: '夜景合成',
  created_at: '2026-08-01T00:00:00Z',
  updated_at: '2026-08-08T10:00:00Z',
  workflow_step: 'camera',
  active_task_id: null,
}

describe('HomePage', () => {
  it('creates and opens projects from the project home', () => {
    const onCreate = vi.fn(async () => undefined)
    const onOpen = vi.fn(async () => undefined)
    render(<HomePage activeProjectId="project-1" busy={false} projects={[project]} onCreate={onCreate} onDelete={vi.fn()} onOpen={onOpen} onRename={vi.fn()} />)

    fireEvent.change(screen.getByLabelText('新项目名称'), { target: { value: '新项目' } })
    fireEvent.click(screen.getByRole('button', { name: '创建项目' }))
    expect(onCreate).toHaveBeenCalledWith('新项目')

    fireEvent.click(screen.getByRole('button', { name: '继续制作' }))
    expect(onOpen).toHaveBeenCalledWith(project)
    expect(screen.getAllByText('调整机位')).toHaveLength(1)
    const card = screen.getByRole('article')
    expect(within(card).getByRole('button', { name: '重命名项目 夜景合成' })).toHaveTextContent('✎')
    expect(within(card).getByRole('button', { name: '删除项目 夜景合成' })).toHaveTextContent('×')
    expect(within(card).getAllByRole('button')).toHaveLength(3)
  })

  it('explains that deleting a project preserves shared assets', () => {
    render(<HomePage activeProjectId={null} busy={false} projects={[project]} onCreate={vi.fn()} onDelete={vi.fn()} onOpen={vi.fn()} onRename={vi.fn()} />)
    fireEvent.click(screen.getByRole('button', { name: '删除项目 夜景合成' }))
    expect(screen.getByText(/共享素材库中的视频和 PLY 不受影响/)).toBeInTheDocument()
  })
})
