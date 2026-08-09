import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import type { BackendClient } from '../../api/backend-client'
import type { ProjectDto } from '../../api/types'
import type { PlatformBridge } from '../../platform/platform-bridge'
import { AssetLibraryPage } from './asset-library-page'

describe('AssetLibraryPage', () => {
  it('keeps video and PLY navigation separate and protects referenced assets', async () => {
    const backend = {
      listAssets: vi.fn(async () => [{
        asset: {
          asset_id: 'asset-1', kind: 'video', original_filename: 'clip.mp4',
          stored_relative_path: 'aa/hash.mp4', size: 1024, sha256: 'a'.repeat(64),
          imported_at: '2026-08-08T10:00:00Z', video_summary: null, scene_summary: null,
        },
        references: [{
          project_id: 'project-1', name: '夜景合成', created_at: '2026-08-01T00:00:00Z',
          updated_at: '2026-08-08T10:00:00Z', workflow_step: 'camera', active_task_id: null,
        }],
      }]),
    } as unknown as BackendClient
    const platform = { kind: 'browser', pickInputFile: vi.fn() } as unknown as PlatformBridge

    render(<AssetLibraryPage backend={backend} busy={false} kind="video" onError={vi.fn()} onProjectChange={vi.fn()} platform={platform} project={null} returnProjectId="project-1" selectionAuthority={0} />)

    await waitFor(() => expect(screen.getByText('clip.mp4')).toBeInTheDocument())
    expect(screen.getByRole('link', { name: '视频' })).toHaveAttribute('href', '#/assets/video?returnProject=project-1')
    expect(screen.getByRole('link', { name: 'PLY' })).toHaveAttribute('href', '#/assets/ply?returnProject=project-1')
    expect(screen.getByText('夜景合成')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '删除' })).toBeDisabled()
  })

  it('disables project selection while an asset assignment is pending', async () => {
    const backend = {
      listAssets: vi.fn(async () => [{
        asset: {
          asset_id: 'asset-1', kind: 'video', original_filename: 'clip.mp4',
          stored_relative_path: 'aa/hash.mp4', size: 1024, sha256: 'a'.repeat(64),
          imported_at: '2026-08-08T10:00:00Z', video_summary: null, scene_summary: null,
        },
        references: [],
      }]),
      selectProjectAsset: vi.fn(() => new Promise<ProjectDto>(() => undefined)),
    } as unknown as BackendClient
    const platform = { kind: 'browser', pickInputFile: vi.fn() } as unknown as PlatformBridge
    const project = {
      project_id: 'project-1', source_video_asset_id: null, scene_ply_asset_id: null,
    } as unknown as ProjectDto
    const user = userEvent.setup()

    render(<AssetLibraryPage backend={backend} busy={false} kind="video" onError={vi.fn()} onProjectChange={vi.fn()} platform={platform} project={project} returnProjectId="project-1" selectionAuthority={0} />)

    const select = await screen.findByRole('button', { name: '用于当前项目' })
    await user.click(select)
    expect(select).toBeDisabled()
    expect(backend.selectProjectAsset).toHaveBeenCalledWith('source_video', 'asset-1', 'project-1')
  })
})
