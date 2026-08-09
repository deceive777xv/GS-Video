import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { expect, it, vi } from 'vitest'

import type { BackendClient } from '../../api/backend-client'
import type { StorageLayoutDto } from '../../api/types'
import type { PlatformBridge } from '../../platform/platform-bridge'
import { StorageSettingsPage } from './storage-settings-page'

const initial: StorageLayoutDto = {
  project_library_root: 'E:\GS Video\project-library',
  project_library_id: 'project-id',
  cache_root: 'E:\GS Video\cache-library',
  cache_id: 'cache-id',
  restart_required: false,
  editable: true,
  blocked_reason: null,
  project_library_bytes: 1024,
  project_library_free_bytes: 1024 ** 3,
  cache_bytes: 2048,
  cache_free_bytes: 2 * 1024 ** 3,
}

it('chooses both desktop directories, saves, and offers restart', async () => {
  const user = userEvent.setup()
  const updated: StorageLayoutDto = {
    ...initial,
    project_library_root: 'D:\Projects',
    cache_root: 'F:\Cache',
    restart_required: true,
    editable: false,
    blocked_reason: 'restart_required',
  }
  const updateStorageLayout = vi.fn().mockResolvedValue(updated)
  const backend = { updateStorageLayout } as unknown as BackendClient
  const pickDirectory = vi.fn()
    .mockResolvedValueOnce('D:\Projects')
    .mockResolvedValueOnce('F:\Cache')
  const restartApp = vi.fn().mockResolvedValue(undefined)
  const platform = {
    kind: 'tauri',
    pickDirectory,
    restartApp,
  } as unknown as PlatformBridge

  render(
    <StorageSettingsPage
      backend={backend}
      busy={false}
      initial={initial}
      onChange={vi.fn()}
      onError={vi.fn()}
      platform={platform}
    />,
  )

  const chooseButtons = screen.getAllByRole('button', { name: '选择目录' })
  await user.click(chooseButtons[0]!)
  await user.click(chooseButtons[1]!)
  await user.click(screen.getByRole('button', { name: '应用目录设置' }))

  expect(updateStorageLayout).toHaveBeenCalledWith({
    project_library_root: 'D:\Projects',
    cache_root: 'F:\Cache',
    project_action: 'migrate',
    cache_action: 'start_fresh',
  })
  await user.click(await screen.findByRole('button', { name: '重启应用' }))
  expect(restartApp).toHaveBeenCalledOnce()
})

it('keeps directory changes read-only in browser mode', () => {
  render(
    <StorageSettingsPage
      backend={{} as BackendClient}
      busy={false}
      initial={initial}
      onChange={vi.fn()}
      onError={vi.fn()}
      platform={{ kind: 'browser' } as PlatformBridge}
    />,
  )

  expect(screen.getByText(/浏览器模式只能查看/)).toBeInTheDocument()
  expect(screen.getAllByRole('button', { name: '选择目录' })[0]).toBeDisabled()
})

it('executes cleanup only with the confirmed backend plan token', async () => {
  const user = userEvent.setup()
  const planStorageCacheCleanup = vi.fn().mockResolvedValue({
    plan_token: 'a'.repeat(32),
    mode: 'safe',
    removable_entries: 2,
    reclaimable_bytes: 2048,
    expires_in_seconds: 300,
  })
  const cleanupStorageCache = vi.fn().mockResolvedValue({
    mode: 'safe',
    removed_entries: 2,
    freed_bytes: 2048,
    storage: initial,
  })
  vi.spyOn(window, 'confirm').mockReturnValue(true)

  render(
    <StorageSettingsPage
      backend={{ planStorageCacheCleanup, cleanupStorageCache } as unknown as BackendClient}
      busy={false}
      initial={initial}
      onChange={vi.fn()}
      onError={vi.fn()}
      platform={{ kind: 'tauri', pickDirectory: vi.fn() } as unknown as PlatformBridge}
    />,
  )

  await user.click(screen.getByRole('button', { name: '安全清理' }))

  expect(planStorageCacheCleanup).toHaveBeenCalledWith('safe')
  expect(cleanupStorageCache).toHaveBeenCalledWith('safe', 'a'.repeat(32))
})
