import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { readdir, readFile } from 'node:fs/promises'
import path from 'node:path'
import { createElement } from 'react'
import { describe, expect, it, vi } from 'vitest'

import {
  BrowserCompositionRoot,
  TauriCompositionRoot,
} from '../composition-root'
import type { BackendClient } from '../api/backend-client'
import { BrowserPlatformBridge } from './browser-platform-bridge'
import { TauriPlatformBridge } from './tauri-platform-bridge'

const bootstrap = {
  api_version: '1',
  capabilities: [],
  environment: {
    ready: true,
    vram_mb: 8192,
    vram_limit_mb: 8192,
    issues: [],
    renderer_versions: null,
  },
  vram_budget: {
    mode: 'standard',
    minimum_vram_mb: 1024,
    total_vram_mb: 8192,
    selected_vram_mb: 8192,
    editable: true,
    blocked_reason: null,
    recovered_from_invalid_preference: false,
  },
  project: {
    schema_version: 3,
    project_id: 'project-1',
    name: 'Project',
    created_at: '2026-07-17T00:00:00Z',
    source_video: null,
    scene_ply: null,
    stages: {},
    workflow: {
      source_summary: null,
      scene_summary: null,
      subject_prompt: null,
      target_camera: null,
      preview_epoch: 0,
      confirmed_camera_revision: null,
      confirmed_preview_artifact_id: null,
      foot_point: null,
      motion_scale: 1,
      preview_height: 540,
      active_task_id: null,
      preview: null,
      export_result: null,
    },
  },
} as const

const fakeClient = (): BackendClient => ({
  bootstrap: vi.fn().mockResolvedValue(bootstrap),
  getVramBudget: vi.fn().mockResolvedValue(bootstrap.vram_budget),
  updateVramBudget: vi.fn().mockResolvedValue(bootstrap.vram_budget),
  getStorageLayout: vi.fn(),
  updateStorageLayout: vi.fn(),
  cleanupStorageCache: vi.fn(),
  planStorageCacheCleanup: vi.fn(),
  getEnvironmentRepair: vi.fn().mockResolvedValue({
    state: 'idle',
    job_id: null,
    step: null,
    resource_id: null,
    resource_name: null,
    progress: 0,
    downloaded_bytes: 0,
    total_bytes: null,
    message: null,
    resume_available: false,
    restart_required: false,
    error: null,
    environment: null,
  }),
  startEnvironmentRepair: vi.fn(),
  cancelEnvironmentRepair: vi.fn(),
  listProjects: vi.fn(),
  createProject: vi.fn(),
  activateProject: vi.fn(),
  renameProject: vi.fn(),
  deleteProject: vi.fn(),
  listAssets: vi.fn(),
  deleteAsset: vi.fn(),
  selectProjectAsset: vi.fn(),
  importLocalPath: vi.fn().mockResolvedValue({
    kind: 'source_video',
    path: 'source/video.mp4',
    size: 42,
    sha256: 'a'.repeat(64),
  }),
  createUpload: vi.fn(),
  putUploadChunk: vi.fn(),
  getUpload: vi.fn(),
  completeUpload: vi.fn(),
  cancelUpload: vi.fn(),
  getProject: vi.fn(),
  updateProject: vi.fn(),
  renderPreview: vi.fn(),
  renderLivePreview: vi.fn().mockResolvedValue(new Blob()),
  closeLivePreview: vi.fn().mockResolvedValue(undefined),
  fetchPreviewArtifact: vi.fn(),
  pickFootPoint: vi.fn(),
  confirmCamera: vi.fn(),
  getVerifiedExport: vi.fn(),
  fetchExportArtifact: vi.fn(),
  getCompositePreview: vi.fn(),
  fetchCompositePreviewArtifact: vi.fn(),
  copyVerifiedExport: vi.fn(),
  getSubjectMedia: vi.fn(),
  fetchSubjectMediaArtifact: vi.fn(),
  startTask: vi.fn(),
  getTask: vi.fn(),
  cancelTask: vi.fn(),
})

async function findImports(
  directory: string,
  forbidden: RegExp,
  excludedFiles: readonly string[] = [],
): Promise<string[]> {
  const root = path.resolve(directory)
  const excluded = new Set(
    excludedFiles.map((file) => path.resolve(file)),
  )
  const matches: string[] = []
  const visit = async (current: string): Promise<void> => {
    const entries = await readdir(current, { withFileTypes: true })
    await Promise.all(
      entries.map(async (entry) => {
        const target = path.join(current, entry.name)
        if (entry.isDirectory()) return visit(target)
        if (excluded.has(path.normalize(target))) return
        if (!/\.[cm]?[jt]sx?$/.test(entry.name)) return
        const contents = await readFile(target, 'utf8')
        for (const match of contents.matchAll(/(?:from\s+|import\s*\()(['"])([^'"]+)\1/g)) {
          if (forbidden.test(match[2] ?? '')) matches.push(target)
          forbidden.lastIndex = 0
        }
      }),
    )
  }
  await visit(root)
  return matches
}

describe('platform boundary', () => {
  const workingDirectory = process.cwd()
  const webRoot = workingDirectory.endsWith(path.join('apps', 'web'))
    ? workingDirectory
    : path.join(workingDirectory, 'apps', 'web')

  it('fails closed when the import scan root does not exist', async () => {
    await expect(
      findImports(
        path.join(webRoot, 'src', 'missing-import-scan-root'),
        /^@tauri-apps\//,
      ),
    ).rejects.toMatchObject({ code: 'ENOENT' })
  })

  it('keeps tauri imports inside the dedicated platform adapter', async () => {
    const forbidden = await findImports(
      path.join(webRoot, 'src'),
      /^@tauri-apps\//,
      [path.join(webRoot, 'src', 'platform', 'tauri-platform-bridge.ts')],
    )
    expect(forbidden).toEqual([])
  })

  it('imports a selected desktop path through BackendClient', async () => {
    const client = fakeClient()
    const bridge = new TauriPlatformBridge(client, {
      loadDialog: async () => ({
        open: vi.fn().mockResolvedValue('C:\\media\\clip.mp4'),
        save: vi.fn(),
      }),
      loadOpener: async () => ({ openUrl: vi.fn(), revealItemInDir: vi.fn() }),
      loadCore: async () => ({ invoke: vi.fn() }),
    })

    const picked = await bridge.pickInputFile({
      kind: 'source_video',
      extensions: ['mp4'],
    })

    expect(client.importLocalPath).toHaveBeenCalledWith(
      'source_video',
      'C:\\media\\clip.mp4',
    )
    expect(picked).toMatchObject({ kind: 'local-asset' })
  })

  it('downloads browser exports with a temporary object URL', async () => {
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, 'click')
      .mockImplementation(() => undefined)
    const createObjectURL = vi
      .spyOn(URL, 'createObjectURL')
      .mockReturnValue('blob:export')
    const revokeObjectURL = vi
      .spyOn(URL, 'revokeObjectURL')
      .mockImplementation(() => undefined)
    const bridge = new BrowserPlatformBridge()

    await bridge.saveExport('result.mp4', {
      kind: 'browser-download',
      blob: new Blob(['video']),
    })

    expect(createObjectURL).toHaveBeenCalledOnce()
    expect(click).toHaveBeenCalledOnce()
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:export')
  })
})

describe('browser and Tauri composition roots', () => {
  it('connects from a browser page and clears the session token input', async () => {
    const client = fakeClient()
    const createClient = vi.fn(() => client)
    const user = userEvent.setup()
    const storage = vi.spyOn(Storage.prototype, 'setItem')

    render(createElement(BrowserCompositionRoot, { createClient }))
    expect(screen.getByRole('main')).toHaveClass('connection-screen')
    expect(screen.getByRole('form', { name: '连接本地服务' })).toHaveClass('connection-card')
    expect(screen.getByText(/一次性会话令牌只保存在当前内存中/)).toBeInTheDocument()
    const port = screen.getByRole('textbox', { name: 'Local API port' })
    const token = screen.getByLabelText('Session token') as HTMLInputElement
    await user.type(port, '49152')
    await user.type(token, 'one-time-secret')
    await user.click(screen.getByRole('button', { name: 'Connect' }))

    await screen.findByText('Connected to local service')
    expect(createClient).toHaveBeenCalledWith({
      origin: 'http://127.0.0.1:49152',
      token: 'one-time-secret',
    })
    expect(token.value).toBe('')
    expect(storage).not.toHaveBeenCalled()
  })

  it('keeps the connection page visible when bootstrap fails', async () => {
    const client = fakeClient()
    vi.mocked(client.bootstrap).mockRejectedValue(new Error('offline'))

    render(createElement(BrowserCompositionRoot, { createClient: () => client }))
    fireEvent.change(screen.getByRole('textbox', { name: 'Local API port' }), {
      target: { value: '49152' },
    })
    fireEvent.change(screen.getByLabelText('Session token'), {
      target: { value: 'secret' },
    })
    fireEvent.click(screen.getByRole('button', { name: 'Connect' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('确认本地服务正在运行')
    expect(screen.getByRole('button', { name: 'Connect' })).toBeEnabled()
  })

  it('uses injected in-memory configuration without showing the connection page', async () => {
    const client = fakeClient()
    const createClient = vi.fn(() => client)

    render(
      createElement(TauriCompositionRoot, {
        session: {
          origin: 'http://127.0.0.1:49152',
          token: 'injected-secret',
        },
        createClient,
      }),
    )

    await waitFor(() => expect(client.bootstrap).toHaveBeenCalledOnce())
    expect(screen.queryByRole('button', { name: 'Connect' })).toBeNull()
    expect(screen.queryByText('injected-secret')).toBeNull()
    expect(await screen.findByText('Connected to local service')).toBeVisible()
  })
})
