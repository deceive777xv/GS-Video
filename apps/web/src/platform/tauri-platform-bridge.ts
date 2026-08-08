import type { BackendClient } from '../api/backend-client'
import type {
  ExportSource,
  PickedFile,
  PickFileOptions,
  PlatformBridge,
} from './platform-bridge'

interface DialogApi {
  open(options: {
    multiple: false
    directory: false
    filters: { name: string; extensions: string[] }[]
  }): Promise<string | string[] | null>
  save(options: { defaultPath: string }): Promise<string | null>
}

interface OpenerApi {
  openUrl(url: string): Promise<void>
  revealItemInDir(path: string | string[]): Promise<void>
}

interface TauriPlatformLoaders {
  loadDialog(): Promise<DialogApi>
  loadOpener(): Promise<OpenerApi>
}

const defaultLoaders: TauriPlatformLoaders = {
  loadDialog: () => import('@tauri-apps/plugin-dialog'),
  loadOpener: () => import('@tauri-apps/plugin-opener'),
}

export class TauriPlatformBridge implements PlatformBridge {
  readonly kind = 'tauri' as const
  readonly #client: BackendClient
  readonly #loaders: TauriPlatformLoaders

  constructor(
    client: BackendClient,
    loaders: TauriPlatformLoaders = defaultLoaders,
  ) {
    this.#client = client
    this.#loaders = loaders
  }

  async pickInputFile(options: PickFileOptions): Promise<PickedFile | null> {
    const dialog = await this.#loaders.loadDialog()
    const selected = await dialog.open({
      multiple: false,
      directory: false,
      filters: [
        {
          name: options.description ?? 'Input file',
          extensions: options.extensions,
        },
      ],
    })
    const path = Array.isArray(selected) ? selected[0] : selected
    if (path === undefined || path === null) return null
    const asset = options.assignToCurrent === undefined
      ? await this.#client.importLocalPath(options.kind, path)
      : await this.#client.importLocalPath(options.kind, path, options.assignToCurrent)
    return { kind: 'local-asset', asset }
  }

  async saveExport(suggestedName: string, source: ExportSource): Promise<void> {
    if (source.kind !== 'local-export') {
      throw new TypeError('Desktop exports require a local export source')
    }
    const dialog = await this.#loaders.loadDialog()
    const destination = await dialog.save({ defaultPath: suggestedName })
    if (destination !== null) await source.saveTo(destination)
  }

  async revealPath(path: string): Promise<void> {
    const opener = await this.#loaders.loadOpener()
    await opener.revealItemInDir(path)
  }

  async openExternal(url: string): Promise<void> {
    const parsed = new URL(url)
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      throw new TypeError('Only HTTP external links are supported')
    }
    const opener = await this.#loaders.loadOpener()
    await opener.openUrl(parsed.toString())
  }
}
