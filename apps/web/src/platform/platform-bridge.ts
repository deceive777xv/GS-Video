import type { AssetDto, AssetKind } from '../api/types'

export interface PickFileOptions {
  kind: AssetKind
  extensions: string[]
  description?: string
  assignToCurrent?: boolean
}

export type PickedFile =
  | { kind: 'browser-file'; file: File }
  | { kind: 'local-asset'; asset: AssetDto }

export type ExportSource =
  | { kind: 'browser-download'; blob: Blob }
  | {
      kind: 'local-export'
      path: string
      saveTo(destination: string): Promise<void>
    }

export interface PlatformBridge {
  readonly kind: 'browser' | 'tauri'
  pickInputFile(options: PickFileOptions): Promise<PickedFile | null>
  saveExport(suggestedName: string, source: ExportSource): Promise<void>
  revealPath?(path: string): Promise<void>
  openExternal(url: string): Promise<void>
}
