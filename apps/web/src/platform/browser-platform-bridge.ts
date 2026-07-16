import type {
  ExportSource,
  PickedFile,
  PickFileOptions,
  PlatformBridge,
} from './platform-bridge'

export class BrowserPlatformBridge implements PlatformBridge {
  readonly kind = 'browser' as const

  pickInputFile(options: PickFileOptions): Promise<PickedFile | null> {
    return new Promise((resolve) => {
      const input = document.createElement('input')
      input.type = 'file'
      input.accept = options.extensions.map((extension) => `.${extension}`).join(',')
      const finish = (): void => {
        const file = input.files?.item(0) ?? null
        resolve(file === null ? null : { kind: 'browser-file', file })
      }
      input.addEventListener('change', finish, { once: true })
      input.addEventListener('cancel', () => resolve(null), { once: true })
      input.click()
    })
  }

  async saveExport(suggestedName: string, source: ExportSource): Promise<void> {
    if (source.kind !== 'browser-download') {
      throw new TypeError('Browser exports require a Blob source')
    }
    const url = URL.createObjectURL(source.blob)
    try {
      const anchor = document.createElement('a')
      anchor.download = suggestedName
      anchor.href = url
      anchor.rel = 'noopener'
      anchor.click()
    } finally {
      URL.revokeObjectURL(url)
    }
  }

  async openExternal(value: string): Promise<void> {
    const url = new URL(value)
    if (url.protocol !== 'http:' && url.protocol !== 'https:') {
      throw new TypeError('Only HTTP external links are supported')
    }
    window.open(url, '_blank', 'noopener,noreferrer')
  }
}
