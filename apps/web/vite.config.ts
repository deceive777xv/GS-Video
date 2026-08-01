import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

import react from '@vitejs/plugin-react'
import { defineConfig, type Plugin } from 'vite'

function sharedAppIcon(): Plugin {
  const appIconPath = fileURLToPath(
    new URL('../desktop/app-icon.svg', import.meta.url),
  )

  return {
    name: 'gs-video-shared-app-icon',
    configureServer(server) {
      server.middlewares.use('/app-icon.svg', (_request, response) => {
        response.statusCode = 200
        response.setHeader('Content-Type', 'image/svg+xml')
        response.end(readFileSync(appIconPath))
      })
    },
    generateBundle() {
      this.emitFile({
        type: 'asset',
        fileName: 'app-icon.svg',
        source: readFileSync(appIconPath),
      })
    },
  }
}

export function localProxyTarget(value: string): string {
  const url = new URL(value)
  const hostname = url.hostname.replace(/^\[|\]$/g, '')
  if (
    url.protocol !== 'http:' ||
    !['127.0.0.1', 'localhost', '::1'].includes(hostname) ||
    url.username !== '' ||
    url.password !== '' ||
    (url.pathname !== '' && url.pathname !== '/') ||
    url.search !== '' ||
    url.hash !== ''
  ) {
    throw new TypeError('GS_VIDEO_DEV_API_ORIGIN must be an HTTP loopback origin')
  }
  return url.origin
}

export function localDevServer(configuredTarget?: string) {
  const target =
    configuredTarget === undefined
      ? undefined
      : localProxyTarget(configuredTarget)
  return {
    host: '127.0.0.1',
    port: 1420,
    strictPort: true,
    ...(target === undefined
      ? {}
      : {
          proxy: {
            '/api': { target, ws: true as const },
            '/ws': {
              target,
              ws: true as const,
              rewrite: (requestPath: string) =>
                requestPath.replace(/^\/ws/, ''),
            },
          },
        }),
  }
}

export default defineConfig(() => {
  const configuredTarget = process.env.GS_VIDEO_DEV_API_ORIGIN

  return {
    base: './',
    plugins: [sharedAppIcon(), react()],
    server: localDevServer(configuredTarget),
  }
})
