import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

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
    plugins: react(),
    server: localDevServer(configuredTarget),
  }
})
