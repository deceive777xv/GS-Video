import { describe, expect, it } from 'vitest'

import { localDevServer, localProxyTarget } from './vite.config'

describe('desktop development server', () => {
  it('starts on one strict loopback origin without requiring an API proxy', () => {
    const server = localDevServer()

    expect(server).toMatchObject({
      host: '127.0.0.1',
      port: 1420,
      strictPort: true,
    })
    expect('proxy' in server).toBe(false)
  })

  it('keeps the optional browser proxy on a validated loopback target', () => {
    const server = localDevServer('http://localhost:43210')

    expect(server.proxy?.['/api']).toMatchObject({
      target: 'http://localhost:43210',
      ws: true,
    })
  })

  it.each([
    'https://127.0.0.1:43210',
    'http://example.com:43210',
    'http://user@127.0.0.1:43210',
  ])('rejects unsafe proxy target %s', (target) => {
    expect(() => localProxyTarget(target)).toThrow(
      'GS_VIDEO_DEV_API_ORIGIN must be an HTTP loopback origin',
    )
  })
})
