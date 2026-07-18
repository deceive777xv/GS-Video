import { describe, expect, it, vi } from 'vitest'

import { HASH_CHUNK_SIZE, sha256Hex } from './incremental-sha256'

describe('incremental SHA-256', () => {
  it('matches the standard empty and abc vectors', async () => {
    await expect(sha256Hex(new Blob([]))).resolves.toBe(
      'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    )
    await expect(sha256Hex(new Blob(['abc']), 1)).resolves.toBe(
      'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad',
    )
  })

  it('keeps the digest exact across SHA block and input chunk boundaries', async () => {
    const bytes = new TextEncoder().encode('boundary-'.repeat(1000))
    const expected = Array.from(new Uint8Array(
      await crypto.subtle.digest('SHA-256', bytes),
    ), (byte) => byte.toString(16).padStart(2, '0')).join('')

    await expect(sha256Hex(new Blob([bytes]), 65)).resolves.toBe(expected)
  })

  it('never reads the whole Blob and bounds every slice', async () => {
    const sliceSizes: number[] = []
    class GuardedBlob extends Blob {
      override arrayBuffer(): Promise<ArrayBuffer> {
        throw new Error('whole blob arrayBuffer must not be called')
      }

      override slice(start?: number, end?: number, contentType?: string): Blob {
        sliceSizes.push((end ?? this.size) - (start ?? 0))
        return super.slice(start, end, contentType)
      }
    }
    const blob = new GuardedBlob([new Uint8Array(HASH_CHUNK_SIZE * 2 + 17)])
    const digest = await sha256Hex(blob)

    expect(digest).toHaveLength(64)
    expect(sliceSizes).toEqual([HASH_CHUNK_SIZE, HASH_CHUNK_SIZE, 17])
    expect(Math.max(...sliceSizes)).toBeLessThanOrEqual(HASH_CHUNK_SIZE)
    expect(vi.isMockFunction(blob.arrayBuffer)).toBe(false)
  })

  it('stops after an in-flight chunk read when its caller aborts', async () => {
    let release: ((value: ArrayBuffer) => void) | undefined
    class SlowBlob extends Blob {
      override slice(): Blob {
        return {
          arrayBuffer: () => new Promise<ArrayBuffer>((resolve) => { release = resolve }),
        } as Blob
      }
    }
    const controller = new AbortController()
    const digest = sha256Hex(new SlowBlob(['abc']), HASH_CHUNK_SIZE, controller.signal)
    await vi.waitFor(() => expect(release).toBeTypeOf('function'))

    controller.abort()
    release?.(new Uint8Array([97, 98, 99]).buffer)

    await expect(digest).rejects.toMatchObject({ name: 'AbortError' })
  })
})
