import '@testing-library/jest-dom/vitest'

import { cleanup } from '@testing-library/react'
import { afterEach, vi } from 'vitest'

afterEach(() => {
  cleanup()
})

if (URL.createObjectURL === undefined) {
  URL.createObjectURL = vi.fn(() => 'blob:test')
}
if (URL.revokeObjectURL === undefined) {
  URL.revokeObjectURL = vi.fn()
}
