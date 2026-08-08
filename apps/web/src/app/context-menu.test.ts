import { describe, expect, it } from 'vitest'

import { installContextMenuGuard } from './context-menu'

describe('installContextMenuGuard', () => {
  it('prevents the native menu until the guard is removed', () => {
    const remove = installContextMenuGuard(document)
    const blocked = new MouseEvent('contextmenu', { bubbles: true, cancelable: true })
    document.body.dispatchEvent(blocked)
    expect(blocked.defaultPrevented).toBe(true)

    remove()
    const allowed = new MouseEvent('contextmenu', { bubbles: true, cancelable: true })
    document.body.dispatchEvent(allowed)
    expect(allowed.defaultPrevented).toBe(false)
  })
})
