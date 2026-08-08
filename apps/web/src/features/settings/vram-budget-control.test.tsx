import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { expect, test, vi } from 'vitest'

import type { BackendClient } from '../../api/backend-client'
import type { VramBudgetDto } from '../../api/types'
import { VramBudgetControl } from './vram-budget-control'


function budget(overrides: Partial<VramBudgetDto> = {}): VramBudgetDto {
  return {
    mode: 'standard',
    minimum_vram_mb: 1024,
    total_vram_mb: 24_576,
    selected_vram_mb: 8192,
    editable: true,
    blocked_reason: null,
    recovered_from_invalid_preference: false,
    ...overrides,
  }
}


function backend(
  current: VramBudgetDto,
  updated: VramBudgetDto,
): Pick<BackendClient, 'getVramBudget' | 'updateVramBudget'> {
  return {
    getVramBudget: vi.fn().mockResolvedValue(current),
    updateVramBudget: vi.fn().mockResolvedValue(updated),
  }
}


test('applies a custom quarter-GiB budget and refreshes authority', async () => {
  const user = userEvent.setup()
  const current = budget()
  const updated = budget({ mode: 'custom', selected_vram_mb: 12_288 })
  const client = backend(current, updated)
  const onBudgetChange = vi.fn()
  const view = render(
    <VramBudgetControl
      backend={client}
      budget={current}
      busy={false}
      onBudgetChange={onBudgetChange}
    />,
  )

  await user.click(screen.getByRole('button', { name: '显存 8 GB' }))
  await user.click(screen.getByRole('radio', { name: /自定义/ }))
  const input = screen.getByRole('spinbutton', { name: '可用显存（GB）' })
  await user.clear(input)
  await user.type(input, '12')
  await user.click(screen.getByRole('button', { name: '应用' }))

  await waitFor(() => expect(client.updateVramBudget).toHaveBeenCalledWith({
    mode: 'custom',
    selected_vram_mb: 12_288,
  }))
  expect(onBudgetChange).toHaveBeenLastCalledWith(updated)

  view.rerender(
    <VramBudgetControl
      backend={client}
      budget={updated}
      busy={false}
      onBudgetChange={onBudgetChange}
    />,
  )
  expect(screen.getByRole('button', { name: '显存 12 GB' })).toBeInTheDocument()
})


test('all-memory action preserves a non-quarter-GiB physical total', async () => {
  const user = userEvent.setup()
  const current = budget({ total_vram_mb: 24_321 })
  const updated = budget({
    mode: 'custom',
    total_vram_mb: 24_321,
    selected_vram_mb: 24_321,
  })
  const client = backend(current, updated)

  render(
    <VramBudgetControl
      backend={client}
      budget={current}
      busy={false}
      onBudgetChange={vi.fn()}
    />,
  )

  await user.click(screen.getByRole('button', { name: '显存 8 GB' }))
  await user.click(screen.getByRole('radio', { name: /自定义/ }))
  await user.click(screen.getByRole('button', { name: '全部显存' }))
  await user.click(screen.getByRole('button', { name: '应用' }))

  await waitFor(() => expect(client.updateVramBudget).toHaveBeenCalledWith({
    mode: 'custom',
    selected_vram_mb: 24_321,
  }))
})


test('keeps settings viewable but disables edits while a GPU task is active', async () => {
  const user = userEvent.setup()
  const current = budget()
  const client = backend(current, current)

  render(
    <VramBudgetControl
      backend={client}
      budget={current}
      busy
      onBudgetChange={vi.fn()}
    />,
  )

  await user.click(screen.getByRole('button', { name: '显存 8 GB' }))

  expect(screen.getByText('GPU 任务结束后可修改。')).toBeInTheDocument()
  expect(screen.getByRole('radio', { name: /自定义/ })).toBeDisabled()
  expect(screen.getByRole('button', { name: '应用' })).toBeDisabled()
})


test('retains the draft and reports a save failure inline', async () => {
  const user = userEvent.setup()
  const current = budget()
  const client = backend(current, current)
  vi.mocked(client.updateVramBudget).mockRejectedValueOnce(new Error('磁盘不可用'))

  render(
    <VramBudgetControl
      backend={client}
      budget={current}
      busy={false}
      onBudgetChange={vi.fn()}
    />,
  )

  await user.click(screen.getByRole('button', { name: '显存 8 GB' }))
  await user.click(screen.getByRole('radio', { name: /自定义/ }))
  const input = screen.getByRole('spinbutton', { name: '可用显存（GB）' })
  await user.clear(input)
  await user.type(input, '12')
  await user.click(screen.getByRole('button', { name: '应用' }))

  expect(await screen.findByRole('alert')).toHaveTextContent('磁盘不可用')
  expect(input).toHaveValue(12)
})
