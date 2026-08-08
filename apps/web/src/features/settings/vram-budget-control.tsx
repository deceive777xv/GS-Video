import { useEffect, useRef, useState } from 'react'

import type { BackendClient } from '../../api/backend-client'
import { BackendClientError } from '../../api/http-backend-client'
import type {
  VramBudgetDto,
  VramBudgetMode,
  VramBudgetUpdate,
} from '../../api/types'

interface VramBudgetControlProps {
  backend: Pick<BackendClient, 'getVramBudget' | 'updateVramBudget'>
  budget: VramBudgetDto
  busy: boolean
  onBudgetChange: (budget: VramBudgetDto) => void
}

function formatGib(mib: number): string {
  const value = mib / 1024
  return Number.isInteger(value)
    ? value.toLocaleString()
    : value.toFixed(2).replace(/0+$/, '').replace(/\.$/, '')
}

function inputGib(mib: number): string {
  return Number((mib / 1024).toFixed(4)).toString()
}

function errorMessage(value: unknown): string {
  if (value instanceof BackendClientError) return value.message
  return value instanceof Error ? value.message : '显存设置未能保存，请重试。'
}

function blockedMessage(reason: string | null, busy: boolean): string | null {
  if (busy || reason === 'gpu_task_active') return 'GPU 任务结束后可修改。'
  if (reason === 'environment_repair_active') return '环境修复结束后可修改。'
  if (reason === 'update_in_progress') return '另一项显存设置正在保存。'
  if (reason === 'gpu_unavailable') return '当前无法检测 GPU 物理总显存。'
  if (reason !== null) return '当前运行环境不支持修改显存预算。'
  return null
}

export function VramBudgetControl({
  backend,
  budget,
  busy,
  onBudgetChange,
}: VramBudgetControlProps) {
  const [open, setOpen] = useState(false)
  const [mode, setMode] = useState<VramBudgetMode>(budget.mode)
  const [customGib, setCustomGib] = useState(inputGib(budget.selected_vram_mb))
  const [saving, setSaving] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const rootRef = useRef<HTMLDivElement>(null)
  const buttonRef = useRef<HTMLButtonElement>(null)
  const panelRef = useRef<HTMLDivElement>(null)

  const closeAndRestoreFocus = (): void => {
    setOpen(false)
    buttonRef.current?.focus()
  }

  const syncDraft = (snapshot: VramBudgetDto): void => {
    setMode(snapshot.mode)
    setCustomGib(inputGib(snapshot.selected_vram_mb))
  }

  useEffect(() => {
    syncDraft(budget)
  }, [budget])

  useEffect(() => {
    if (!open) return
    const onKeyDown = (event: KeyboardEvent): void => {
      if (event.key !== 'Escape') return
      closeAndRestoreFocus()
    }
    const onPointerDown = (event: MouseEvent): void => {
      if (rootRef.current?.contains(event.target as Node) === false) setOpen(false)
    }
    document.addEventListener('keydown', onKeyDown)
    document.addEventListener('mousedown', onPointerDown)
    queueMicrotask(() => panelRef.current?.querySelector<HTMLElement>('input')?.focus())
    return () => {
      document.removeEventListener('keydown', onKeyDown)
      document.removeEventListener('mousedown', onPointerDown)
    }
  }, [open])

  const toggle = (): void => {
    if (open) {
      setOpen(false)
      return
    }
    setOpen(true)
    setError(null)
    setLoading(true)
    void backend.getVramBudget().then((snapshot) => {
      syncDraft(snapshot)
      onBudgetChange(snapshot)
    }).catch((value: unknown) => setError(errorMessage(value))).finally(() => setLoading(false))
  }

  const numericGib = Number(customGib)
  const customMib = Number.isFinite(numericGib) ? Math.round(numericGib * 1024) : 0
  const customValid = Number.isFinite(numericGib)
    && numericGib > 0
    && (Number.isInteger(numericGib * 1024) || customMib === budget.total_vram_mb)
    && customMib >= budget.minimum_vram_mb
    && customMib <= budget.total_vram_mb
    && (customMib % 256 === 0 || customMib === budget.total_vram_mb)
  const disabled = busy || !budget.editable || saving || loading
  const message = blockedMessage(budget.blocked_reason, busy)

  const apply = async (): Promise<void> => {
    if (disabled || (mode === 'custom' && !customValid)) return
    setSaving(true)
    setError(null)
    const update: VramBudgetUpdate = mode === 'standard'
      ? { mode: 'standard', selected_vram_mb: null }
      : { mode: 'custom', selected_vram_mb: customMib }
    try {
      const snapshot = await backend.updateVramBudget(update)
      onBudgetChange(snapshot)
      syncDraft(snapshot)
      setOpen(false)
      buttonRef.current?.focus()
    } catch (value) {
      setError(errorMessage(value))
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="vram-budget-control" ref={rootRef}>
      <button
        aria-expanded={open}
        aria-haspopup="dialog"
        className="vram-budget-trigger"
        onClick={toggle}
        ref={buttonRef}
        type="button"
      >
        显存 {formatGib(budget.selected_vram_mb)} GB
      </button>
      {open ? (
        <div
          aria-label="显存模式"
          className="vram-budget-panel"
          ref={panelRef}
          role="dialog"
        >
          <div className="vram-budget-heading">
            <span>显存模式</span>
            <strong>{formatGib(budget.selected_vram_mb)} GB</strong>
          </div>
          <dl className="vram-budget-stats">
            <div><dt>物理总显存</dt><dd>{formatGib(budget.total_vram_mb)} GB</dd></div>
            <div><dt>允许范围</dt><dd>{formatGib(budget.minimum_vram_mb)}–{formatGib(budget.total_vram_mb)} GB</dd></div>
          </dl>
          {budget.recovered_from_invalid_preference ? (
            <p className="vram-budget-notice" role="status">原显存偏好无效，已恢复安全默认值。</p>
          ) : null}
          {message !== null ? <p className="vram-budget-notice">{message}</p> : null}
          <fieldset disabled={disabled}>
            <legend>选择模式</legend>
            <label>
              <input
                checked={mode === 'standard'}
                name="vram-mode"
                onChange={() => setMode('standard')}
                type="radio"
              />
              <span><strong>标准 8 GB</strong><small>低于 8 GB 的显卡自动使用全部显存</small></span>
            </label>
            <label>
              <input
                checked={mode === 'custom'}
                name="vram-mode"
                onChange={() => setMode('custom')}
                type="radio"
              />
              <span><strong>自定义</strong><small>按当前设备调整 GPU 资源预算</small></span>
            </label>
          </fieldset>
          <div className="vram-custom-controls">
            <label htmlFor="vram-custom-gib">可用显存（GB）</label>
            <div>
              <input
                aria-describedby="vram-custom-help"
                disabled={disabled || mode !== 'custom'}
                id="vram-custom-gib"
                inputMode="decimal"
                max={budget.total_vram_mb / 1024}
                min={budget.minimum_vram_mb / 1024}
                onChange={(event) => setCustomGib(event.target.value)}
                step="0.25"
                type="number"
                value={customGib}
              />
              <button
                className="button-secondary"
                disabled={disabled || mode !== 'custom'}
                onClick={() => setCustomGib(inputGib(budget.total_vram_mb))}
                type="button"
              >
                全部显存
              </button>
            </div>
            <input
              aria-label="显存预算滑块"
              disabled={disabled || mode !== 'custom'}
              max={budget.total_vram_mb}
              min={budget.minimum_vram_mb}
              onChange={(event) => setCustomGib(inputGib(Number(event.target.value)))}
              step="256"
              type="range"
              value={Math.min(budget.total_vram_mb, Math.max(budget.minimum_vram_mb, customMib))}
            />
            <small id="vram-custom-help">常规步进 0.25 GB；“全部显存”使用检测到的精确总量。</small>
          </div>
          {mode === 'custom' && customGib !== '' && !customValid ? (
            <p className="vram-budget-error" role="alert">请输入允许范围内、以 0.25 GB 递增的数值。</p>
          ) : null}
          {error !== null ? <p className="vram-budget-error" role="alert">{error}</p> : null}
          <div className="vram-budget-actions">
            <button className="button-secondary" onClick={closeAndRestoreFocus} type="button">取消</button>
            <button
              disabled={disabled || (mode === 'custom' && !customValid)}
              onClick={() => void apply()}
              type="button"
            >
              {saving ? '正在应用…' : '应用'}
            </button>
          </div>
        </div>
      ) : null}
    </div>
  )
}
