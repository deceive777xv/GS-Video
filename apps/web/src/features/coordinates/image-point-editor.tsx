import {
  type ChangeEvent,
  type RefObject,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from 'react'

import {
  type ImagePoint,
  type ImageSize,
  imagePointToViewport,
  sameImagePoint,
  validateImagePoint,
} from './image-point'

function pointText(point: ImagePoint | null, axis: 'x' | 'y'): string {
  return point === null ? '' : String(point[axis])
}

export interface ImagePointDraft {
  xText: string
  yText: string
  point: ImagePoint | null
  error: string | null
  dirty: boolean
  setXText(value: string): void
  setYText(value: string): void
  select(point: ImagePoint): void
  accept(point: ImagePoint | null): void
}

export function useImagePointDraft(
  image: ImageSize | null,
  authority: string,
  initialPoint: ImagePoint | null,
): ImagePointDraft {
  const [baseline, setBaseline] = useState<ImagePoint | null>(initialPoint)
  const [xText, setXText] = useState(() => pointText(initialPoint, 'x'))
  const [yText, setYText] = useState(() => pointText(initialPoint, 'y'))

  useEffect(() => {
    setBaseline(initialPoint)
    setXText(pointText(initialPoint, 'x'))
    setYText(pointText(initialPoint, 'y'))
  }, [authority, image?.height, image?.width, initialPoint?.x, initialPoint?.y])

  const validation = useMemo(
    () => validateImagePoint(xText, yText, image),
    [image, xText, yText],
  )
  const dirty = validation.error !== null
    || !sameImagePoint(validation.point, baseline)
    || (validation.point === null && (xText !== '' || yText !== ''))

  const select = useCallback((point: ImagePoint): void => {
    setXText(String(point.x))
    setYText(String(point.y))
  }, [])
  const accept = useCallback((point: ImagePoint | null): void => {
    setBaseline(point)
    setXText(pointText(point, 'x'))
    setYText(pointText(point, 'y'))
  }, [])
  return {
    xText,
    yText,
    point: validation.point,
    error: validation.error,
    dirty,
    setXText,
    setYText,
    select,
    accept,
  }
}

interface ImagePointMarkerProps {
  containerRef: RefObject<HTMLElement | null>
  mediaRef: RefObject<HTMLElement | null>
  image: ImageSize | null
  point: ImagePoint | null
  pending: boolean
}

export function ImagePointMarker({ containerRef, mediaRef, image, point, pending }: ImagePointMarkerProps) {
  const [position, setPosition] = useState<ImagePoint | null>(null)

  useEffect(() => {
    const container = containerRef.current
    const media = mediaRef.current
    if (container === null || media === null || image === null || point === null) {
      setPosition(null)
      return
    }
    const update = (): void => {
      const containerBounds = container.getBoundingClientRect()
      const mediaBounds = media.getBoundingClientRect()
      const mediaPosition = imagePointToViewport(point, mediaBounds, image)
      setPosition(mediaPosition === null ? null : {
        x: mediaBounds.left - containerBounds.left + mediaPosition.x,
        y: mediaBounds.top - containerBounds.top + mediaPosition.y,
      })
    }
    update()
    const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(update)
    observer?.observe(container)
    observer?.observe(media)
    window.addEventListener('resize', update)
    return () => {
      observer?.disconnect()
      window.removeEventListener('resize', update)
    }
  }, [containerRef, image, mediaRef, point])

  if (point === null || position === null || image === null) return null
  const edgeClasses = [
    point.x >= image.width * 0.7 ? 'is-right-edge' : '',
    point.y < image.height * 0.18 ? 'is-top-edge' : '',
  ].filter(Boolean).join(' ')
  return (
    <span
      aria-hidden="true"
      className={`${pending ? 'image-point-marker is-pending' : 'image-point-marker is-confirmed'} ${edgeClasses}`.trim()}
      style={{ left: `${position.x}px`, top: `${position.y}px` }}
    >
      <span className="image-point-crosshair" />
      <span className="image-point-label">X {point.x} · Y {point.y}</span>
    </span>
  )
}

interface ImagePointFieldsProps {
  draft: ImagePointDraft
  image: ImageSize | null
  disabled?: boolean
  xLabel: string
  yLabel: string
  emptyLabel?: string
  pendingLabel?: string
  confirmedLabel?: string
}

export function ImagePointFields({
  draft,
  image,
  disabled = false,
  xLabel,
  yLabel,
  emptyLabel = '尚未选择坐标',
  pendingLabel = '候选坐标待确认',
  confirmedLabel = '坐标已确认',
}: ImagePointFieldsProps) {
  const changeX = (event: ChangeEvent<HTMLInputElement>): void => draft.setXText(event.currentTarget.value)
  const changeY = (event: ChangeEvent<HTMLInputElement>): void => draft.setYText(event.currentTarget.value)
  const maxX = image === null ? undefined : image.width - 1
  const maxY = image === null ? undefined : image.height - 1
  const status = draft.point === null
    ? emptyLabel
    : draft.dirty ? pendingLabel : confirmedLabel

  return (
    <div className="image-point-editor">
      <p className="coordinate-origin">原点在左上角；X 向右，Y 向下。</p>
      <div className="coordinate-fields">
        <label>
          <span>{xLabel}</span>
          <small>{maxX === undefined ? '等待图像尺寸' : `有效范围 0–${maxX}`}</small>
          <input
            aria-label={xLabel}
            disabled={disabled || image === null}
            inputMode="numeric"
            max={maxX}
            min="0"
            onChange={changeX}
            step="1"
            type="number"
            value={draft.xText}
          />
        </label>
        <label>
          <span>{yLabel}</span>
          <small>{maxY === undefined ? '等待图像尺寸' : `有效范围 0–${maxY}`}</small>
          <input
            aria-label={yLabel}
            disabled={disabled || image === null}
            inputMode="numeric"
            max={maxY}
            min="0"
            onChange={changeY}
            step="1"
            type="number"
            value={draft.yText}
          />
        </label>
      </div>
      <p
        aria-live="polite"
        className={draft.error === null ? 'coordinate-status' : 'coordinate-status is-error'}
      >
        {draft.error ?? status}
      </p>
    </div>
  )
}
