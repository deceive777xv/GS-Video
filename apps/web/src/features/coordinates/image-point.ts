export interface ImageSize { width: number; height: number }
export interface ImagePoint { x: number; y: number }
export interface PointEvent { clientX: number; clientY: number }
export interface PointBounds { left: number; top: number; width: number; height: number }

export interface ContainedImageRect {
  left: number
  top: number
  width: number
  height: number
  scale: number
}

export interface ImagePointValidation {
  point: ImagePoint | null
  error: string | null
}

export function getContainedImageRect(bounds: PointBounds, image: ImageSize): ContainedImageRect | null {
  if (bounds.width <= 0 || bounds.height <= 0 || image.width <= 0 || image.height <= 0) return null
  const scale = Math.min(bounds.width / image.width, bounds.height / image.height)
  const width = image.width * scale
  const height = image.height * scale
  return {
    left: bounds.left + (bounds.width - width) / 2,
    top: bounds.top + (bounds.height - height) / 2,
    width,
    height,
    scale,
  }
}

export function toImagePoint(
  event: PointEvent,
  bounds: PointBounds,
  image: ImageSize,
): ImagePoint | null {
  const rect = getContainedImageRect(bounds, image)
  if (rect === null) return null
  const x = (event.clientX - rect.left) / rect.scale
  const y = (event.clientY - rect.top) / rect.scale
  if (x < 0 || y < 0 || x >= image.width || y >= image.height) return null
  return { x: Math.floor(x), y: Math.floor(y) }
}

export function imagePointToViewport(
  point: ImagePoint,
  bounds: PointBounds,
  image: ImageSize,
): ImagePoint | null {
  const rect = getContainedImageRect(bounds, image)
  if (rect === null || point.x < 0 || point.y < 0 || point.x >= image.width || point.y >= image.height) {
    return null
  }
  return {
    x: rect.left - bounds.left + (point.x + 0.5) * rect.scale,
    y: rect.top - bounds.top + (point.y + 0.5) * rect.scale,
  }
}

export function validateImagePoint(xText: string, yText: string, image: ImageSize | null): ImagePointValidation {
  if (image === null || image.width <= 0 || image.height <= 0) {
    return { point: null, error: null }
  }
  const trimmedX = xText.trim()
  const trimmedY = yText.trim()
  if (trimmedX === '' && trimmedY === '') return { point: null, error: null }
  if (trimmedX === '' || trimmedY === '') {
    return { point: null, error: '请同时输入 X 和 Y 坐标。' }
  }
  const x = Number(trimmedX)
  const y = Number(trimmedY)
  if (!Number.isInteger(x) || !Number.isInteger(y)) {
    return { point: null, error: 'X 和 Y 必须是整数像素坐标。' }
  }
  if (x < 0 || x >= image.width || y < 0 || y >= image.height) {
    return {
      point: null,
      error: `坐标超出图像范围：X 为 0–${image.width - 1}，Y 为 0–${image.height - 1}。`,
    }
  }
  return { point: { x, y }, error: null }
}

export function sameImagePoint(left: ImagePoint | null, right: ImagePoint | null): boolean {
  return left === null ? right === null : right !== null && left.x === right.x && left.y === right.y
}
