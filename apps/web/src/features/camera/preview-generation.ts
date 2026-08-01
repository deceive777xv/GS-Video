let lastPreviewGeneration = 0

export function nextPreviewGeneration(floor = 0): number {
  const generation = Math.max(
    lastPreviewGeneration + 1,
    Math.trunc(floor) + 1,
    Math.trunc(Date.now()),
    1,
  )
  if (!Number.isSafeInteger(generation)) {
    throw new Error('Preview generation exceeded the safe integer range.')
  }
  lastPreviewGeneration = generation
  return generation
}
