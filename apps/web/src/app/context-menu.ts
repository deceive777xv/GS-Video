export function installContextMenuGuard(target: Document): () => void {
  const preventNativeMenu = (event: MouseEvent): void => event.preventDefault()
  target.addEventListener('contextmenu', preventNativeMenu, { capture: true })
  return () => target.removeEventListener('contextmenu', preventNativeMenu, { capture: true })
}
