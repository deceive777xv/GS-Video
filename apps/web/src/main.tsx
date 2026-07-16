import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

import {
  BrowserCompositionRoot,
  TauriCompositionRoot,
} from './composition-root'
import type { SessionConfig } from './api/types'

declare global {
  interface Window {
    __GS_VIDEO_SESSION__?: SessionConfig
  }
}

const container = document.querySelector('#root')
if (container === null) throw new Error('The application root is missing')

const desktopSession = window.__GS_VIDEO_SESSION__
delete window.__GS_VIDEO_SESSION__

createRoot(container).render(
  <StrictMode>
    {desktopSession === undefined ? (
      <BrowserCompositionRoot />
    ) : (
      <TauriCompositionRoot session={desktopSession} />
    )}
  </StrictMode>,
)
