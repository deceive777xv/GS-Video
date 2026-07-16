import {
  createContext,
  type FormEvent,
  type ReactElement,
  type ReactNode,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react'

import type { BackendClient } from './api/backend-client'
import { HttpBackendClient } from './api/http-backend-client'
import type { SessionConfig } from './api/types'
import { BrowserPlatformBridge } from './platform/browser-platform-bridge'
import type { PlatformBridge } from './platform/platform-bridge'
import { TauriPlatformBridge } from './platform/tauri-platform-bridge'

export type BackendClientFactory = (session: SessionConfig) => BackendClient

const defaultClientFactory: BackendClientFactory = (session) =>
  new HttpBackendClient(session)

const BackendClientContext = createContext<BackendClient | null>(null)
const PlatformBridgeContext = createContext<PlatformBridge | null>(null)

export function useBackendClient(): BackendClient {
  const client = useContext(BackendClientContext)
  if (client === null) throw new Error('BackendClient is not available')
  return client
}

export function usePlatformBridge(): PlatformBridge {
  const bridge = useContext(PlatformBridgeContext)
  if (bridge === null) throw new Error('PlatformBridge is not available')
  return bridge
}

function RuntimeProviders({
  client,
  bridge,
  children,
}: {
  client: BackendClient
  bridge: PlatformBridge
  children: ReactNode
}): ReactElement {
  return (
    <BackendClientContext value={client}>
      <PlatformBridgeContext value={bridge}>{children}</PlatformBridgeContext>
    </BackendClientContext>
  )
}

function ConnectedShell(): ReactElement {
  return (
    <main>
      <p>Connected to local service</p>
    </main>
  )
}

export function BrowserCompositionRoot({
  createClient = defaultClientFactory,
}: {
  createClient?: BackendClientFactory
}): ReactElement {
  const bridge = useMemo(() => new BrowserPlatformBridge(), [])
  const [port, setPort] = useState('')
  const [token, setToken] = useState('')
  const [client, setClient] = useState<BackendClient | null>(null)
  const [connecting, setConnecting] = useState(false)
  const [failed, setFailed] = useState(false)
  const portInput = useRef<HTMLInputElement>(null)
  const tokenInput = useRef<HTMLInputElement>(null)

  const connect = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault()
    const portNumber = Number(port)
    if (!Number.isInteger(portNumber) || portNumber < 1 || portNumber > 65_535) {
      setFailed(true)
      return
    }
    setConnecting(true)
    setFailed(false)
    const candidate = createClient({
      origin: `http://127.0.0.1:${String(portNumber)}`,
      token,
    })
    try {
      await candidate.bootstrap()
      if (portInput.current !== null) portInput.current.value = ''
      if (tokenInput.current !== null) tokenInput.current.value = ''
      setPort('')
      setToken('')
      setClient(candidate)
    } catch {
      setFailed(true)
    } finally {
      setConnecting(false)
    }
  }

  if (client !== null) {
    return (
      <RuntimeProviders client={client} bridge={bridge}>
        <ConnectedShell />
      </RuntimeProviders>
    )
  }
  return (
    <main>
      <h1>Connect to the local service</h1>
      <form onSubmit={(event) => void connect(event)}>
        <label>
          Local API port
          <input
            autoComplete="off"
            inputMode="numeric"
            name="port"
            onChange={(event) => setPort(event.currentTarget.value)}
            ref={portInput}
            value={port}
          />
        </label>
        <label>
          Session token
          <input
            autoComplete="off"
            name="token"
            onChange={(event) => setToken(event.currentTarget.value)}
            ref={tokenInput}
            type="password"
            value={token}
          />
        </label>
        <button disabled={connecting || token.length === 0} type="submit">
          {connecting ? 'Connecting…' : 'Connect'}
        </button>
      </form>
      {failed ? <p role="alert">Could not connect to the local service.</p> : null}
    </main>
  )
}

export function TauriCompositionRoot({
  session,
  createClient = defaultClientFactory,
}: {
  session: SessionConfig
  createClient?: BackendClientFactory
}): ReactElement {
  const [client] = useState(() => createClient(session))
  const bridge = useMemo(() => new TauriPlatformBridge(client), [client])
  const [connection, setConnection] = useState<'connecting' | 'ready' | 'failed'>(
    'connecting',
  )

  useEffect(() => {
    const controller = new AbortController()
    void client
      .bootstrap(controller.signal)
      .then(() => setConnection('ready'))
      .catch(() => {
        if (!controller.signal.aborted) setConnection('failed')
      })
    return () => controller.abort()
  }, [client])

  if (connection === 'failed') {
    return <p role="alert">Desktop local service unavailable.</p>
  }
  if (connection === 'connecting') return <p>Connecting to local service…</p>
  return (
    <RuntimeProviders client={client} bridge={bridge}>
      <ConnectedShell />
    </RuntimeProviders>
  )
}
