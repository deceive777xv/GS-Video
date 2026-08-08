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
import { WebSocketTaskEventSource } from './api/task-events'
import type { BootstrapDto, SessionConfig } from './api/types'
import { App } from './app/app'
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

function ConnectedShell({
  client,
  bridge,
  bootstrap,
  eventSource,
}: {
  client: BackendClient
  bridge: PlatformBridge
  bootstrap: BootstrapDto
  eventSource: WebSocketTaskEventSource
}): ReactElement {
  return (
    <>
      <p className="sr-only">Connected to local service</p>
      <App
        backend={client}
        eventSource={eventSource}
        initialBootstrap={bootstrap}
        platform={bridge}
        startAtHome
      />
    </>
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
  const [runtime, setRuntime] = useState<{
    client: BackendClient
    bootstrap: BootstrapDto
    eventSource: WebSocketTaskEventSource
  } | null>(null)
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
      const session = {
        origin: `http://127.0.0.1:${String(portNumber)}`,
        token,
      }
      const initialBootstrap = await candidate.bootstrap()
      if (portInput.current !== null) portInput.current.value = ''
      if (tokenInput.current !== null) tokenInput.current.value = ''
      setPort('')
      setToken('')
      setRuntime({
        client: candidate,
        bootstrap: initialBootstrap,
        eventSource: new WebSocketTaskEventSource(session),
      })
    } catch {
      setFailed(true)
    } finally {
      setConnecting(false)
    }
  }

  if (runtime !== null) {
    return (
      <RuntimeProviders client={runtime.client} bridge={bridge}>
        <ConnectedShell
          bootstrap={runtime.bootstrap}
          bridge={bridge}
          client={runtime.client}
          eventSource={runtime.eventSource}
        />
      </RuntimeProviders>
    )
  }
  return (
    <main className="connection-screen">
      <form
        aria-describedby="connection-help"
        aria-label="连接本地服务"
        className="connection-card"
        onSubmit={(event) => void connect(event)}
      >
        <div aria-hidden="true" className="connection-mark"><span /><span /><span /></div>
        <p className="eyebrow">LOCAL WORKFLOW</p>
        <h1>Connect to the local service</h1>
        <p className="connection-help" id="connection-help">
          输入桌面服务启动时显示的端口与一次性令牌。一次性会话令牌只保存在当前内存中，连接后立即清空输入框。
        </p>
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
        {failed ? <p className="connection-error" role="alert">无法连接。请确认本地服务正在运行，端口与一次性令牌仍然有效。</p> : null}
        <p className="connection-footnote">仅允许连接 127.0.0.1；不会把令牌写入浏览器存储。</p>
      </form>
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
  const eventSource = useMemo(() => new WebSocketTaskEventSource(session), [session])
  const [initialBootstrap, setInitialBootstrap] = useState<BootstrapDto | null>(null)
  const [connection, setConnection] = useState<'connecting' | 'ready' | 'failed'>(
    'connecting',
  )

  useEffect(() => {
    const controller = new AbortController()
    void client
      .bootstrap(controller.signal)
      .then((value) => {
        setInitialBootstrap(value)
        setConnection('ready')
      })
      .catch(() => {
        if (!controller.signal.aborted) setConnection('failed')
      })
    return () => controller.abort()
  }, [client])

  if (connection === 'failed') {
    return <p role="alert">Desktop local service unavailable.</p>
  }
  if (connection === 'connecting' || initialBootstrap === null) return <p>Connecting to local service…</p>
  return (
    <RuntimeProviders client={client} bridge={bridge}>
      <ConnectedShell
        bootstrap={initialBootstrap}
        bridge={bridge}
        client={client}
        eventSource={eventSource}
      />
    </RuntimeProviders>
  )
}
