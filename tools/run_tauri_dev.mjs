import { spawn } from 'node:child_process'
import { existsSync } from 'node:fs'
import { delimiter, dirname, join } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const root = dirname(dirname(fileURLToPath(import.meta.url)))
let environment = { ...process.env }
const cargoHome = join(root, '.runtime', 'cargo')
const rustupHome = join(root, '.runtime', 'rustup')
const cargoBinary = join(cargoHome, 'bin', process.platform === 'win32' ? 'cargo.exe' : 'cargo')

export function prependExecutablePath(
  baseEnvironment,
  directory,
  platform = process.platform,
) {
  const updated = { ...baseEnvironment }
  const pathKeys = Object.keys(updated).filter(
    (key) => key.toLowerCase() === 'path',
  )
  const preferredKey =
    platform === 'win32'
      ? (pathKeys.find((key) => key === 'Path') ?? pathKeys[0] ?? 'Path')
      : (pathKeys.find((key) => key === 'PATH') ?? pathKeys[0] ?? 'PATH')
  const originalPath = updated[preferredKey] ?? ''
  for (const key of pathKeys) delete updated[key]
  updated[preferredKey] = `${directory}${delimiter}${originalPath}`
  return updated
}

if (existsSync(cargoBinary)) {
  environment.CARGO_HOME = cargoHome
  environment.RUSTUP_HOME = rustupHome
  environment.CARGO_TARGET_DIR = join(root, '.runtime', 'cargo-target', 'desktop')
  environment = prependExecutablePath(
    environment,
    join(cargoHome, 'bin'),
  )
}

export function npmInvocation({ nodeBinary, npmCli }) {
  if (npmCli === undefined || npmCli === '') {
    throw new Error(
      'npm_execpath is unavailable; start the desktop host with `npm run tauri:dev`',
    )
  }
  return {
    program: nodeBinary,
    args: [
      npmCli,
      'run',
      'tauri',
      '--workspace',
      '@gs-video/desktop',
      '--',
      'dev',
    ],
    options: { shell: false },
  }
}

function main() {
  const invocation = npmInvocation({
    nodeBinary: process.execPath,
    npmCli: process.env.npm_execpath,
  })
  if (!existsSync(invocation.args[0])) {
    throw new Error(`npm CLI entrypoint is unavailable: ${invocation.args[0]}`)
  }
  const child = spawn(invocation.program, invocation.args, {
    cwd: root,
    env: environment,
    stdio: 'inherit',
    windowsHide: true,
    ...invocation.options,
  })

  child.on('error', (error) => {
    console.error(`Unable to start the Tauri development host: ${error.message}`)
    process.exitCode = 1
  })

  child.on('exit', (code, signal) => {
    if (signal !== null) {
      process.kill(process.pid, signal)
      return
    }
    process.exitCode = code ?? 1
  })
}

if (
  process.argv[1] !== undefined &&
  import.meta.url === pathToFileURL(process.argv[1]).href
) {
  main()
}
