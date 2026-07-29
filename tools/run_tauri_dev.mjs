import { spawn } from 'node:child_process'
import { existsSync } from 'node:fs'
import { delimiter, dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = dirname(dirname(fileURLToPath(import.meta.url)))
const environment = { ...process.env }
const cargoHome = join(root, '.runtime', 'cargo')
const rustupHome = join(root, '.runtime', 'rustup')
const cargoBinary = join(cargoHome, 'bin', process.platform === 'win32' ? 'cargo.exe' : 'cargo')

if (existsSync(cargoBinary)) {
  environment.CARGO_HOME = cargoHome
  environment.RUSTUP_HOME = rustupHome
  environment.CARGO_TARGET_DIR = join(root, '.runtime', 'cargo-target', 'desktop')
  environment.PATH = `${join(cargoHome, 'bin')}${delimiter}${environment.PATH ?? ''}`
}

const npm = process.platform === 'win32' ? 'npm.cmd' : 'npm'
const child = spawn(
  npm,
  ['run', 'tauri', '--workspace', '@gs-video/desktop', '--', 'dev'],
  {
    cwd: root,
    env: environment,
    stdio: 'inherit',
    windowsHide: true,
  },
)

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
