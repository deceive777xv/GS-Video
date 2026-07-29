import assert from 'node:assert/strict'
import test from 'node:test'

import {
  npmInvocation,
  prependExecutablePath,
} from '../../../tools/run_tauri_dev.mjs'

test('uses the npm CLI JavaScript entrypoint instead of spawning npm.cmd', () => {
  const invocation = npmInvocation({
    nodeBinary: 'C:\\Program Files\\nodejs\\node.exe',
    npmCli: 'C:\\Program Files\\nodejs\\node_modules\\npm\\bin\\npm-cli.js',
  })

  assert.equal(invocation.program, 'C:\\Program Files\\nodejs\\node.exe')
  assert.deepEqual(invocation.args, [
    'C:\\Program Files\\nodejs\\node_modules\\npm\\bin\\npm-cli.js',
    'run',
    'tauri',
    '--workspace',
    '@gs-video/desktop',
    '--',
    'dev',
  ])
  assert.equal(invocation.options.shell, false)
})

test('rejects launch outside an npm-managed script', () => {
  assert.throws(
    () => npmInvocation({ nodeBinary: 'node.exe', npmCli: undefined }),
    /npm_execpath/,
  )
})

test('preserves a Windows Path key while prepending project Cargo', () => {
  const environment = prependExecutablePath(
    {
      Path: 'C:\\Program Files\\nodejs;C:\\Windows',
      npm_execpath: 'C:\\npm-cli.js',
    },
    'E:\\Project\\GS-Video\\.runtime\\cargo\\bin',
    'win32',
  )

  assert.deepEqual(
    Object.keys(environment).filter((key) => key.toLowerCase() === 'path'),
    ['Path'],
  )
  assert.equal(
    environment.Path,
    'E:\\Project\\GS-Video\\.runtime\\cargo\\bin;C:\\Program Files\\nodejs;C:\\Windows',
  )
})
