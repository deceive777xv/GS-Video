# GS Video Tauri 开发版桌面宿主设计

- 文档状态：已确认，待实施规划
- 日期：2026-07-29
- 目标范围：开发环境中的一键桌面启动
- 关联文档：`2026-07-11-gs-video-mvp-design.md`、`2026-07-11-gs-video-mvp-implementation.md` Task 15

## 1. 目的

本文定义 GS Video MVP 的 Tauri 2 开发版桌面宿主。完成后，开发者在仓库根目录执行 `npm run tauri:dev`，即可启动 Vite、Tauri WebView 和项目本地 Python FastAPI 服务；桌面应用自动完成私密会话握手，不要求用户手工查找随机端口或输入一次性令牌。

本轮只验证桌面运行闭环，不产出 Windows 安装包。设计保留后端可执行程序边界，使后续把开发环境中的 Python 进程替换为 PyInstaller/Tauri `externalBin` 时，无需改变握手协议、前端会话模型或关闭顺序。

## 2. 验收范围

本轮必须做到：

- `npm run tauri:dev` 启动 Vite 和 Tauri 2 开发宿主；
- Rust 宿主启动仓库 `.venv` 中的 Python API；
- Python 在 `127.0.0.1` 上使用系统分配的随机端口；
- 会话令牌只通过子进程 stdin 传递；
- Python 通过单行 JSON 返回端口、API 版本和进程 ID；
- Rust 验证握手并使用令牌调用 `/healthz`；
- 健康检查成功后，主 WebView 获得内存会话并显示；
- 关闭应用时先执行 API/worker 的正常清理，超时后回收子进程树；
- 启动失败、握手损坏和健康检查超时均产生可理解的诊断；
- 普通浏览器模式及其手工连接页面保持可用。

本轮不包含：

- PyInstaller onefile/onedir 构建；
- Tauri `externalBin` 发布包；
- NSIS、MSI、签名、自动更新或安装器测试；
- 自动下载缺失的 Python、Node、Rust、模型或 GPU 依赖；
- 面向最终用户的运行时配置编辑界面；
- sidecar 崩溃后的自动任务重启。

## 3. 方案选择

### 3.1 采用：Rust 生命周期管理器启动开发 Python

Tauri Rust 宿主直接启动仓库 `.venv/Scripts/python.exe`，传入显式的运行配置路径和握手开关。Rust 持有子进程句柄、stdin、stdout 和退出状态，负责启动超时、健康检查及退出清理。

后端启动命令由一个小型接口或数据结构提供。开发实现解析仓库根目录和 `.venv`；未来发布实现改为解析 Tauri sidecar 路径。生命周期管理器只依赖“可执行文件、固定参数、工作目录”以及统一握手协议，不依赖 Python 或 PyInstaller 的具体布局。

### 3.2 暂不采用：现在构建 PyInstaller sidecar

该方案最接近发布形态，但会把 PyInstaller hidden imports、目标三元组命名、二进制清单和 Tauri bundle 验证提前带入本轮。当前目标是先验证桌面宿主闭环，因此延后到单独的发布任务。

### 3.3 拒绝：npm/PowerShell 编排多个长期进程

外部脚本虽然更快，但难以可靠拥有 Python 子进程树，也容易迫使端口或令牌通过临时文件、环境变量或控制台交接。它不能满足既定的私密握手和退出回收要求。

## 4. 总体结构

```text
npm run tauri:dev
├─ Tauri CLI beforeDevCommand → Vite dev server（127.0.0.1:1420，strictPort）
└─ Tauri Rust host
   ├─ 解析开发后端命令与 runtime config
   ├─ 生成一次性高熵 token
   ├─ spawn Python API
   ├─ stdin 写入 token 后立即关闭
   ├─ 解析 stdout 单行握手
   ├─ 认证请求 /healthz
   ├─ 创建带内存 session 的主 WebView
   └─ 退出时关闭 API 与整个子进程树
```

共享 React/Vite 应用继续通过 `BackendClient` 使用 HTTP/WebSocket API。Tauri 不承载项目业务命令；桌面专有的文件对话框和 opener 仍只存在于 `TauriPlatformBridge`。

## 5. 开发运行配置

桌面开发启动使用位于 `.runtime` 下、被 Git 忽略的生成配置。仓库提交一个确定性的准备脚本，负责：

1. 定位仓库根目录；
2. 校验主 `.venv/Scripts/python.exe`；
3. 校验分割与渲染 worker 的 Python、EdgeTAM 配置和 checkpoint；
4. 创建 `.runtime/projects/default` 等必要的本地目录；
5. 将绝对路径写入 `.runtime/desktop-runtime.json`；
6. 使用现有 `WorkflowRuntimeConfig` 读取器回读验证；
7. 只在内容变化时重写文件。

准备脚本不下载依赖、不写用户配置目录，也不把令牌、端口或 Origin 写入 JSON。路径全部留在工程目录内。缺少依赖时，它以非零状态退出并指出缺失路径。

`npm run tauri:dev` 在启动 Tauri 前运行该准备步骤。Rust 仍会再次检查配置和 Python 路径，避免脚本完成后文件被移动造成难以理解的启动失败。

Vite 开发服务器固定监听 `127.0.0.1:1420` 并启用 `strictPort`，使 WebView Origin 可以被安全地预先列入 API allowlist。现有 `GS_VIDEO_DEV_API_ORIGIN` proxy 改为可选：显式提供时继续服务浏览器代理调试；未提供时启动无 proxy 的 SPA，桌面和浏览器连接页均使用各自的绝对 session origin。Tauri 启动 Python 时显式传入 `--browser-origin http://127.0.0.1:1420`。端口被占用时 Vite 必须失败，不得静默换端口导致 CORS 配置漂移。

## 6. Python 启动握手

### 6.1 CLI 契约

Python CLI 增加显式的机器启动模式，例如 `--startup-handshake`。该模式必须与 `--serve`、`--runtime-config <absolute-path>` 和 `--session-token-stdin` 一起使用，不能用于 `--doctor`。

令牌读取沿用现有有界单行 stdin 契约。Rust 写入令牌和换行后立即关闭子进程 stdin；令牌不得出现在 argv、环境变量、stdout、stderr、runtime JSON 或应用日志中。

### 6.2 无竞争随机端口

Python 在启动 Uvicorn 前自行创建 TCP socket，绑定 `127.0.0.1:0` 并保留该 socket。绑定成功后从 socket 读取实际端口，再把同一个已占用 socket 交给 Uvicorn。不得采用“先查询空闲端口、关闭探测 socket、再让 Uvicorn 绑定”的方式，以免产生端口竞争窗口。

### 6.3 stdout 协议

握手模式下，stdout 的第一条非空记录必须是单行 UTF-8 JSON：

```json
{"port": 49152, "apiVersion": "v1", "pid": 12345, "parentPid": 12344}
```

约束：

- `port` 是 1 至 65535 的整数；
- `apiVersion` 必须等于宿主支持的版本；
- `pid` 必须是正整数；它必须与 Rust 直接启动的进程一致，或由匹配的 `parentPid` 表明它是该进程通过 Windows Python 虚拟环境重定向器启动的一层子进程；
- 行长度设置小上限，拒绝无限或超大输出；
- token 和完整 runtime 配置不得出现在该记录中；
- 握手后应用诊断写 stderr 或项目日志，不再复用 stdout 传输控制消息。

JSON 只表示“端口已被本进程保留”，不表示 API 已经可服务。Rust 仍须完成认证健康检查。

## 7. Rust 启动状态机

生命周期管理器采用显式状态，避免窗口事件、异步健康检查和退出清理互相竞争：

```text
Idle → Spawning → AwaitingHandshake → CheckingHealth → Ready
                       └──────────────┴──────────────→ Failed
Ready → Stopping → Exited
任意非终态 → Stopping（应用退出或超时）
```

关键行为：

- 使用加密安全随机源生成至少 256 bit 会话令牌；
- 子进程 stdin/stdout/stderr 使用管道，Windows 下不打开额外控制台窗口；
- 从 spawn 开始计算 15 秒总启动期限，而非每阶段分别等待 15 秒；
- 严格解析握手 schema，并验证直接子进程或单层虚拟环境重定向关系；
- 健康检查与关闭请求使用专用 loopback HTTP client，并显式禁用系统和环境代理；
- 在期限内使用短退避重复调用认证 `/healthz`；
- 任何解析错误、提前退出、版本不匹配或健康检查失败都会进入统一清理路径；
- 日志只能包含阶段、错误类别、退出码和安全路径摘要，不包含 token；
- 生命周期状态放入 Tauri managed state，保证关闭事件和异常处理共享同一所有权。

开发后端命令不通过 shell 字符串执行，所有参数作为独立 argv 项传递，避免 PowerShell/cmd 转义和注入问题。

## 8. WebView 会话启动

主窗口不在 `tauri.conf.json` 中自动创建。Rust 只有在健康检查成功后才创建主 WebView，并在前端代码执行前注入本次会话：

```ts
window.__GS_VIDEO_SESSION__ = {
  origin: "http://127.0.0.1:<port>",
  token: "<memory-only-token>"
}
```

现有 `main.tsx` 读取后立即删除该全局字段，并选择 `TauriCompositionRoot`。会话不写入 localStorage、sessionStorage、IndexedDB 或文件。普通浏览器没有该字段，因此继续进入 `BrowserCompositionRoot`。

注入脚本需要使用正确的 JavaScript/JSON 转义，不允许通过字符串拼接插入原始 token。主窗口在完成注入后才显示。开发 WebView 使用固定的 `http://127.0.0.1:1420`；生产资源路径留给后续打包任务处理。

如果 Tauri 在创建 WebView 前失败，宿主显示简短的原生启动错误或明确写入 stderr，并提供安全的诊断位置。失败时不得退回浏览器手工连接页，因为这会掩盖桌面握手缺陷。

## 9. 关闭和异常退出

桌面退出只有一个幂等清理入口：

1. 标记 `Stopping`，拒绝重复执行；
2. 使用会话令牌请求后端已有的取消/关闭能力；
3. 等待 Python API 关闭其任务、GPU worker 和项目锁；
4. 在有限期限内等待子进程退出；
5. 超时后终止 Python 及其仍存活的子进程树；
6. 回收 stdout/stderr 读取任务和句柄；
7. 清空内存 token。

窗口关闭与 Tauri 全局退出事件都调用该入口。Python 在握手前退出时，Rust 收集有限长度的 stderr 尾部作为诊断；已进入 Ready 后意外退出时，主窗口显示“本地服务已退出”，不使用旧 token/port 自动重启任务。

API 正常关闭的具体 HTTP 入口若当前不存在，则增加仅限已认证本地会话的 shutdown 端点；它只触发服务器退出，不接受 PID、命令或路径。即使 HTTP 关闭失败，Rust 仍必须执行超时回收。

## 10. Tauri 权限与网络边界

开发宿主使用 Rust 自有的子进程生命周期代码，前端不获得 shell 权限。capability 只保留主窗口所需的 core、dialog 和 opener 权限；不授予前端任意 execute、spawn 或文件系统范围。

API 继续只监听 `127.0.0.1`，并验证 Bearer token、WebSocket 首消息令牌和严格 Origin。发布 WebView Origin 使用现有 Tauri allowlist；开发模式额外且仅允许固定的 `http://127.0.0.1:1420`。开发 CSP 只增加该 Vite Origin 和随机 loopback API/WS 所需的连接范围；不允许远程脚本。

## 11. 代码边界

预计代码按以下职责拆分：

- Python启动 socket/握手模块：预绑定端口、输出协议、把 socket 交给 Uvicorn；
- Python CLI：参数组合与私密 stdin 校验；
- 开发 runtime 准备脚本：工程内路径发现和配置生成；
- Rust backend command：解析开发 Python 命令，未来可替换为 packaged sidecar；
- Rust handshake parser：纯函数与严格 schema；
- Rust lifecycle manager：spawn、超时、健康检查、状态和清理；
- Tauri app builder：插件、managed state、WebView 创建及事件接线；
- 共享前端：只保留既有内存 bootstrap 边界，不新增业务层 Tauri import。

握手解析和生命周期管理不直接依赖 UI，以便使用 fake sidecar 做 Rust 测试。

## 12. 测试策略

### 12.1 Python

- CLI 拒绝不完整或冲突的握手参数；
- 预绑定 socket 仅使用 loopback 且返回实际随机端口；
- 握手 JSON schema、API 版本和 PID 正确；
- stdout 不包含 token；
- Uvicorn 接收的是保留中的 socket；
- shutdown 只接受已认证请求并执行幂等清理；
- runtime 准备脚本生成全部工程内绝对路径，缺失依赖时失败。
- Vite 未设置 proxy target 时仍可启动，显式设置时仍拒绝非 loopback target；
- Vite 端口被占用时不自动递增；

### 12.2 Rust

- 合法、畸形、超长、版本不匹配、PID 不匹配，以及虚拟环境重定向 PID 的握手；
- fake sidecar 正常启动并通过 fake health endpoint；
- 即使进程环境设置了代理且没有 `NO_PROXY`，loopback 健康检查也必须直连；
- 无握手超时、握手后提前退出、健康检查失败；
- token 只写 stdin，捕获日志中不可见；
- 正常关闭和强制回收均不遗留 fake child；
- 多个关闭事件只触发一次清理；
- WebView bootstrap 序列化能安全处理所有 token 字符。

### 12.3 前端和集成

- 既有 `window.__GS_VIDEO_SESSION__` 读取后删除测试继续通过；
- 浏览器模式仍显示手工连接页；
- API 接受固定 Vite 开发 Origin，拒绝其他未列入名单的本地页面；
- 前端边界测试确认 feature/API 模块未直接导入 Tauri；
- `npm run build:web` 和 TypeScript 检查通过；
- `cargo test` 通过；
- 最终在当前 Windows 开发机执行 `npm run tauri:dev`，确认窗口自动进入已连接状态，关闭后 Python API 和 GPU worker 均退出。

真实 GPU 工作流不作为桌面宿主单元测试的前提。桌面 smoke 可以使用已配置的本地 runtime，但生命周期测试必须使用轻量 fake sidecar，确保离线、确定且快速。

## 13. 失败信息

启动错误至少区分：

- 主 Python 虚拟环境缺失；
- runtime 配置准备失败或模型/worker 路径缺失；
- Python 无法启动；
- 握手超时、损坏或版本不兼容；
- API 健康检查失败；
- 主窗口创建失败；
- 关闭超时并被强制终止。

面向用户的信息给出下一步操作，例如运行环境准备脚本或查看项目日志；内部错误保留有限上下文。任何错误都不得输出 token。

## 14. 后续发布迁移

后续打包任务将新增 PyInstaller spec、目标三元组命名的 API 可执行文件和 Tauri `externalBin`。迁移时只替换开发 backend command provider：

```text
DevelopmentPythonCommand → PackagedSidecarCommand
```

以下契约保持不变：

- stdin 单行令牌；
- stdout 单行握手 JSON；
- 随机 loopback 端口；
- 认证健康检查；
- WebView 内存 session；
- 启动状态机和关闭顺序；
- 前端 `BackendClient`/`PlatformBridge` 边界。

发布任务再加入 Tauri shell/externalBin 最小权限、PyInstaller manifest、安装包和干净机器验证，不在本轮留下未经验证的半成品配置。

## 15. 完成定义

本设计完成的判定条件是：

1. 全新终端在已准备依赖的仓库中执行 `npm run tauri:dev`，无需手工输入端口或 token；
2. 窗口只在认证健康检查通过后进入桌面应用；
3. Python API 使用随机 loopback 端口，token 不出现在进程参数、环境、文件或日志；
4. 浏览器模式与现有 Web/API 契约无回归；
5. 启动失败和关闭超时可诊断且无遗留子进程；
6. Python、Rust、前端定向测试以及一次真实 Windows 桌面 smoke 全部通过；
7. 实现边界允许后续替换为打包 sidecar，而无需重写前端或握手协议。
