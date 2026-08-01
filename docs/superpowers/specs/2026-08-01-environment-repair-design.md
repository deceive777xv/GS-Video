# GS Video 运行环境一键修复设计

- 文档状态：已确认，已实施
- 日期：2026-08-01
- 目标范围：在应用内下载、配置并验证工程目录内的运行时资源
- 关联文档：`2026-07-29-tauri-development-host-design.md`、`2026-07-11-gs-video-mvp-design.md`、`2026-07-11-gs-video-mvp-implementation.md`

## 1. 背景与目标

当前应用可以检测运行环境，但检测到 CUDA、PyTorch、gsplat、分割 worker 或模型缺失时，只能提示用户手工处理。桌面开发启动还要求仓库内的 Python、worker、EdgeTAM 配置和 checkpoint 已经存在，缺失资源会让用户在进入工作流后才看到“环境需要修复”。

本设计增加一个面向 Windows 开发版和本地浏览器版的环境修复流程：用户从环境卡片发起修复，后端在工程目录的 `.runtime` 下下载、解压、创建虚拟环境、安装固定依赖、配置模型与 FFmpeg，并重新运行完整环境探针。修复完成后，原有 `EnvironmentDoctor` 仍是“环境是否可用”的唯一判断来源。

目标包括：

- 覆盖主 Python 环境、分割 worker 环境、渲染 worker 环境及其所需依赖；
- 覆盖 PyTorch CUDA wheel、gsplat、EdgeTAM 代码与 checkpoint、FFmpeg/ffprobe；
- 所有下载源使用 HTTPS，资源版本、大小、SHA-256 和许可证信息由仓库内清单固定；
- 支持断点续传、重试、取消、进度展示和 API 重启后的继续修复；
- 安装过程不修改系统 PATH、注册表、系统 Python、用户缓存目录或管理员权限范围外的文件；
- 修复失败时保留当前可用环境，不让半成品成为新的运行环境。

## 2. 不在范围内的内容

- 不自动安装或升级 NVIDIA 驱动、系统 CUDA Toolkit、Node.js、Rust、Tauri CLI 或基础 Python；
- 不使用管理员权限，不写入系统级目录、注册表或全局包环境；
- 不允许用户在页面输入任意下载 URL 或任意安装命令；
- 不把大模型、wheel、FFmpeg 或其他二进制资源提交到 Git；
- 不在修复期间运行视频处理任务；
- 不把自动修复逻辑放入 Tauri Rust 生命周期管理器；Tauri 启动前的基础工具缺失仍由 CLI 诊断并处理。

PyTorch CUDA wheel 只提供 CUDA runtime 依赖，不能替代 NVIDIA 驱动。驱动不可用时，修复流程会完成可下载的资源配置，但最终 EnvironmentDoctor 仍会报告 CUDA 不可用，并将驱动问题作为独立的可操作错误展示。

当前仓库根目录的 `.venv` 是 API 自身的启动解释器，运行中的 API 不能安全地替换自己的解释器。清单仍然覆盖主 API 环境的固定依赖：在 API 已启动时，runner 只能为它准备并校验可重启后使用的 staging 环境，并在快照中标记 `restart_required`；实际切换由下一次桌面/浏览器 API 启动完成。分割、渲染、模型和 FFmpeg 资源在任务成功后即可由当前 API 使用。若连启动 API 所需的基础解释器或主环境都不存在，则沿用启动前 CLI 修复路径，不让页面假装已经可以自举。

## 3. 方案选择

### 3.1 采用：API 管理器 + 独立修复 runner

修复任务由 Python API 创建并查询，但实际下载和安装运行在独立的短生命周期子进程中：

```text
React environment card
        │ HTTP
        ▼
EnvironmentRepairManager
        │ shell=False，argv 数组，JSONL 进度
        ▼
tools/environment_repair.py
        ├─ .runtime/downloads/<resource>.partial
        ├─ .runtime/staging/<job-id>/
        ├─ .runtime/<component>/...
        └─ EnvironmentDoctor.check()
```

API 进程在下载期间仍然可响应健康检查、状态查询和取消请求。管理器负责唯一任务约束、子进程生命周期、进度快照和工作流锁；runner 负责清单执行、文件校验、安装和最终探针。

### 3.2 不采用：API 进程内直接下载

它实现简单，但大文件下载、解压和 Python 安装会占用 API 进程的执行资源，取消、异常恢复和 API 重启后的状态处理也更脆弱。

### 3.3 不采用：启动前 Tauri 安装器

它适合发布版首次安装，但无法覆盖已经启动的浏览器开发模式，也无法在现有页面中展示后端运行时的细粒度状态。本轮先把修复能力放在 Python API 边界，未来发布安装器可以复用同一份清单和 runner。

## 4. 启动边界与 runtime 配置

`tools/prepare_desktop_runtime.py` 继续负责生成 `.runtime/desktop-runtime.json`，但需要区分“准备安全路径”和“严格启动校验”两个阶段：

1. 准备阶段创建工程目录内的 `.runtime` 子目录，并为尚不存在的资源生成绝对路径；
2. 准备阶段允许资源文件暂时缺失，但拒绝路径逃逸、符号链接、Windows reparse point 和非普通文件；
3. API 启动后，EnvironmentDoctor 报告缺失项并暴露修复能力；
4. 正常工作流启动 worker 前仍使用严格校验，缺失或探针失败时拒绝任务；
5. runner 完成安装后重新读取生成配置，执行与正式工作流相同的 worker probe 和 `EnvironmentDoctor.check()`。

基础 Python、Node.js、Rust 或 Tauri CLI 缺失时 API 可能无法启动。此时 `npm run tauri:dev` 或 CLI 只负责给出缺失工具、版本和安装位置提示；进入页面后才由本功能处理 GPU worker、模型和 FFmpeg 资源。

## 5. 运行时清单

### 5.1 清单职责

新增并跟踪 `tools/runtime-manifest.json`，作为可复现安装的唯一资源清单。清单不包含大文件，只包含资源元数据和受限的结构化安装步骤。每次变更版本或下载内容都必须更新 SHA-256、大小和清单版本，并经过测试验证。

清单至少覆盖：

- 主项目 Python 依赖；
- 分割 worker 的 Python 依赖、PyTorch CUDA runtime 和 EdgeTAM 代码；
- 渲染 worker 的 Python 依赖、匹配版本的 PyTorch CUDA runtime 和 gsplat；
- EdgeTAM 配置文件与 checkpoint；
- Windows x64 的 FFmpeg 和 ffprobe；
- 当前项目的可编辑安装或等价的本地源码配置。

清单中的 Python 依赖应优先固定为可校验的 wheel 资源，以便安装时使用离线 `--no-index` 模式；需要源码归档的资源必须固定提交版本和归档 hash。下载 URL 必须是官方发布源或官方仓库归档直链，不能使用不受控的 latest URL。

### 5.2 结构化资源定义

每项资源包含以下字段：

```json
{
  "id": "renderer-torch-cuda",
  "kind": "python-wheel",
  "version": "2.6.0+cu126",
  "url": "https://official.example.invalid/resource.whl",
  "sha256": "64-character-lowercase-hex",
  "size": 123456789,
  "target": ".runtime/downloads/renderer-torch-cuda.whl",
  "extract": null,
  "license": "BSD-3-Clause",
  "license_url": "https://official.example.invalid/license"
}
```

实际实现将为 `kind`、平台、目标目录和安装操作定义代码级 allowlist。清单不能注入任意 shell 命令；安装步骤使用受限操作类型，例如 `create_venv`、`install_wheels`、`extract_archive`、`copy_file`、`install_editable_project` 和 `activate_runtime`，由 runner 映射为固定的 argv 调用。

清单加载时校验：

- manifest version 和当前平台/架构匹配；
- URL 为 HTTPS，且主机在代码审查过的官方源 allowlist 中；
- id、版本、目标路径、操作类型和依赖引用合法；
- SHA-256 为 64 位小写十六进制，size 为正整数；
- 所有目标和解压目录位于 `.runtime` 或仓库源码根目录的明确允许范围内；
- 不允许符号链接、reparse point、绝对归档成员路径和包含 `..` 的归档成员。

许可证字段用于状态、日志和诊断展示。需要额外登录、私有 token 或网页条款确认的资源不会被 runner 静默绕过；它们应以明确的“无法自动下载”错误结束，而不是记录用户凭据。

## 6. 下载、安装与激活流程

### 6.1 Preflight

runner 启动后先执行不修改资源的预检：

- 读取并校验 runtime manifest；
- 检查当前 Windows 架构、基础 Python 和可用的进程权限；
- 检查 `.runtime` 所在磁盘的预计空间，至少覆盖下载、解压、临时虚拟环境和保留旧环境的空间；
- 检查是否有工作流任务或其他修复 runner 正在运行；
- 检查已有资源是否已通过 hash，可直接复用；
- 生成 job id、staging 目录和受限日志路径。

空间不足、清单错误或资源被占用时，在下载前返回明确的错误码，不创建半成品安装。

### 6.2 Download

复用 `tools/fetch_test_assets.py` 已验证的 HTTPS、Range、`.partial`、SHA-256 和安全路径逻辑，并将其抽取为通用下载模块。下载行为如下：

- 数据先写入 `.runtime/downloads/<id>.partial`；
- 如果服务器支持 Range，则从已有长度继续；不支持时安全地从零开始并覆盖该 partial；
- 连接中断、超时和暂时性 5xx 使用有限次数退避重试；
- 每个资源完成后校验实际大小和 SHA-256；
- 校验通过后用同一文件系统上的原子 rename 转为最终下载文件；
- 校验失败不进入安装，保留或重建 partial，并报告期望值与实际值摘要；
- 取消只中断当前请求，保留 partial，供下一次修复继续。

runner 通过 JSONL 向管理器报告 `downloaded_bytes`、`total_bytes`、当前 resource id、速率可选值和阶段提示。管理器限制单条消息长度和读取缓冲，避免异常 runner 无限输出拖垮 API。

### 6.3 Staging 与安装

解压、创建虚拟环境和项目安装都先写入 `.runtime/staging/<job-id>/` 下的临时目标：

- ZIP/TAR 成员逐项验证，拒绝路径穿越、绝对路径、符号链接和 reparse point；
- 新的 Python 环境使用固定的基础 Python 创建，不调用 shell；
- 安装使用环境自身的 Python 和固定参数，优先离线 wheel 安装，不从任意依赖解析器拉取未锁定版本；
- 项目可编辑安装使用当前仓库绝对路径，并禁止将源码复制进下载目录；
- EdgeTAM、模型、FFmpeg 和 ffprobe 只写入 manifest 指定的工程目录；
- 每个组件完成后执行轻量文件/版本检查，再进入下一组件。

### 6.4 原子激活与回滚

runner 不直接把未验证的目录当作新运行环境。目录资源先准备为 staging sibling；激活时：

1. 确认没有 worker 进程持有目标文件；
2. 将现有目标改名为当前 job 的 backup（若存在）；
3. 将 staging 目标改名为正式目标；
4. 以临时文件写入并原子替换 `desktop-runtime.json`；
5. 使用正式配置运行完整 EnvironmentDoctor 和 worker probes；
6. 全部通过后保留可回滚 backup，记录成功 manifest fingerprint；
7. 任一步失败时按相反顺序恢复旧目录和旧配置，保留诊断信息。

激活前的现有环境必须仍可用；激活失败不得清理旧环境。清理历史 backup 是独立的后续维护动作，不在首次修复任务中自动删除用户可能需要的回滚资源。

## 7. 任务状态与 API

### 7.1 状态模型

管理器对外提供以下状态：

```text
idle → running → cancelling → cancelled
                    └────────→ failed
running ─────────────────────→ succeeded
```

`failed` 携带 `retryable` 和稳定的错误码。API 进程重启时不会伪造仍在运行的状态：内存中的子进程句柄消失后，下一次 `GET` 返回 `idle`，并根据 `.partial` 和已校验下载文件给出 `resume_available`。终态摘要可以原子写入 `.runtime/repair/last-result.json`，只存资源 id、阶段、错误码和时间，不存 token 或敏感环境变量。

管理器为 runner 建立带过期时间的工程内 lease，并在运行期间持续刷新心跳。runner 发现 lease 过期、job id 不匹配或父级管理器已经失联时，必须在激活前退出；新的 API 实例只能在确认旧 lease 已过期且没有存活 runner 后接管 partial。这样 API 崩溃不会留下一个仍有机会替换 runtime 的孤儿进程。

状态快照字段：

```json
{
  "state": "running",
  "job_id": "repair-20260801-...",
  "step": "download",
  "resource_id": "renderer-torch-cuda",
  "resource_name": "渲染环境 PyTorch CUDA",
  "progress": 0.42,
  "downloaded_bytes": 123456789,
  "total_bytes": 293847561,
  "message": "正在下载…",
  "resume_available": true,
  "error": null,
  "environment": null
}
```

成功时 `environment` 填充最终 EnvironmentDoctor 报告；失败时填充最后一次安全探针结果。路径只返回相对于工程目录的安全摘要，不返回用户目录中的任意绝对路径。

### 7.2 HTTP 接口

新增受保护的 API：

- `GET /api/v1/environment/repair`：返回当前快照；无任务时返回 `idle`；
- `POST /api/v1/environment/repair`：启动新任务或继续已有 partial。新任务返回 202；已有 running 任务返回当前任务快照，避免重复启动；
- `DELETE /api/v1/environment/repair`：发送取消请求并返回 `cancelling` 快照。runner 真正退出后变为 `cancelled`；没有运行任务时保持幂等；
- `GET /api/v1/bootstrap` 的 capabilities 增加 `environment_repair`，environment 字段继续由 EnvironmentDoctor 提供。

任务创建、删除和工作流创建共享同一把进程内锁。修复状态为 `running` 或 `cancelling` 时，涉及分割、渲染、合成和导出的任务创建返回 `503 environment_repair_in_progress`；仅导入静态文件等不启动 worker 的操作可继续。

管理器在 API shutdown 时先请求 runner 取消，等待有限期限；超时才终止子进程。API 关闭不会删除 partial 或可回滚 backup。

## 8. 前端交互

导入页现有环境卡片增加以下状态：

- 未就绪且无修复任务：显示“修复环境”按钮及检测到的问题；
- 修复中：显示当前资源、阶段、进度条、字节数和可读提示，提供“取消修复”；
- 已取消或可重试失败：显示“继续修复”或“重试”；
- 成功：重新请求 bootstrap，显示“环境已就绪”，并恢复下一步按钮；
- 失败：保留原问题列表，显示稳定错误码、建议动作和重试入口。

前端通过 `BackendClient` 调用接口，不直接导入 Tauri API 或访问文件系统。修复中每秒轮询一次状态；页面卸载时停止轮询，重新进入页面后从 GET 快照恢复。状态和进度使用 `aria-live`/原生 progress 语义，不能只依赖颜色表达失败或进行中。

修复进行中，导入页的“下一步”和会启动 worker 的操作均禁用；取消按钮只表示取消请求已发送，直到收到 `cancelled` 才允许再次启动。成功后前端重新读取 bootstrap，而不是自行推断 CUDA、torch 或 gsplat 已可用。

## 9. 并发、错误和安全边界

- runner 使用 `shell=False`，所有命令参数为独立 argv；不拼接 PowerShell/cmd 字符串；
- 下载、解压、安装和日志写入均限制在 `.runtime` 允许范围；每个路径组件都检查符号链接和 reparse point；
- 只接受代码 allowlist 中的官方 HTTPS 源，拒绝 HTTP、任意用户 URL 和非预期重定向；
- 每项资源必须通过 size + SHA-256，不能因为文件名或 HTTP 状态码看起来正确就安装；
- 失败消息不包含访问 token、完整环境变量、Authorization header 或敏感路径；
- 不修改系统 Python、系统 PATH、注册表和用户级缓存；worker 运行时继续使用项目内 `.runtime` cache；
- 一次只允许一个修复任务；任务开始前必须没有工作流 worker；任务期间工作流不允许启动；
- 文件被占用、磁盘空间不足、网络不可达、hash 不匹配、许可证/权限要求无法满足时，返回不同的稳定错误码；
- 取消和 API 崩溃不会把目标目录切换到半成品；激活只在所有资源准备完毕后发生。
- lease 心跳和进程存活检查必须在下载、解压、安装和激活边界都执行；lease 过期时 runner 只保留可续传文件，不执行激活或回滚旧环境。

## 10. 代码边界

实现拆分为：

- `src/gs_video/environment/manifest.py`：清单 schema、平台匹配、URL/路径/操作校验；
- `src/gs_video/environment/download.py`：从现有测试资产下载器抽取的 HTTPS、续传、hash 和安全解压能力；
- `src/gs_video/environment/repair.py`：管理器、状态快照、锁、runner 生命周期、取消和 API 重启恢复语义；
- `tools/environment_repair.py`：独立 runner CLI、JSONL 输出、预检、安装、激活、回滚和最终 probe；
- `tools/runtime-manifest.json`：受审查的 Windows x64 资源版本、hash、大小、许可证和结构化操作；
- `src/gs_video/runtime.py` / `tools/prepare_desktop_runtime.py`：允许修复前生成缺失资源的安全路径，同时保留严格工作流校验；
- `src/gs_video/api/schemas.py` / `src/gs_video/api/routes.py`：快照 schema 和三个 API；
- `apps/web/src/api/*`：BackendClient 修复接口；
- `apps/web/src/features/import/import-page.tsx`：环境修复卡片、轮询、取消和刷新 bootstrap。

管理器不把下载进度塞进现有视频任务 EventBus；修复任务是独立生命周期，REST 快照是重连后的权威状态，避免前端错过 WebSocket 事件后显示错误进度。

## 11. 测试策略

### 11.1 单元测试

- 清单 schema、平台、版本、hash、size、官方 HTTPS host 和操作 allowlist；
- Windows 路径逃逸、符号链接/reparse point、ZIP/TAR 路径穿越和恶意归档成员；
- Range 续传、服务器不支持 Range、超时重试、hash/size 不匹配和取消；
- runner JSONL 消息解析、超长/畸形消息和稳定错误码；
- 管理器状态转换、重复 POST、取消幂等、API 重启后的 `resume_available` 和工作流锁；
- runtime 准备阶段允许缺失资源但严格拒绝不安全路径，正常加载仍拒绝缺失 worker。

### 11.2 集成测试

- 使用小型本地测试 manifest 和 fake HTTPS server 完成下载、解压、创建临时 venv、激活和回滚；
- 在 fake worker probe 失败时确认旧 runtime config 和旧目录仍可用；
- API 启动修复 runner、轮询进度、取消、重试和成功后 bootstrap 刷新；
- 修复运行中提交视频任务，确认返回 `environment_repair_in_progress`；
- API 终止/重启后确认没有“假运行”状态，partial 仍可续传；
- 前端环境卡片覆盖 idle、running、cancelled、failed、succeeded 五类状态。

### 11.3 Windows 实机验收

- 在缺少一个或多个 `.runtime` 资源的干净工程中，从页面完成首次修复；
- 人为中断大文件下载，再次点击继续，确认 Range 和 hash 校验正确；
- 让 CUDA 驱动、磁盘、网络或文件占用分别失败，确认错误可操作且没有半成品激活；
- 完成修复后用真实 `assemble_api_services(...).environment_doctor.check()` 验证 `ready: true`，并确认分割和渲染 worker 都能实际启动；
- 现有已就绪环境执行修复时不重复下载已通过 hash 的资源，也不破坏旧环境。

## 12. 验收标准

功能可交付的最低条件：


- 页面能从环境卡片发起、观察、取消和重试修复；
- 所有资源均从清单中的 HTTPS 官方源获取，并通过固定大小和 SHA-256；
- 下载中断后可以从 partial 继续，API 重启不会留下错误的运行中状态；
- 修复期间视频任务被阻止，API 和健康检查仍可用；
- 修复失败会回滚，不破坏已有可用环境；
- 修复成功后完整 EnvironmentDoctor 和 worker probes 通过，bootstrap 自动恢复为可用；
- 单元、集成和必要的 Windows 实机测试通过；
- README 和 Tauri 开发宿主文档说明“启动前置工具”和“页面运行时修复”的边界。
