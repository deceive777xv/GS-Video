# Gaussian 场景实时预览设计

日期：2026-08-04

## 背景与基线

机位页当前在每次相机变化后等待 150 ms，再请求一张 960×540 的
`RGB+ED` 预览。生产配置通过一次性隔离 renderer worker 完成每一帧：启动新
Python 进程、校验并读取 PLY、构建场景运行时数据、上传 CUDA、渲染、写入 NPZ，
随后主进程读取 NPZ、编码 PNG、持久化项目并由前端再次下载图片。

使用当前默认项目的真实场景测得：

- PLY 为 630,225,580 字节，包含 2,541,226 个 Gaussian，估算显存 1,080 MiB；
- 当前单帧端到端耗时约 5.6～9.5 秒；
- 两次完整文件 SHA-256 校验合计约 0.72 秒，一次性 worker 约 4.84 秒；
- 同一进程内 PLY 首次载入约 1.81 秒、首帧约 2.01 秒，后续帧约 159 ms；
- 场景张量常驻 CUDA 后，960×540 的暖态光栅化约 4.5～4.9 ms；
- 暖态 RGB 光栅化、回读转换与 JPEG 编码合计约 13 ms，编码结果约 138 KiB。

因此主要瓶颈是逐帧进程冷启动、文件校验、场景重载、CPU 预处理、CUDA 上传和
落盘协议，而不是 GPU 光栅化或显存容量。

## 目标

- 当前 254 万 Gaussian 场景在相机连续交互时稳定达到 20～30 FPS。
- 停止交互后生成与当前相机严格匹配的 960×540 RGB+深度权威帧，暖态目标为
  300 ms 内完成。
- 保留 renderer 进程隔离、场景身份校验、generation 防陈旧规则、落脚点深度绑定
  和严格项目持久化。
- 实时请求不形成队列，不让旧帧覆盖新帧，也不让临时交互状态污染项目权威状态。
- 预览常驻显存不得妨碍分割、正式渲染或应用关闭。

## 非目标

- 不在浏览器中引入第二套 Gaussian 渲染器，也不向浏览器传输完整 PLY。
- 不改变正式背景序列渲染、合成或导出的质量参数。
- 不让实时帧参与相机确认、落脚点选择、工作流失效或项目恢复。
- 不追求显示器刷新率级别的 60/120 FPS；本次以稳定 20～30 FPS 和低交互延迟为准。
- 不放宽 PLY 路径、文件身份、输出路径、进程树或本地认证安全检查。

## 方案选择

采用常驻隔离 renderer worker，并在同一预览会话中提供实时 RGB 和权威
RGB+深度两种渲染模式。

未采用直接在 FastAPI 进程中缓存 `GsplatRenderer` 的方案。它实现较少，但会让
CUDA、PyTorch 或 gsplat 的进程级故障直接终止本地 API，削弱现有隔离设计。

未采用浏览器端 Gaussian viewer。当前 PLY 约 630 MB，浏览器加载、解析和显存复制
成本高，并会形成与正式 gsplat 输出可能不一致的第二套相机和材质实现。

## 模块与接口

新增深模块 `PreviewSession`，外部 seam 位于 API 工作流与 renderer worker 之间。
调用方只需要知道以下能力：

1. `render_live(...) -> LivePreviewFrame`：返回有界 JPEG 字节及请求序号，不持久化；
2. `render_authoritative(...) -> PickBuffer`：返回 RGB 与 expected depth，由现有 API
   逻辑发布 PNG、更新相机 revision 和项目权威状态；
3. `close(reason) -> None`：幂等释放进程、CPU 场景和 CUDA 张量。

场景验证、惰性启动、进程协议、PLY 缓存、CUDA 常驻、请求串行化、临时文件、崩溃
恢复与空闲回收均隐藏在模块 implementation 内。调用方每次都传入场景 authority，
模块以相对路径、SHA-256、大小和稳定文件身份组成 cache key；cache key 变化时必须先
销毁旧会话，再验证并加载新场景。

`PreviewSession` 是唯一新增的外部 seam。worker 协议、编码器和时钟可以作为模块
内部 seam 由测试替换，不向 API route 或 React 暴露。

## 常驻 worker implementation

### 会话建立

首次实时或权威请求惰性启动隔离进程。父进程继续使用现有受控 Python executable、
隐藏窗口、进程树 guard、GPU admission gate 和有界 stdout/stderr。worker 完成握手后：

1. 对 PLY 执行现有普通文件、路径所有权、大小、稳定身份和 SHA-256 校验；
2. 只加载一次 `GaussianScene`；
3. 只计算一次 quaternion 归一化、scale 指数和 opacity sigmoid；
4. 把 means、quats、scales、opacities 和 SH colors 转为 CUDA tensor 并保持常驻；
5. 执行一次暖机渲染，避免第一次用户拖动承担 CUDA kernel 初始化成本。

后续帧只创建很小的 view matrix、intrinsics 和背景 tensor。Gaussian tensor 不再经由
`torch.as_tensor(..., device="cuda")` 逐帧上传。

### 进程协议

常驻 worker 使用有界、严格校验的命令/结果协议。每条命令包含单调 request id、模式、
相机、尺寸和 worker 所有的临时输出路径。实时模式输出 JPEG；权威模式输出包含 RGB
与 expected depth 的 NPZ。输出仍位于项目 `previews` 目录下的随机、未预先存在的受控
路径，父进程验证普通文件身份、大小、格式、尺寸和完整 inventory 后读取并删除。

协议不把大块图像 base64 塞入 JSONL，也不放宽现有 64 KiB 事件上限。进程只串行执行
一条命令；关闭、场景变化或父进程取消时由 process-tree guard 回收整个树。

### 两种渲染模式

- 实时模式使用 `render_mode="RGB"`、`packed=True`、960×540，并编码为 quality 85
  的 JPEG。它不生成 depth、不写项目 JSON、不发布长期 artifact。
- 权威模式使用 `render_mode="RGB+ED"`、`packed=True`、960×540，继续返回无损 RGB
  与 float32 expected depth，随后沿用现有 `PreviewArtifactStore` 和项目更新事务。

两种模式使用同一份常驻场景和相机数学，避免实时画面与权威帧采用不同渲染实现。

## API 与数据流

新增受本地认证保护的实时预览 route。请求携带 request id、相机和 960×540 尺寸，
响应直接返回 `image/jpeg`，并在响应头回显 request id；响应使用 `Cache-Control:
no-store`。实时 route 不读取或修改 `workflow.preview`，也不分配 camera revision。

现有 `POST /api/v1/projects/current/preview` 保持权威语义和响应模型，但底层从一次性
`WorkerPreviewService` 切换为 `PreviewSession.render_authoritative`。generation 冲突、
scene epoch、repository compare-and-set、artifact 发布、相机 revision 和落脚点失效规则
保持不变。

交互数据流如下：

1. 进入机位页后保留已有权威图片，并在后台用当前相机准备会话；没有已有图片时显示
   现有载入状态。
2. 相机变化后立即记录最新 camera fingerprint。前端任一时刻最多发出一个实时请求。
3. 实时响应到达时，只要 request id 新于当前已显示实时帧就替换画面；即使用户仍在
   拖动，也允许显示最近完成的一帧，避免连续交互期间永远丢弃所有画面。
4. 若响应期间相机又变化，前端立即用最新相机发下一帧；所有中间相机被合并，不排队。
5. 每次相机变化重置 120 ms settle timer。timer 到期后发送新的权威 generation。
6. 权威请求进入会话时优先于尚未开始的实时请求。当前正在执行的实时帧最多造成一帧
   延迟，不强行中断 CUDA kernel。
7. 权威帧完成后替换实时 JPEG，更新 `frameCameraFingerprint`，恢复确认和落脚点能力。

实时帧永远不使“确认机位”按钮可用。按钮只在现有权威帧 fingerprint 与当前相机完全
一致且权威请求结束时启用。

## 生命周期与 GPU 协调

- 场景 authority 改变时立即关闭旧会话；新请求只能建立新会话。
- 离开机位页时前端 best-effort 请求关闭；服务端仍以 30 秒空闲超时作为最终保证。
- 启动分割、正式背景渲染或其他独占 GPU 工作前，工作流先幂等关闭预览会话，再进入
  现有 `GpuAdmissionGate`。关闭操作不得在持有 gate 时等待，避免锁顺序死锁。
- 相机确认本身不强制关闭会话；进入下一工作流步骤或其他 GPU 任务时关闭。
- API shutdown 通过现有 `WorkerRegistry` 关闭会话，等待上限后由 process-tree guard
  强制回收。
- 空闲回收和显式关闭都清除 CPU cache key、CUDA tensor 和临时输出；下一次请求重新
  做完整 authority 校验。

## 并发、陈旧响应与背压

前端串行泵和后端单会话 actor 共同保证每个客户端最多一个实时渲染在途。后端只保留
最新尚未开始的实时命令；新的实时命令替换旧的 pending 命令。权威命令不可被实时命令
替换，并拥有更高调度优先级。

request id 只决定实时显示顺序，不具备项目 authority。generation 只用于现有权威
route，两者不能互换。React 卸载或 camera authority 改变后仍用 AbortController 和本地
authority token 忽略迟到响应；服务端即使已完成被取消的实时帧，也不会产生持久副作用。

## 错误处理与降级

- 会话第一次崩溃时清除所有缓存并自动重建一次；同一用户请求再次失败则返回现有
  `preview_worker_failed` 类别，不无限重启。
- 实时请求失败时保留最后一张画面，停止实时泵，并立即走一次现有权威预览。页面显示
  非阻塞提示，但保存和后续工作流仍以权威结果为准。
- 权威请求失败沿用现有错误横幅，确认和落脚点继续保持禁用。
- 输出超过基于分辨率计算的上限、JPEG/NPZ inventory 异常、尺寸或 dtype 不符、文件
  身份变化都 fail closed，并销毁当前会话。
- PLY 在会话建立期间或之后发生身份变化时关闭会话并返回 `scene_changed`；不得继续使用
  已缓存但失去 authority 的 GPU 数据。
- GPU 显存不足时关闭会话并报告现有资源错误，不自动降低正式权威帧质量。

## 测试与测量

### 自动化回归

- worker client 连续实时请求只启动一个进程，只加载一次场景，并按 request id 返回；
- scene cache key 改变、空闲超时、显式关闭、API shutdown 和其他 GPU 任务开始都会回收
  会话及临时文件；
- 实时 pending 请求按 latest-only 合并，权威请求不会被替换且优先于 pending 实时请求；
- worker 崩溃只自动重建一次，协议、路径、大小、格式或 inventory 异常均 fail closed；
- 实时 route 返回 JPEG 与 request id，但不修改 project JSON、camera revision、preview
  artifact、generation 或落脚点；
- 现有权威 route 的 generation 合并/冲突、旧 generation、scene epoch 和 artifact 绑定
  测试全部继续通过；
- React 连续 pointer、wheel 和 FOV 事件最多维持一个实时请求，并在响应后只补发最新
  camera；较旧 request id 不会覆盖较新画面；
- 120 ms settle 后只提交当前 camera 的权威请求；实时帧不会启用确认或落脚点；
- 卸载、切换项目和失败降级不会更新已失效页面，也不会泄漏 object URL。

### 性能工具与实机验收

实现阶段增加可重复运行的本地 preview benchmark，分别报告会话建立、暖态 GPU、回读、
JPEG、权威深度和完整调用耗时。性能工具只读取已导入场景，并把临时结果写入受控临时
目录后清理。

使用当前 2,541,226 Gaussian 场景验收：

- 暖态实时帧端到端 p95 不超过 50 ms，持续交互达到 20～30 FPS；
- 任一时刻实时渲染在途数不超过 1，连续拖动 10 秒不产生递增请求队列；
- 暖态权威帧在相机停止后 300 ms 内完成，且落脚点使用同一帧的 expected depth；
- 初次会话建立期间旧权威图片保持可见，页面不闪白；
- 离开机位页或启动后续 GPU 阶段后，常驻 CUDA allocation 被释放；
- worker 异常退出后 API 与桌面宿主保持存活，并能按规则重建或降级。

若实际 HTTP/浏览器链路无法达到 p95 50 ms，应使用 benchmark 的分段结果定位瓶颈，
而不是降低 authority 校验或允许并发积压。

## 完成标准

真实大场景的机位交互达到稳定 20～30 FPS；权威帧、确认与落脚点仍严格绑定；所有
实时请求 latest-only 且无队列增长；renderer 故障不带走 API；预览会话在场景变化、
离页、其他 GPU 工作、空闲和退出时可靠释放；前端全量测试、Python 非 GPU 测试、GPU
定向测试、类型检查、生产构建和真实桌面交互验收均通过。
