# GS Video

GS Video 是一个面向 Windows + NVIDIA GPU 的本地 Gaussian Splatting 视频背景替换 MVP。它从源视频中提取屏幕空间人物前景，使用 ViPE 解算完整视频的相机运动，将轨迹映射到静态 Gaussian Splatting 场景，再完成渲染、Alpha 合成、可组合后处理和 MP4 导出。

共享的 React/Vite 前端既可以运行在普通浏览器中，也可以由 Tauri 2 桌面开发宿主加载。项目、素材和中间产物均保存在本机；本仓库当前提供开发版运行方式，不包含签名安装器、自动更新或发布版打包流程。

## 当前能力

- 管理多个本地项目，以及可跨项目复用的视频、PLY 和 3D LUT 素材。
- 使用 EdgeTAM 完成人物分割，使用 ViPE 从完整 RGB 视频解算逐帧相机内参与位姿。
- 在 GS 场景中探索机位、拟合目标地面，并调整轨迹位置、比例、方位和固定输出裁剪。
- 实时预览 Gaussian Splatting 场景，统一生成背景渲染、前景修边和 Alpha 合成结果。
- 在合成后应用可为空、可排序、可重复的效果链：基础校色、3D `.cube` LUT、辉光、暗角和锐化；每个效果都有独立启用状态与 Mix。
- 导出 H.264 或 H.265 MP4，支持恒定质量和双遍 VBR，并验证帧数、时长、音轨及 BT.709 元数据。
- 配置项目库、缓存目录和显存预算，并安全清理未引用缓存或使缓存失效后执行深度清理。

## 工作流

界面按六步引导完成一次合成：

1. **导入**：选择或导入源视频（MP4、MOV、MKV）与 Gaussian Splatting PLY。
2. **人物**：在代表帧上提供一次人物提示，生成全片前景 Mask。
3. **场景对齐**：解算 ViPE 相机轨迹，探索 GS，提交地面提示并调整轨迹映射。
4. **预览**：确定固定输出裁剪，预览运动、抠像修边和 Alpha 合成。
5. **后期**：编辑效果链，比较代表帧，并按需生成整片后处理预览。
6. **导出**：选择编码设置，生成并保存经过验证的 MP4。

内部权威阶段依赖为：

```text
ingest → segment → solve_camera → map_trajectory → render → composite → post_process → export
```

后处理即使使用空效果链，也会发布统一的 `post_process` 权威帧，导出始终只读取这一份输入。修改后期效果只会使 `post_process → export` 失效；修改编码只会使 `export` 失效。

## 色彩与处理边界

当前版本只接受 **SDR Rec.709** 内容，不会对 HDR、HLG、PQ 或宽色域输入执行静默色调映射。后处理作用于前景与 GS 背景完成 Alpha 合成后的整帧 RGB；人物 Mask 的收缩/扩张、羽化和边缘颜色净化属于合成前的抠像修边，不属于后处理效果链。

生产后处理、人物分割和 GS 渲染使用隔离的 CUDA worker，不会在 CUDA 不可用时静默回退 CPU。代表帧、整片预览与最终输出共用同一套后处理数学引擎。

## 开发环境

### 前置要求

- Windows 11 x64。
- 支持 CUDA 的 NVIDIA GPU，以及可用的 NVIDIA 驱动。
- Python `>=3.11,<3.12`。
- Node.js `>=24.14.0,<25` 与 npm `>=11.12.1,<12`。
- 桌面模式需要 Rust `>=1.77.2` 及 Tauri 在 Windows 上所需的 C++/WebView2 构建环境。
- 首次修复 worker、模型和 FFmpeg 环境时需要联网，并需要为 `.runtime` 预留足够空间。

所有命令都从仓库根目录运行。首次检出后先准备主开发环境：

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
npm ci
```

主 `.venv`、Node.js 和 Rust/Tauri 工具链属于启动前置环境。分割、相机、渲染和后处理 worker、EdgeTAM 权重及 FFmpeg 由应用安装到 `.runtime`，不会修改系统 Python、CUDA Toolkit 或 `PATH`。

### 桌面端

```powershell
npm run tauri:dev
```

该命令会：

1. 检查主 Python、运行配置和本地 worker 资源；资源暂缺时仍允许应用进入“导入”页执行修复。
2. 生成被 Git 忽略的 `.runtime/desktop-runtime.json`。
3. 在 `127.0.0.1:1420` 启动 Vite 开发服务器。
4. 启动 Tauri，并由 Rust 宿主在随机 loopback 端口启动 Python API。
5. 通过 stdin 传递一次性令牌，完成认证健康检查后显示窗口。

关闭桌面窗口时，宿主会先请求 API 正常停止任务和 GPU worker；超时后再回收由本次会话创建的进程树。

### 浏览器端

打开两个 PowerShell 终端，在仓库根目录分别运行：

```powershell
npm run api:dev
```

```powershell
npm run dev:web
```

第一个终端会显示一次性 session token，并输出包含随机 API `port` 的启动 JSON。浏览器打开 [http://127.0.0.1:1420](http://127.0.0.1:1420)，在连接页输入该端口和 token。

token 只显示在当前开发终端，不写入项目配置或浏览器存储。停止 API 时在第一个终端按 `Ctrl+C`。

### 修复运行环境

应用发现 CUDA、PyTorch、ViPE、gsplat、EdgeTAM、模型权重或 FFmpeg 缺失时，“导入”页会显示“修复环境”。修复任务在独立进程中运行，支持进度、取消、失败重试和 `.partial` 断点续传；下载内容来自仓库内的固定清单，经过 HTTPS、大小和 SHA-256 校验后才会安装到 `.runtime`。

修复不会安装或升级 NVIDIA 驱动、系统 Python、Node.js、Rust 或系统 CUDA Toolkit。修复完成后页面会重新运行环境探针，必要时提示重启应用。

## 本地数据与缓存

默认运行数据位于 `.runtime/data`：

- **项目库**保存项目配置和应用接管的原始素材，应当长期保留。
- **缓存目录**保存可重建的帧、Mask、相机结果、渲染、合成、后处理和导出产物。
- `.runtime/user-settings.json` 保存存储布局和显存预算等本机设置。

可以在“设置”中迁移项目库与缓存目录。项目库和缓存必须位于本地固定磁盘、互不重叠；路径变更应用后需要重启。删除项目会删除该项目配置与生成结果，但不会删除仍由素材库管理的共享源文件。

## 验证

前端检查：

```powershell
npm run typecheck:web
npm run test:web -- --run
npm run build:web
```

Python 非 GPU 测试：

```powershell
.venv\Scripts\python.exe -m pytest -m "not gpu"
```

静态检查：

```powershell
.venv\Scripts\python.exe -m ruff check src tests tools
.venv\Scripts\python.exe -m mypy
```

Rust 测试从 `apps/desktop/src-tauri/Cargo.toml` 运行。`npm run tauri:dev` 会自动发现 `.runtime/cargo/bin/cargo.exe` 并设置工程内 Rust 工具链所需变量；直接运行 Cargo 时，需要自行设置 `RUSTUP_HOME`、`CARGO_HOME` 和 `CARGO_TARGET_DIR`。

非 GPU 测试只能验证协议、状态机、缓存和 API 等逻辑，不能替代 NVIDIA/CUDA/Torch 的真实像素路径验收。

## 代码结构

```text
apps/web/                 React/Vite 前端与六步工作台
apps/desktop/src-tauri/   Tauri 2 桌面开发宿主与 API 生命周期管理
src/gs_video/api/         仅限 loopback、令牌认证的 FastAPI 接口
src/gs_video/pipeline/    阶段依赖、缓存失效与工作流服务
src/gs_video/segmentation EdgeTAM 分割 worker
src/gs_video/camera/      ViPE 相机解算与轨迹映射
src/gs_video/scene/       PLY 校验、gsplat 渲染与实时预览 worker
src/gs_video/composite/   抠像修边与 Alpha 合成
src/gs_video/postprocess/ CUDA 后处理引擎与 worker
src/gs_video/media/       FFmpeg 探测、转码和导出
src/gs_video/project/     多项目目录、素材库与迁移
src/gs_video/storage/     项目库、缓存布局和清理
tests/                    单元、集成、安全与 GPU 测试
tools/                    启动、环境准备、修复和基准工具
```

详细架构、数据模型、算法流程、API、存储与排障说明见 [技术文档](docs/technical-documentation.md)。