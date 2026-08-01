# GS Video

GS Video 是一个本地 Gaussian Splatting 视频背景替换 MVP。共享 React/Vite 前端既可以运行在普通浏览器中，也可以由 Tauri 2 桌面宿主加载。

## 开发环境启动

应用运行资源放在工程目录的 `.venv` 与 `.runtime` 中。主 `.venv`、Node.js 和 Tauri 工具仍是启动桌面开发所需的宿主环境；分割/渲染 worker、EdgeTAM 权重和 FFmpeg 缺失时，启动后可在“导入”页点击“修复环境”，由应用下载并配置到 `.runtime`，不会修改系统 Python、CUDA 或 PATH。所有命令都从仓库根目录运行。

### 桌面端

```powershell
npm run tauri:dev
```

该命令会自动：

1. 校验工程内 Python、EdgeTAM、renderer 和模型权重；缺少 worker 资源时生成可进入“导入”页修复的运行配置；
2. 生成被 Git 忽略的 `.runtime/desktop-runtime.json`；
3. 启动固定在 `127.0.0.1:1420` 的 Vite 开发服务器；
4. 启动 Tauri，并由 Rust 宿主在随机 loopback 端口启动 Python API；
5. 通过 stdin 传递一次性令牌，完成认证健康检查后显示窗口。

关闭桌面窗口时，宿主会先请求 API 正常关闭任务和 GPU worker，超时后再回收进程树。

### Web 端

打开两个 PowerShell 终端，在仓库根目录分别运行：

```powershell
npm run api:dev
```

```powershell
npm run dev:web
```

第一个终端会显示一次性 session token，并输出包含随机 API `port` 的启动 JSON。浏览器打开 [http://127.0.0.1:1420](http://127.0.0.1:1420)，在连接页输入该端口和 token。

浏览器 token 只显示在当前开发终端，不写入项目配置或浏览器存储。停止 API 时在第一个终端按 `Ctrl+C`。

### 运行环境修复

进入桌面端或浏览器连接页后，如果环境探针发现 CUDA、PyTorch、gsplat、EdgeTAM 或 FFmpeg 缺失，导入页会显示“修复环境”。修复任务在独立进程中运行，支持进度、取消、失败重试和 `.partial` 断点续传；下载资源使用仓库内固定清单、HTTPS 和 SHA-256 校验，并只写入 `.runtime`。修复完成后页面会自动刷新环境状态，必要时提示重启应用。

## 常用验证

```powershell
npm run typecheck:web
npm run test:web -- --run
npm run build:web
```

Python 测试使用工程虚拟环境：

```powershell
.venv\Scripts\python.exe -m pytest -m "not gpu"
```

Rust 测试从 `apps/desktop/src-tauri/Cargo.toml` 运行。若使用工程内 Rust 工具链，需要先设置 `RUSTUP_HOME`、`CARGO_HOME` 和 `CARGO_TARGET_DIR` 指向 `.runtime` 下对应目录。

`npm run tauri:dev` 会自动发现 `.runtime/cargo/bin/cargo.exe` 并设置这些变量；只有直接运行 Cargo 命令时才需要手工设置。
