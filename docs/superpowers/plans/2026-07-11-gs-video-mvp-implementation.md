# GS Video MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个在 Windows 11 与 8 GB NVIDIA 显存环境中运行的内部桌面工具，让用户通过选择人物、确认 Gaussian 场景初始机位和指定落脚点，将 10–30 秒单人单镜头视频自动合成为新的 Gaussian 背景视频。

**Architecture:** 使用 Tauri 2 承载 React + TypeScript + Vite SPA，业务能力由只监听 `127.0.0.1` 的 FastAPI 本地服务提供。浏览器和 Tauri WebView 复用同一套页面、`BackendClient` 与 REST/WebSocket 契约；Tauri 只负责 API sidecar 生命周期、原生文件对话框、打开/显示文件和窗口/安装包。高变动的分割、相机求解和 Gaussian 渲染继续通过 Python 窄接口与独立 GPU worker 隔离，GPU 阶段串行运行。MVP 相机求解先使用 OpenCV 受控视频基线，后续可通过同一契约接入 ViPE 或 VGGT。

**Tech Stack:** Tauri 2、Rust stable、React 19、TypeScript 5、Vite 8、Vitest 4、Testing Library、Playwright、Python 3.11、FastAPI、Uvicorn、Pydantic 2、NumPy 2、OpenCV 4.13、Pillow、FFmpeg/ffprobe、PyTorch、EdgeTAM、SAM 2.1 Hiera Tiny 对照后端、gsplat 1.x、plyfile、PyInstaller、pytest、ruff、mypy。

## Global Constraints

- 运行平台固定为 Windows 11；WSL 只可作为可选 GPU worker 运行方式，桌面应用必须原生运行。
- 基准硬件为 NVIDIA GPU、8 GB 显存；分割、相机求解和渲染不得同时驻留 GPU。
- 源视频限制为 10–30 秒、最高 1920×1080、单镜头、单个主要人物。
- MVP 只接受已重建完成且包含完整 Gaussian 属性的 `.ply` 静态场景。
- 正常案例只要求选择人物、确认初始机位、指定落脚点三次关键交互。
- MVP 只做 RGB Gaussian 背景与人物 Alpha 合成，不做人物深度、遮挡、阴影或重光照。
- 所有原始素材只读；所有产物写入项目目录；运行过程不发起素材上传。
- FastAPI 只监听 `127.0.0.1` 随机端口；每次启动生成高熵会话令牌。所有 REST、WebSocket 和浏览器上传均须认证；CORS 只允许当前开发或桌面 origin。
- REST 是项目与任务状态的权威来源；WebSocket 只传进度/状态变化事件，断线重连后必须能通过 REST 恢复，不得依赖未持久化事件完成业务流程。
- React 业务组件不得直接导入 `@tauri-apps/*`；平台能力只能通过 `PlatformBridge` 注入。桌面导入使用路径引用，浏览器导入使用分块上传。
- 真实测试素材通过开发脚本从已审查的官方 URL 下载到 Git 忽略的缓存目录；素材清单、许可证记录和 SHA-256 锁文件必须提交。
- `project.json` 是版本化状态源；大体积逐帧数据只保存文件引用和内容摘要。
- 每个昂贵阶段必须支持进度、取消、缓存、失败分类和从最近成功阶段重试。
- 代码先写失败测试，再写最小实现；每个任务完成后单独提交。

## 技术决策说明

- React 页面只依赖 `BackendClient`、`PlatformBridge` 和可替换状态 store；WebSocket 订阅必须清理并以 REST 快照收敛。平台专有实现放在 composition root，不进入业务组件。
- Tauri 使用最小 capability：只允许启动固定的 `gs-video-api` sidecar、打开文件/目录和显示原生文件对话框。不得授予任意 shell、任意 URL 或任意文件系统访问。
- FastAPI 使用 lifespan 管理启动/关闭；API sidecar 负责 worker 子进程树清理，Tauri 负责 sidecar 退出。PyInstaller 使用可审查的 spec 文件生成不含 GPU 运行时的 Windows onefile sidecar，再按 Tauri target-triple 规则命名。
- Pydantic 2 使用 `model_validate_json()` 读取、`model_dump_json()` 写入，并在进入当前模型验证前运行显式字典迁移。
- gsplat 使用 `rasterization(means, quats, scales, opacities, colors, viewmats, Ks, width, height)`；MVP 每次只渲染一个视角以控制显存。
- EdgeTAM 作为 MVP 默认分割后端，SAM 2.1 Hiera Tiny 作为质量对照后端。两者使用同一外部 worker 协议和相近的视频 predictor API，不在桌面进程内驻留模型。
- SAM 2 官方在 Windows 上推荐 WSL；EdgeTAM 也依赖 PyTorch/CUDA 和可选自定义 CUDA 扩展。因此 worker 命令必须可配置为原生 Python 或 WSL Python。
- ViPE 能直接输出相机内参、运动和近度量深度，但其 GPU/第三方模型组合尚未在 8 GB Windows 基准机验证，因此不作为首个必须通过的 MVP 后端。`CameraSolver` 契约必须允许后续无 UI 改动地接入 ViPE。
- OpenCV 基线只服务受控输入：固定、纯旋转和轻中度手持。平移由本质矩阵恢复，尺度由目标落脚点和“运动幅度”参数决定。

实施时优先核对以下官方资料，避免复制过时 API：

- [Tauri 2 documentation](https://v2.tauri.app/)
- [React documentation](https://react.dev/)
- [Vite documentation](https://vite.dev/)
- [Vitest documentation](https://vitest.dev/)
- [FastAPI documentation](https://fastapi.tiangolo.com/)
- [PyInstaller documentation](https://pyinstaller.org/en/stable/)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [OpenCV 4.13 documentation](https://docs.opencv.org/4.13.0/)
- [FFmpeg documentation](https://ffmpeg.org/documentation.html)
- [SAM 2 official repository](https://github.com/facebookresearch/sam2)
- [EdgeTAM official repository](https://github.com/facebookresearch/EdgeTAM)
- [gsplat official repository](https://github.com/nerfstudio-project/gsplat)
- [ViPE official repository](https://github.com/nv-tlabs/vipe)
- [DAVIS 2017 official downloads](https://davischallenge.org/davis2017/code.html)
- [Graphdeco-Inria 3DGS official pre-trained models](https://github.com/graphdeco-inria/gaussian-splatting)

## 目标文件结构

```text
GS-Video/
├─ pyproject.toml                         # 包元数据、依赖、pytest/ruff/mypy 配置
├─ package.json                           # 前端 workspace 与 Tauri CLI 脚本
├─ package-lock.json                      # 固定 Node 依赖
├─ tsconfig.base.json                     # 共享严格 TypeScript 配置
├─ README.md                              # 开发环境、模型安装、运行与验证命令
├─ apps/
│  ├─ web/
│  │  ├─ package.json
│  │  ├─ vite.config.ts
│  │  ├─ vitest.config.ts
│  │  ├─ index.html
│  │  └─ src/
│  │     ├─ app/                          # 页面路由、store 与 composition root
│  │     ├─ api/                          # BackendClient、REST/WS 客户端与 DTO
│  │     ├─ platform/                     # Browser/Tauri PlatformBridge
│  │     ├─ features/                     # import/subject/camera/preview/export
│  │     └─ test/                         # Vitest setup 与 fake adapters
│  └─ desktop/
│     └─ src-tauri/
│        ├─ Cargo.toml
│        ├─ tauri.conf.json
│        ├─ capabilities/default.json
│        ├─ binaries/                     # Git 忽略的 target-triple sidecar 产物
│        └─ src/                          # 启动握手、进程清理、平台 commands
├─ src/gs_video/
│  ├─ __init__.py
│  ├─ __main__.py                         # python -m gs_video 入口
│  ├─ app.py                              # FastAPI app factory 与 lifespan 组装
│  ├─ api/
│  │  ├─ auth.py                         # 启动令牌、origin 与 WS 认证
│  │  ├─ routes.py                       # 项目、素材、任务、预览与导出 REST
│  │  ├─ uploads.py                      # 浏览器分块上传与配额
│  │  ├─ events.py                       # 可重连的任务事件 hub
│  │  └─ schemas.py                      # 版本化 API DTO
│  ├─ domain/
│  │  ├─ models.py                        # 项目、素材、相机、场景、阶段 Pydantic 模型
│  │  ├─ errors.py                        # 可修复/素材不适用/系统错误
│  │  └─ contracts.py                     # Segmenter/CameraSolver/Renderer/Compositor 协议
│  ├─ project/
│  │  ├─ repository.py                    # 原子读取、写入、创建项目目录
│  │  ├─ migrations.py                    # project.json 逐版本迁移
│  │  └─ cache.py                         # 内容摘要、缓存键、下游失效
│  ├─ pipeline/
│  │  ├─ cancellation.py                  # 协作式取消 token
│  │  ├─ events.py                        # 进度与结果事件
│  │  ├─ runner.py                        # 串行阶段执行、重试、状态持久化
│  │  └─ workflow.py                      # MVP 阶段图和依赖
│  ├─ environment/
│  │  └─ doctor.py                        # FFmpeg、GPU、模型、磁盘环境检查
│  ├─ media/
│  │  ├─ ffmpeg.py                        # 安全的 subprocess 参数构造和 ffprobe 解析
│  │  ├─ ingest.py                        # 素材校验、代理帧和音轨提取
│  │  └─ export.py                        # 帧序列、音频和 MP4 导出
│  ├─ segmentation/
│  │  ├─ client.py                        # 外部分割 worker 客户端与 JSONL 协议
│  │  └─ worker.py                        # EdgeTAM/SAM 2.1 视频传播进程入口
│  ├─ camera/
│  │  ├─ opencv_solver.py                 # 受控视频相机求解
│  │  ├─ classify.py                      # fixed/rotation/6DoF 分类与可信度
│  │  └─ mapping.py                       # 源相对位姿到目标场景位姿
│  ├─ scene/
│  │  ├─ ply.py                           # Gaussian PLY 校验与张量加载
│  │  ├─ camera.py                        # 目标相机与轨道控制数学
│  │  └─ gsplat_renderer.py               # 单视角、串行 gsplat 渲染
│  ├─ composite/
│  │  └─ alpha.py                         # Alpha 边缘处理与逐帧合成
├─ tools/
│  ├─ fetch_test_assets.py                 # 官方素材下载、续传、解压、裁剪与哈希校验
│  └─ build_sidecar.py                     # PyInstaller 构建和 Tauri binary 命名
├─ packaging/
│  └─ gs-video-api.spec                    # 可复现 API sidecar 打包入口
└─ tests/
   ├─ assets/
   │  ├─ manifest.json                     # 来源、用途、许可证、选择规则
   │  └─ lock.json                         # 下载归档与选定文件 SHA-256
   ├─ fixtures/                            # 可提交的小型合成视频、PLY、相机与 Alpha 资产
   ├─ unit/
   ├─ integration/
   ├─ e2e/                                # API 与真实浏览器契约/E2E
   └─ security/                           # 令牌、CORS、路径和上传边界
```

---

### Task 1: 建立可测试的 Python 服务骨架与环境诊断

**Files:**
- Create: `pyproject.toml`
- Create: `src/gs_video/__init__.py`
- Create: `src/gs_video/__main__.py`
- Create: `src/gs_video/app.py`
- Create: `src/gs_video/environment/doctor.py`
- Create: `tests/unit/environment/test_doctor.py`

**Interfaces:**
- Produces: `EnvironmentReport`, `EnvironmentDoctor.check() -> EnvironmentReport`
- Produces: `python -m gs_video --doctor --json`

- [ ] **Step 1: 写环境诊断失败测试**

```python
# tests/unit/environment/test_doctor.py
from gs_video.environment.doctor import EnvironmentDoctor


def test_doctor_reports_missing_commands_without_starting_gpu() -> None:
    doctor = EnvironmentDoctor(which=lambda name: None, cuda_probe=lambda: (False, 0))
    report = doctor.check()
    assert report.ready is False
    assert {issue.code for issue in report.issues} == {
        "ffmpeg_missing",
        "ffprobe_missing",
        "cuda_unavailable",
    }
```

- [ ] **Step 2: 运行测试并确认因模块不存在而失败**

Run: `python -m pytest tests/unit/environment/test_doctor.py -v`

Expected: FAIL，包含 `ModuleNotFoundError: No module named 'gs_video'`。

- [ ] **Step 3: 创建包配置和最小诊断实现**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.build_meta"

[project]
name = "gs-video"
version = "0.1.0"
requires-python = ">=3.11,<3.12"
dependencies = [
  "fastapi>=0.128,<1",
  "numpy>=2,<3",
  "opencv-python-headless>=4.13,<5",
  "pillow>=11,<12",
  "plyfile>=1.1,<2",
  "pydantic>=2.11,<3",
  "python-multipart>=0.0.20,<1",
  "uvicorn[standard]>=0.35,<1",
]

[project.optional-dependencies]
dev = ["httpx>=0.28,<1", "mypy>=1.16,<2", "pyinstaller>=6.14,<7", "pytest>=8,<9", "ruff>=0.12,<1"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"

[tool.ruff]
line-length = 100

[tool.mypy]
python_version = "3.11"
strict = true
packages = ["gs_video"]
```

```python
# src/gs_video/environment/doctor.py
from collections.abc import Callable
from shutil import which

from pydantic import BaseModel


class EnvironmentIssue(BaseModel):
    code: str
    message: str


class EnvironmentReport(BaseModel):
    ready: bool
    vram_mb: int
    issues: list[EnvironmentIssue]


class EnvironmentDoctor:
    def __init__(
        self,
        which: Callable[[str], str | None] = which,
        cuda_probe: Callable[[], tuple[bool, int]] = lambda: (False, 0),
    ) -> None:
        self._which = which
        self._cuda_probe = cuda_probe

    def check(self) -> EnvironmentReport:
        issues: list[EnvironmentIssue] = []
        for command in ("ffmpeg", "ffprobe"):
            if self._which(command) is None:
                issues.append(EnvironmentIssue(
                    code=f"{command}_missing", message=f"未找到 {command}"
                ))
        cuda_ok, vram_mb = self._cuda_probe()
        if not cuda_ok:
            issues.append(EnvironmentIssue(
                code="cuda_unavailable", message="未检测到可用的 NVIDIA CUDA GPU"
            ))
        return EnvironmentReport(ready=not issues, vram_mb=vram_mb, issues=issues)
```

默认 `cuda_probe` 使用延迟导入，避免 `--doctor` 在未安装 PyTorch 时崩溃：

```python
def probe_cuda() -> tuple[bool, int]:
    try:
        import torch
    except ImportError:
        return False, 0
    if not torch.cuda.is_available():
        return False, 0
    properties = torch.cuda.get_device_properties(0)
    return True, int(properties.total_memory // (1024 * 1024))
```

`EnvironmentDoctor.__init__` 的默认 `cuda_probe` 必须指向 `probe_cuda`，测试继续注入假探针。

- [ ] **Step 4: 增加 `--doctor --json` 入口并运行质量门**

```python
# src/gs_video/__main__.py
import argparse

from gs_video.environment.doctor import EnvironmentDoctor


def main() -> int:
    parser = argparse.ArgumentParser(prog="gs-video")
    parser.add_argument("--doctor", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    if args.doctor:
        report = EnvironmentDoctor().check()
        print(report.model_dump_json(indent=2) if args.json else report)
        return 0 if report.ready else 2
    if args.serve:
        from gs_video.app import run_api
        return run_api(host=args.host, port=args.port)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

```python
# src/gs_video/app.py
def run_api(host: str, port: int) -> int:
    # Task 12 将此占位入口替换为 FastAPI app factory 与 Uvicorn 启动握手。
    print(f"GS Video API has not been assembled yet: {host}:{port}")
    return 0
```

Run: `python -m pytest tests/unit/environment/test_doctor.py -v`

Expected: PASS。

Run: `python -m ruff check src tests`

Expected: `All checks passed!`

- [ ] **Step 5: 提交**

```powershell
git add pyproject.toml src/gs_video tests/unit/environment
git commit -m "feat: scaffold core service and environment doctor"
```

### Task 2: 定义版本化项目模型、原子存储与迁移

**Files:**
- Create: `src/gs_video/domain/models.py`
- Create: `src/gs_video/project/migrations.py`
- Create: `src/gs_video/project/repository.py`
- Create: `tests/unit/project/test_repository.py`
- Create: `tests/unit/project/test_migrations.py`

**Interfaces:**
- Produces: `Project`, `StageName`, `StageState`, `StageStatus`, `ProjectRepository`
- Produces: `migrate_project_dict(raw: dict[str, object]) -> dict[str, object]`

- [ ] **Step 1: 写创建、原子保存和 v0→v1 迁移测试**

```python
def test_repository_round_trips_project(tmp_path: Path) -> None:
    repo = ProjectRepository(tmp_path)
    project = repo.create("demo")
    repo.save(project)
    loaded = repo.load()
    assert loaded.project_id == project.project_id
    assert loaded.schema_version == 1
    assert (tmp_path / "project.json").exists()


def test_migration_adds_stage_map() -> None:
    migrated = migrate_project_dict({"schema_version": 0, "name": "legacy"})
    assert migrated["schema_version"] == 1
    assert migrated["stages"] == {}
```

- [ ] **Step 2: 运行测试并确认缺少模型和仓储**

Run: `python -m pytest tests/unit/project -v`

Expected: FAIL，导入 `ProjectRepository` 和 `migrate_project_dict` 失败。

- [ ] **Step 3: 实现严格模型和迁移循环**

```python
# src/gs_video/domain/models.py
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class StageName(StrEnum):
    INGEST = "ingest"
    SEGMENT = "segment"
    SOLVE_CAMERA = "solve_camera"
    MAP_TRAJECTORY = "map_trajectory"
    RENDER = "render"
    COMPOSITE = "composite"
    EXPORT = "export"


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    STALE = "stale"


class StageState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: StageStatus = StageStatus.PENDING
    cache_key: str | None = None
    output_paths: list[str] = Field(default_factory=list)
    error_code: str | None = None


class Project(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int = 1
    project_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_video: str | None = None
    scene_ply: str | None = None
    stages: dict[StageName, StageState] = Field(default_factory=dict)
```

```python
# src/gs_video/project/migrations.py
CURRENT_SCHEMA_VERSION = 1


def migrate_project_dict(raw: dict[str, object]) -> dict[str, object]:
    data = dict(raw)
    version = int(data.get("schema_version", 0))
    while version < CURRENT_SCHEMA_VERSION:
        if version == 0:
            data.setdefault("stages", {})
            data["schema_version"] = 1
            version = 1
        else:
            raise ValueError(f"不支持的项目版本: {version}")
    if version > CURRENT_SCHEMA_VERSION:
        raise ValueError(f"项目版本 {version} 高于应用支持版本")
    return data
```

- [ ] **Step 4: 实现同目录临时文件替换的原子仓储**

```python
# src/gs_video/project/repository.py
import os
from pathlib import Path

from gs_video.domain.models import Project
from gs_video.project.migrations import migrate_project_dict


class ProjectRepository:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / "project.json"

    def create(self, name: str) -> Project:
        self.root.mkdir(parents=True, exist_ok=True)
        for folder in ("source", "proxies", "masks", "camera", "renders", "previews", "exports", "logs"):
            (self.root / folder).mkdir(exist_ok=True)
        return Project(name=name)

    def load(self) -> Project:
        raw = __import__("json").loads(self.path.read_text(encoding="utf-8"))
        return Project.model_validate(migrate_project_dict(raw))

    def save(self, project: Project) -> None:
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(project.model_dump_json(indent=2), encoding="utf-8")
        os.replace(temporary, self.path)
```

Run: `python -m pytest tests/unit/project -v`

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add src/gs_video/domain/models.py src/gs_video/project tests/unit/project
git commit -m "feat: add versioned project repository"
```

### Task 3: 建立阶段契约、缓存键、失效规则和可取消 runner

**Files:**
- Create: `src/gs_video/domain/errors.py`
- Create: `src/gs_video/domain/contracts.py`
- Create: `src/gs_video/project/cache.py`
- Create: `src/gs_video/pipeline/cancellation.py`
- Create: `src/gs_video/pipeline/events.py`
- Create: `src/gs_video/pipeline/runner.py`
- Create: `src/gs_video/pipeline/workflow.py`
- Create: `tests/unit/pipeline/test_runner.py`
- Create: `tests/unit/project/test_cache.py`

**Interfaces:**
- Produces: `Stage.execute(context, token, emit) -> StageResult`
- Produces: `PipelineRunner.run(stage_name)`, `CancellationToken.cancel()`
- Produces: `cache_key(stage, inputs, params, implementation_version) -> str`
- Produces: `invalidate_from(project, changed_stage) -> Project`

- [ ] **Step 1: 写缓存稳定性、取消和失败保留前序结果的测试**

```python
def test_cache_key_is_order_independent() -> None:
    left = cache_key("render", {"b": 2, "a": 1}, {"scale": 1.0}, "1")
    right = cache_key("render", {"a": 1, "b": 2}, {"scale": 1.0}, "1")
    assert left == right


def test_runner_marks_cancelled_without_deleting_previous_outputs(project: Project) -> None:
    token = CancellationToken()
    stage = CancellingStage(token)
    runner = PipelineRunner(project, {StageName.RENDER: stage}, save=lambda value: None)
    result = runner.run(StageName.RENDER, token)
    assert result.status is StageStatus.CANCELLED
    assert project.stages[StageName.INGEST].output_paths == ["source/meta.json"]
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/unit/pipeline tests/unit/project/test_cache.py -v`

Expected: FAIL，缺少 cache 与 runner。

- [ ] **Step 3: 实现错误分类、事件和取消 token**

```python
# src/gs_video/domain/errors.py
class GsVideoError(RuntimeError):
    code = "system_error"


class RepairableError(GsVideoError):
    code = "repairable"


class UnsupportedMaterialError(GsVideoError):
    code = "unsupported_material"


class CancelledError(GsVideoError):
    code = "cancelled"
```

```python
# src/gs_video/pipeline/cancellation.py
from threading import Event
from gs_video.domain.errors import CancelledError


class CancellationToken:
    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise CancelledError("任务已取消")
```

- [ ] **Step 4: 实现 canonical JSON 缓存键和显式阶段图**

```python
# src/gs_video/project/cache.py
import hashlib
import json
from typing import Any


def cache_key(stage: str, inputs: dict[str, Any], params: dict[str, Any], version: str) -> str:
    payload = {"stage": stage, "inputs": inputs, "params": params, "version": version}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
```

```python
# src/gs_video/pipeline/workflow.py
from gs_video.domain.models import StageName

DEPENDENCIES: dict[StageName, tuple[StageName, ...]] = {
    StageName.INGEST: (),
    StageName.SEGMENT: (StageName.INGEST,),
    StageName.SOLVE_CAMERA: (StageName.INGEST,),
    StageName.MAP_TRAJECTORY: (StageName.SOLVE_CAMERA,),
    StageName.RENDER: (StageName.MAP_TRAJECTORY,),
    StageName.COMPOSITE: (StageName.SEGMENT, StageName.RENDER),
    StageName.EXPORT: (StageName.COMPOSITE,),
}
```

- [ ] **Step 5: 实现 runner 的状态持久化与错误映射**

Runner 必须在进入阶段前写入 `RUNNING`，成功后写入 `SUCCEEDED`，捕获取消后写入 `CANCELLED`，捕获 `GsVideoError` 后写入 `FAILED` 与错误码；每次状态变化都调用仓储 `save()`。输出先写临时目录，只有 `StageResult` 成功后才登记为有效缓存。

```python
# src/gs_video/domain/contracts.py
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from gs_video.domain.models import Project, StageName
from gs_video.pipeline.cancellation import CancellationToken


@dataclass(frozen=True)
class StageResult:
    output_paths: tuple[Path, ...]
    cache_key: str


class Stage(Protocol):
    name: StageName

    def execute(
        self,
        project: Project,
        token: CancellationToken,
        emit: Callable[[int, int, str], None],
    ) -> StageResult: ...
```

```python
# src/gs_video/pipeline/runner.py
def run(self, name: StageName, token: CancellationToken) -> StageState:
    state = self.project.stages.setdefault(name, StageState())
    state.status = StageStatus.RUNNING
    self.save(self.project)
    try:
        result = self.stages[name].execute(self.project, token, self.emit)
        state.status = StageStatus.SUCCEEDED
        state.cache_key = result.cache_key
        state.output_paths = [str(path) for path in result.output_paths]
    except CancelledError:
        state.status = StageStatus.CANCELLED
    except GsVideoError as error:
        state.status = StageStatus.FAILED
        state.error_code = error.code
    self.save(self.project)
    return state
```

Run: `python -m pytest tests/unit/pipeline tests/unit/project/test_cache.py -v`

Expected: PASS。

- [ ] **Step 6: 提交**

```powershell
git add src/gs_video/domain src/gs_video/project/cache.py src/gs_video/pipeline tests/unit/pipeline tests/unit/project/test_cache.py
git commit -m "feat: add resumable cached pipeline runner"
```

### Task 4: 建立联网测试素材清单、锁文件和可恢复下载器

**Files:**
- Create: `.gitignore`
- Create: `tools/__init__.py`
- Create: `tools/fetch_test_assets.py`
- Create: `tests/assets/manifest.json`
- Create: `tests/assets/lock.json`
- Create: `tests/assets/NOTICE.md`
- Create: `tests/unit/tools/test_fetch_test_assets.py`

**Interfaces:**
- Produces: `python -m tools.fetch_test_assets fetch --group smoke|acceptance`
- Produces: `python -m tools.fetch_test_assets lock --acknowledge-source-review`
- Produces: `AssetManifest`, `AssetLock`, `fetch_asset(entry, lock, cache_root) -> list[Path]`
- Produces: `GS_VIDEO_TEST_ASSETS` 环境变量覆盖默认缓存目录

- [ ] **Step 1: 写路径安全、哈希失败和缓存命中测试**

```python
def test_rejects_zip_member_outside_destination(tmp_path: Path) -> None:
    archive = make_zip(tmp_path / "bad.zip", {"../../escape.txt": b"bad"})
    with pytest.raises(AssetSecurityError, match="越界路径"):
        extract_selected(archive, tmp_path / "output", ["**/*"])


def test_hash_mismatch_removes_partial_download(tmp_path: Path) -> None:
    target = tmp_path / "asset.zip"
    target.write_bytes(b"changed")
    with pytest.raises(AssetIntegrityError):
        verify_or_remove(target, "0" * 64)
    assert target.exists() is False


def test_valid_cached_file_does_not_open_network(tmp_path: Path) -> None:
    target = tmp_path / "asset.zip"
    target.write_bytes(b"locked")
    lock = locked_asset("demo", sha256_bytes(b"locked"))
    fetcher = AssetFetcher(opener=FailIfCalledOpener())
    assert fetcher.fetch(lock, target) == target
```

- [ ] **Step 2: 运行测试并确认工具尚不存在**

Run: `python -m pytest tests/unit/tools/test_fetch_test_assets.py -v`

Expected: FAIL，包含 `ModuleNotFoundError: No module named 'tools.fetch_test_assets'`。

- [ ] **Step 3: 定义官方来源清单与使用约束**

```json
{
  "schema_version": 1,
  "assets": [
    {
      "id": "davis-2017-trainval-480p",
      "group": "acceptance",
      "url": "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip",
      "source_page": "https://davischallenge.org/davis2017/code.html",
      "usage": "internal-research-evaluation",
      "citation": "Pont-Tuset et al., The 2017 DAVIS Challenge on Video Object Segmentation",
      "include": [
        "DAVIS/JPEGImages/480p/breakdance/**",
        "DAVIS/JPEGImages/480p/dance-jump/**",
        "DAVIS/JPEGImages/480p/dance-twirl/**",
        "DAVIS/JPEGImages/480p/parkour/**",
        "DAVIS/Annotations/480p/breakdance/**",
        "DAVIS/Annotations/480p/dance-jump/**",
        "DAVIS/Annotations/480p/dance-twirl/**",
        "DAVIS/Annotations/480p/parkour/**"
      ]
    },
    {
      "id": "graphdeco-3dgs-pretrained-models",
      "group": "acceptance",
      "url": "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip",
      "source_page": "https://github.com/graphdeco-inria/gaussian-splatting",
      "license_url": "https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md",
      "usage": "internal-noncommercial-research-evaluation-only",
      "include": [
        "train/point_cloud/iteration_30000/point_cloud.ply",
        "truck/point_cloud/iteration_30000/point_cloud.ply"
      ]
    }
  ]
}
```

`NOTICE.md` 必须说明：DAVIS 素材保留官方引用信息；Graphdeco 预训练模型只用于内部非商业研究与评估，不得随应用重新分发，商业化前必须换成自有或明确允许商业使用的场景。
由于 Graphdeco 官方预训练模型归档约 14 GB，`fetch --group acceptance --dry-run` 必须在下载前显示预计下载量、缓存位置和剩余磁盘空间；下载需要用户显式执行，不能作为普通单元测试的隐式副作用。

- [ ] **Step 4: 实现 HTTPS 下载、断点续传、选择性解压和锁定**

```python
def safe_member_path(root: Path, member: str) -> Path:
    destination = (root / member).resolve()
    if root.resolve() not in destination.parents:
        raise AssetSecurityError(f"压缩包包含越界路径: {member}")
    return destination


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def default_cache_root() -> Path:
    override = os.environ.get("GS_VIDEO_TEST_ASSETS")
    if override:
        return Path(override)
    return Path(os.environ["LOCALAPPDATA"]) / "GS-Video" / "TestAssets"
```

下载器只接受 HTTPS；使用 `.partial` 文件和 HTTP `Range` 续传；服务器不接受 Range 时重新下载；连接/读取超时均为 60 秒；重定向目标仍必须为 HTTPS。解压只处理 `include` 匹配项，并对归档和每个已选文件计算 SHA-256。

`lock` 子命令必须显式传入 `--acknowledge-source-review`，下载并生成 `tests/assets/lock.json`；后续 `fetch` 只信任锁文件中的归档哈希与成员哈希。CI 和验收不得自动刷新锁文件。

- [ ] **Step 5: 下载官方素材并提交锁文件**

Run: `python -m tools.fetch_test_assets lock --group acceptance --acknowledge-source-review`

Expected: 下载 DAVIS 2017 TrainVal 480p 与 Graphdeco 预训练模型归档；仅提取清单中的视频序列、标注和 `train`/`truck` PLY；`tests/assets/lock.json` 包含非空的归档与成员 SHA-256。

Run: `python -m tools.fetch_test_assets fetch --group acceptance --offline`

Expected: 不访问网络，全部从已验证缓存命中并通过哈希检查。

- [ ] **Step 6: 阻止大型素材进入 Git 并运行测试**

```gitignore
.cache/
tests/.assets/
*.partial
acceptance-report.json
```

Run: `python -m pytest tests/unit/tools/test_fetch_test_assets.py -v`

Expected: PASS。

Run: `git status --short`

Expected: 只显示工具、manifest、lock、NOTICE、测试和 `.gitignore`；不显示下载归档、视频帧或 PLY。

- [ ] **Step 7: 提交**

```powershell
git add .gitignore tools tests/assets tests/unit/tools
git commit -m "test: add locked network acceptance assets"
```

### Task 5: 实现视频探测、输入限制与代理帧生成

**Files:**
- Create: `src/gs_video/media/ffmpeg.py`
- Create: `src/gs_video/media/ingest.py`
- Create: `tests/unit/media/test_ffmpeg.py`
- Create: `tests/integration/media/test_ingest.py`
- Create: `tests/fixtures/media/README.md`

**Interfaces:**
- Produces: `probe_video(path: Path) -> VideoMetadata`
- Produces: `validate_source(metadata: VideoMetadata) -> None`
- Produces: `extract_proxy_frames(source, output_dir, max_height=540) -> list[Path]`
- Produces: `detect_shot_cuts(frame_paths, threshold=0.65) -> list[int]`

- [ ] **Step 1: 写 ffprobe 解析和输入拒绝测试**

```python
def test_parse_probe_preserves_fractional_frame_rate() -> None:
    metadata = parse_probe({
        "streams": [
            {"codec_type": "video", "width": 1920, "height": 1080,
             "avg_frame_rate": "30000/1001", "nb_frames": "300"},
            {"codec_type": "audio", "codec_name": "aac"},
        ],
        "format": {"duration": "10.01"},
    })
    assert metadata.fps.numerator == 30000
    assert metadata.fps.denominator == 1001
    assert metadata.has_audio is True


def test_rejects_video_longer_than_thirty_seconds() -> None:
    with pytest.raises(UnsupportedMaterialError, match="30 秒"):
        validate_source(VideoMetadata(width=1920, height=1080, duration=31, fps="30/1"))


def test_detects_abrupt_shot_cut(tmp_path: Path) -> None:
    frames = write_solid_frames(tmp_path, [0, 0, 255, 255])
    assert detect_shot_cuts(frames, threshold=0.65) == [2]
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/unit/media/test_ffmpeg.py -v`

Expected: FAIL，缺少 `parse_probe`。

- [ ] **Step 3: 实现无 shell 字符串拼接的 FFmpeg wrapper**

```python
def probe_video(path: Path) -> VideoMetadata:
    command = [
        "ffprobe", "-v", "error", "-show_streams", "-show_format",
        "-of", "json", str(path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return parse_probe(json.loads(completed.stdout))


def proxy_command(source: Path, output_dir: Path, max_height: int = 540) -> list[str]:
    return [
        "ffmpeg", "-y", "-i", str(source), "-an",
        "-vf", f"scale=-2:min({max_height}\\,ih)", "-q:v", "2",
        str(output_dir / "%06d.jpg"),
    ]
```

所有命令使用参数列表调用 `subprocess.run(..., shell=False)`；Windows 路径不做手工引号拼接。

代理帧生成后，将每帧 HSV 直方图与前一帧比较；Bhattacharyya 距离超过 `0.65` 且前后各两帧保持各自分布时记为切镜。检测到任何切镜即抛出 `UnsupportedMaterialError("检测到镜头切换")`。

```python
def detect_shot_cuts(frame_paths: list[Path], threshold: float = 0.65) -> list[int]:
    histograms = [normalized_hsv_histogram(cv2.imread(str(path))) for path in frame_paths]
    distances = [
        cv2.compareHist(histograms[index - 1], histograms[index], cv2.HISTCMP_BHATTACHARYYA)
        for index in range(1, len(histograms))
    ]
    return [index for index, distance in enumerate(distances, start=1) if distance >= threshold]
```

- [ ] **Step 4: 生成 2 秒小型测试视频并跑集成测试**

Run: `ffmpeg -y -f lavfi -i testsrc2=size=320x180:rate=10 -f lavfi -i sine=frequency=440 -t 10 -c:v libx264 -pix_fmt yuv420p -c:a aac tests/fixtures/media/source.mp4`

Run: `python -m pytest tests/integration/media/test_ingest.py -v`

Expected: PASS；输出 100 张代理帧，metadata 标记音轨存在。

- [ ] **Step 5: 提交**

```powershell
git add src/gs_video/media tests/unit/media tests/integration/media tests/fixtures/media
git commit -m "feat: validate and ingest source video"
```

### Task 6: 校验 Gaussian PLY、估算显存并建立场景相机数学

**Files:**
- Create: `src/gs_video/scene/ply.py`
- Create: `src/gs_video/scene/camera.py`
- Create: `tests/unit/scene/test_ply.py`
- Create: `tests/unit/scene/test_camera.py`
- Create: `tests/fixtures/scene/tiny_gaussians.ply`

**Interfaces:**
- Produces: `GaussianScene`, `load_gaussian_ply(path) -> GaussianScene`
- Produces: `estimate_scene_vram(scene, width, height) -> int`
- Produces: `OrbitCamera.view_matrix()`, `OrbitCamera.intrinsics(width, height)`

- [ ] **Step 1: 写缺字段拒绝和相机矩阵测试**

```python
def test_rejects_plain_xyz_point_cloud(tmp_path: Path) -> None:
    path = tmp_path / "plain.ply"
    path.write_text("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n")
    with pytest.raises(UnsupportedMaterialError, match="opacity"):
        load_gaussian_ply(path)


def test_intrinsics_put_principal_point_at_image_center() -> None:
    camera = OrbitCamera(target=(0, 0, 0), distance=2, yaw=0, pitch=0, fov_y_degrees=60)
    K = camera.intrinsics(1920, 1080)
    assert K[0, 2] == pytest.approx(960)
    assert K[1, 2] == pytest.approx(540)
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/unit/scene -v`

Expected: FAIL，场景模块不存在。

- [ ] **Step 3: 使用 plyfile 实现完整字段映射**

`GaussianScene` 固定包含 `means[N,3]`、`scales[N,3]`、`quats[N,4]`、`opacities[N]`、`colors[N,K,3]`。加载器接受 graphdeco 常用属性：`x/y/z`、`scale_0..2`、`rot_0..3`、`opacity`、`f_dc_0..2` 与连续 `f_rest_*`；对尺度和 opacity 保留原始值，由 renderer 在进入 gsplat 前分别应用 `exp` 和 `sigmoid`。

```python
REQUIRED_PROPERTIES = {
    "x", "y", "z", "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3", "opacity",
    "f_dc_0", "f_dc_1", "f_dc_2",
}
```

- [ ] **Step 4: 实现 orbit 相机和保守显存预估**

显存预估至少包含 Gaussian 张量字节数、RGBA framebuffer、投影中间量和 1.5 倍安全系数；当估算值超过可用显存的 80% 时返回拒绝建议。领域层相机统一保存 camera-to-world 矩阵；renderer 在边界求逆得到 OpenCV 约定的 world-to-camera view matrix。

```python
def estimate_scene_vram(scene: GaussianScene, width: int, height: int) -> int:
    gaussian_bytes = sum(array.nbytes for array in (
        scene.means, scene.scales, scene.quats, scene.opacities, scene.colors
    ))
    framebuffer_bytes = width * height * 4 * 4
    projection_bytes = scene.count * 48
    return int((gaussian_bytes + framebuffer_bytes + projection_bytes) * 1.5)


def intrinsics(width: int, height: int, fov_y_degrees: float) -> np.ndarray:
    fy = 0.5 * height / np.tan(np.deg2rad(fov_y_degrees) * 0.5)
    return np.array([[fy, 0.0, width / 2], [0.0, fy, height / 2], [0.0, 0.0, 1.0]])
```

Run: `python -m pytest tests/unit/scene -v`

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add src/gs_video/scene tests/unit/scene tests/fixtures/scene
git commit -m "feat: load Gaussian PLY and model scene camera"
```

### Task 7: 实现受控输入的 OpenCV 相机求解与轨迹映射

**Files:**
- Create: `src/gs_video/camera/classify.py`
- Create: `src/gs_video/camera/opencv_solver.py`
- Create: `src/gs_video/camera/mapping.py`
- Create: `tests/unit/camera/test_classify.py`
- Create: `tests/unit/camera/test_mapping.py`
- Create: `tests/integration/camera/test_solver.py`

**Interfaces:**
- Produces: `CameraSolution(intrinsics, camera_to_world, kind, confidence)`
- Produces: `OpenCvCameraSolver.solve(frame_paths, emit, token) -> CameraSolution`
- Produces: `map_trajectory(solution, target_start, motion_scale) -> list[np.ndarray]`

- [ ] **Step 1: 写 fixed/rotation/6DoF 分类与相对轨迹映射测试**

```python
def test_fixed_when_median_flow_is_subpixel() -> None:
    result = classify_motion(median_flow_px=0.3, homography_inliers=0.99, essential_inliers=0.1)
    assert result.kind is CameraKind.FIXED


def test_mapping_keeps_target_start_and_scales_only_translation() -> None:
    source = [np.eye(4), pose(tx=1.0, yaw_degrees=10)]
    target_start = pose(tx=5.0, yaw_degrees=90)
    mapped = map_relative_poses(source, target_start, motion_scale=0.25)
    np.testing.assert_allclose(mapped[0], target_start)
    assert np.linalg.norm(mapped[1][:3, 3] - mapped[0][:3, 3]) == pytest.approx(0.25)
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/unit/camera -v`

Expected: FAIL，缺少分类和映射函数。

- [ ] **Step 3: 实现逐帧特征跟踪和相对位姿**

每对相邻代理帧执行：灰度化 → `goodFeaturesToTrack(maxCorners=2000, qualityLevel=0.01, minDistance=8)` → `calcOpticalFlowPyrLK` → 双向误差过滤 → `findHomography(..., RANSAC)` 与 `findEssentialMat(..., cameraMatrix, RANSAC)` → `recoverPose()`。OpenCV 的 `R,t` 先组成 world-to-camera 增量，再求逆并累计为统一的 camera-to-world 位姿；同时记录特征数、RANSAC 内点率与 cheirality 内点率。

当特征少于 80、连续 5 帧求解失败或整体可信度低于 0.55 时抛出 `UnsupportedMaterialError("相机轨迹可信度过低")`。

```python
def solve_pair(previous: np.ndarray, current: np.ndarray, K: np.ndarray) -> PairPose:
    points0 = cv2.goodFeaturesToTrack(previous, 2000, 0.01, 8)
    if points0 is None or len(points0) < 80:
        raise UnsupportedMaterialError("相机轨迹可信度过低：特征不足")
    points1, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points0, None)
    back, back_status, _ = cv2.calcOpticalFlowPyrLK(current, previous, points1, None)
    error = np.linalg.norm(points0 - back, axis=2).reshape(-1)
    keep = status.reshape(-1).astype(bool) & back_status.reshape(-1).astype(bool) & (error < 1.0)
    matched0, matched1 = points0[keep, 0], points1[keep, 0]
    essential, mask = cv2.findEssentialMat(matched0, matched1, K, cv2.RANSAC, 0.999, 1.0)
    if essential is None or mask is None:
        raise UnsupportedMaterialError("相机轨迹可信度过低：本质矩阵失败")
    inliers, rotation, translation, pose_mask = cv2.recoverPose(
        essential, matched0, matched1, K, mask=mask
    )
    return PairPose(rotation=rotation, translation=translation[:, 0], inlier_count=int(inliers))
```

- [ ] **Step 4: 实现轨迹平滑和镜头类型简化**

- fixed：所有帧使用首帧位姿；
- rotation：只累计旋转，平移清零；
- 6DoF：累计旋转与单位平移方向，对平移应用 5 帧 Savitzky-Golay 等价的局部二次平滑；
- 映射：所有输入均为 camera-to-world，`relative(t) = inverse(source(0)) @ source(t)`，`target(t) = target_start @ relative(t)`；只对 `relative(t)` 的平移乘 `motion_scale`。进入 gsplat 前再对目标 camera-to-world 求逆得到 view matrix。

```python
def map_relative_poses(
    source_camera_to_world: list[np.ndarray],
    target_start: np.ndarray,
    motion_scale: float,
) -> list[np.ndarray]:
    source0_inverse = np.linalg.inv(source_camera_to_world[0])
    mapped: list[np.ndarray] = []
    for source_pose in source_camera_to_world:
        relative = source0_inverse @ source_pose
        relative = relative.copy()
        relative[:3, 3] *= motion_scale
        mapped.append(target_start @ relative)
    return mapped
```

Run: `python -m pytest tests/unit/camera tests/integration/camera -v`

Expected: PASS；合成旋转夹具被分类为 `rotation`，固定夹具被分类为 `fixed`。

- [ ] **Step 5: 提交**

```powershell
git add src/gs_video/camera tests/unit/camera tests/integration/camera
git commit -m "feat: solve and map controlled camera motion"
```

### Task 8: 实现 EdgeTAM 默认分割 worker 与 SAM 2.1 对照后端

**Files:**
- Modify: `src/gs_video/domain/contracts.py`
- Modify: `src/gs_video/environment/doctor.py`
- Create: `src/gs_video/segmentation/client.py`
- Create: `src/gs_video/segmentation/worker.py`
- Create: `tests/unit/segmentation/test_worker_client.py`
- Create: `tests/integration/segmentation/test_worker_protocol.py`

**Interfaces:**
- Produces: `ForegroundSegmenter.segment(frames, prompt, output_dir, emit, token) -> MaskSequence`
- Produces: newline-delimited JSON worker events: `progress`, `result`, `error`
- Consumes: `SegmentationBackend.EDGETAM|SAM2`、每个后端独立的 worker prefix、checkpoint 和 config

- [ ] **Step 1: 写 worker 命令、进度解析和取消测试**

```python
def test_client_parses_progress_and_result(tmp_path: Path) -> None:
    lines = [
        '{"type":"progress","current":1,"total":2}\n',
        '{"type":"result","mask_dir":"masks","frames":2}\n',
    ]
    client = VideoSegmenterClient(
        backend=SegmentationBackend.EDGETAM,
        process_factory=fake_process(lines),
    )
    result = client.segment([tmp_path / "000001.jpg"], Prompt(frame_index=0, x=10, y=20), tmp_path / "masks", lambda event: None, CancellationToken())
    assert result.frame_count == 2
    assert result.mask_dir == tmp_path / "masks"
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/unit/segmentation tests/integration/segmentation -v`

Expected: FAIL，缺少 `VideoSegmenterClient`。

- [ ] **Step 3: 实现不经 shell 的外部 worker 客户端**

客户端命令格式固定为：

```text
<worker-prefix> -m gs_video.segmentation.worker
  --backend edgetam
  --frames <proxy-dir>
  --output <mask-dir>
  --frame-index <int>
  --point <x>,<y>
  --config <edgetam.yaml>
  --checkpoint <edgetam.pt>
```

EdgeTAM 与 SAM 2.1 使用各自的 worker prefix，避免两个仓库都提供 `sam2` Python namespace 时发生包覆盖。prefix 可配置为原生 Python，也可配置为 `wsl.exe -d Ubuntu -- <venv-python>`。取消时先发送 `terminate()`，5 秒未退出再 `kill()`；无论何种退出都读取 stderr 并写入项目日志。

```python
def _start_worker(self, request: SegmentRequest) -> subprocess.Popen[str]:
    command = [
        *self.worker_prefix, "-m", "gs_video.segmentation.worker",
        "--backend", self.backend.value,
        "--frames", str(request.frames_dir),
        "--output", str(request.output_dir),
        "--frame-index", str(request.prompt.frame_index),
        "--point", f"{request.prompt.x},{request.prompt.y}",
        "--config", str(self.model_config),
        "--checkpoint", str(self.checkpoint),
    ]
    return subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, shell=False, creationflags=subprocess.CREATE_NO_WINDOW,
    )
```

- [ ] **Step 4: 实现共享 predictor 协议和双后端装配**

EdgeTAM worker 环境使用 EdgeTAM 官方 config/checkpoint 调用 `build_sam2_video_predictor`；SAM 2.1 worker 环境使用 SAM 2.1 Hiera Tiny config/checkpoint 调用同名入口。两者都初始化视频状态、在代表帧加入正点提示、向前传播全部帧，再从代表帧反向传播缺失帧。每帧写 8-bit PNG Alpha，文件名与代理帧编号一致。推理包裹在 `torch.inference_mode()` 与合适的 autocast 中；进程退出前删除 predictor 并调用 `torch.cuda.empty_cache()`。

```python
class SegmentationBackend(StrEnum):
    EDGETAM = "edgetam"
    SAM2 = "sam2"


def build_predictor(config: Path, checkpoint: Path) -> object:
    from sam2.build_sam import build_sam2_video_predictor
    return build_sam2_video_predictor(str(config), str(checkpoint))
```

集成测试不加载真实模型，而是向 worker 注入 `FakeVideoPredictor`，验证双向帧覆盖、PNG 命名和 JSONL 事件。真实 checkpoint smoke test 标记为 `@pytest.mark.gpu`。

```python
def propagate_masks(predictor: object, state: object, output_dir: Path) -> int:
    written: set[int] = set()
    for reverse in (False, True):
        for frame_index, object_ids, logits in predictor.propagate_in_video(state, reverse=reverse):
            if frame_index in written:
                continue
            mask = (logits[0] > 0).to(torch.uint8).mul(255).cpu().numpy().squeeze()
            Image.fromarray(mask, mode="L").save(output_dir / f"{frame_index + 1:06d}.png")
            written.add(frame_index)
            print(json.dumps({"type": "progress", "current": len(written), "total": len(state["images"])}), flush=True)
    return len(written)
```

Worker 完成后计算每帧非零 Alpha 比例；连续 15 帧低于 `0.001` 时返回 `unsupported_material`，信息为“主要人物长时间不可见”。环境诊断在启动分割前检查所选后端的 worker 命令、config 和 checkpoint 均存在且可读取，并通过 `--probe` 子进程确认实际载入的后端与配置一致。

Run: `python -m pytest tests/unit/segmentation tests/integration/segmentation -v -m "not gpu"`

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add src/gs_video/domain/contracts.py src/gs_video/segmentation tests/unit/segmentation tests/integration/segmentation
git commit -m "feat: add isolated EdgeTAM and SAM2 segmentation"
```

### Task 9: 实现 gsplat 串行渲染器和预览降采样

**Files:**
- Modify: `src/gs_video/domain/contracts.py`
- Modify: `src/gs_video/environment/doctor.py`
- Create: `src/gs_video/scene/gsplat_renderer.py`
- Create: `tests/unit/scene/test_renderer.py`
- Create: `tests/integration/scene/test_gsplat_smoke.py`

**Interfaces:**
- Produces: `SceneRenderer.render(scene, cameras, output_dir, settings, emit, token) -> RenderSequence`
- Produces: `SceneRenderer.render_pick(scene, camera, width, height) -> PickBuffer`
- Produces: `RenderSettings(width, height, sh_degree, background, preview_stride)`

- [ ] **Step 1: 写单帧调用形状、串行调用和取消测试**

```python
def test_renderer_calls_rasterizer_one_camera_at_a_time(tmp_path: Path) -> None:
    rasterizer = RecordingRasterizer()
    renderer = GsplatRenderer(rasterizer=rasterizer, device="cpu")
    result = renderer.render(tiny_scene(), [camera(), camera()], tmp_path, settings(), lambda event: None, CancellationToken())
    assert rasterizer.view_batch_sizes == [1, 1]
    assert result.frame_count == 2


def test_pick_buffer_contains_rgb_and_expected_depth(tmp_path: Path) -> None:
    renderer = GsplatRenderer(rasterizer=FakeRgbDepthRasterizer(), device="cpu")
    pick = renderer.render_pick(tiny_scene(), camera(), width=64, height=36)
    assert pick.rgb.shape == (36, 64, 3)
    assert pick.expected_depth.shape == (36, 64)
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/unit/scene/test_renderer.py -v`

Expected: FAIL，缺少 renderer。

- [ ] **Step 3: 实现 gsplat 1.x 参数转换和逐帧写出**

```python
render, alpha, meta = rasterization(
    means=means,
    quats=torch.nn.functional.normalize(quats, dim=-1),
    scales=torch.exp(scales),
    opacities=torch.sigmoid(opacities),
    colors=colors,
    viewmats=viewmat[None, ...],
    Ks=K[None, ...],
    width=settings.width,
    height=settings.height,
    sh_degree=settings.sh_degree,
    render_mode="RGB",
)
```

每次循环只保留一个相机的输出，立即转换为 `uint8` PNG 并释放 frame tensor。每帧检查取消；每 10 帧报告一次峰值显存。预览默认高度 540，最终渲染使用源分辨率。`render_pick()` 单独调用 `render_mode="RGB+ED"` 并只返回当前视口的 RGB 与 expected-depth；该深度不写入视频阶段缓存，也不传给合成器。

环境诊断在进入 renderer 前独立导入 `torch` 与 `gsplat`，检查 CUDA 可用、场景预估小于可用显存的 80%，并把版本写入项目阶段的实现版本字段。

- [ ] **Step 4: 跑 CPU fake 与 GPU smoke test**

Run: `python -m pytest tests/unit/scene/test_renderer.py -v`

Expected: PASS。

Run on configured NVIDIA machine: `python -m pytest tests/integration/scene/test_gsplat_smoke.py -v -m gpu`

Expected: PASS；tiny PLY 输出非空 RGB PNG，峰值显存小于 1 GB。

- [ ] **Step 5: 提交**

```powershell
git add src/gs_video/domain/contracts.py src/gs_video/scene/gsplat_renderer.py tests/unit/scene/test_renderer.py tests/integration/scene/test_gsplat_smoke.py
git commit -m "feat: render Gaussian backgrounds with gsplat"
```

### Task 10: 实现 Alpha 合成、音轨恢复和帧精确导出

**Files:**
- Create: `src/gs_video/composite/alpha.py`
- Create: `src/gs_video/media/export.py`
- Create: `tests/unit/composite/test_alpha.py`
- Create: `tests/integration/media/test_export.py`

**Interfaces:**
- Produces: `composite_frame(foreground, background, alpha, edge_px) -> Image`
- Produces: `export_mp4(frames_dir, source_video, fps, frame_count, output) -> ExportResult`

- [ ] **Step 1: 写 Alpha 端点、尺寸校验和音视频时长测试**

```python
def test_alpha_endpoints_select_exact_sources() -> None:
    fg = np.full((2, 2, 3), 200, np.uint8)
    bg = np.full((2, 2, 3), 20, np.uint8)
    alpha = np.array([[0, 255], [0, 255]], np.uint8)
    out = composite_frame(fg, bg, alpha, edge_px=0)
    np.testing.assert_array_equal(out[:, 0], bg[:, 0])
    np.testing.assert_array_equal(out[:, 1], fg[:, 1])
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/unit/composite tests/integration/media/test_export.py -v`

Expected: FAIL，缺少合成和导出模块。

- [ ] **Step 3: 实现线性 Alpha 与可关闭的边缘收缩/羽化**

Alpha 先归一化到 `[0,1]`；`edge_px>0` 时用椭圆核腐蚀 1–3 像素并高斯羽化，随后按 `fg * alpha + bg * (1-alpha)` 合成。前景、背景和 Alpha 尺寸不一致时抛出 `RepairableError`，不得静默拉伸人物。

```python
def composite_frame(
    foreground: np.ndarray,
    background: np.ndarray,
    alpha: np.ndarray,
    edge_px: int,
) -> np.ndarray:
    if foreground.shape != background.shape or foreground.shape[:2] != alpha.shape[:2]:
        raise RepairableError("前景、背景和 Alpha 尺寸不一致")
    matte = alpha.astype(np.float32) / 255.0
    if edge_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_px * 2 + 1,) * 2)
        matte = cv2.erode(matte, kernel)
        matte = cv2.GaussianBlur(matte, (0, 0), sigmaX=max(0.5, edge_px / 2))
    matte = matte[..., None]
    blended = foreground.astype(np.float32) * matte + background.astype(np.float32) * (1 - matte)
    return np.clip(blended, 0, 255).astype(np.uint8)
```

- [ ] **Step 4: 实现帧序列与源音轨导出**

```python
command = [
    "ffmpeg", "-y",
    "-framerate", f"{fps.numerator}/{fps.denominator}",
    "-i", str(frames_dir / "%06d.png"),
    "-i", str(source_video),
    "-map", "0:v:0", "-map", "1:a:0?",
    "-frames:v", str(frame_count),
    "-c:v", "libx264", "-pix_fmt", "yuv420p",
    "-c:a", "aac", "-shortest", str(output),
]
```

导出后再次用 ffprobe 校验视频帧数、视频时长和音轨存在性；时长误差大于一帧时删除不合格输出并抛出系统错误。

Run: `python -m pytest tests/unit/composite tests/integration/media/test_export.py -v`

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add src/gs_video/composite src/gs_video/media/export.py tests/unit/composite tests/integration/media/test_export.py
git commit -m "feat: composite frames and export synchronized mp4"
```

### Task 11: 组装完整工作流、缓存依赖和低分辨率预览

**Files:**
- Modify: `src/gs_video/pipeline/workflow.py`
- Modify: `src/gs_video/pipeline/runner.py`
- Create: `tests/integration/pipeline/test_workflow.py`
- Create: `tests/integration/pipeline/test_invalidation.py`

**Interfaces:**
- Produces: `build_mvp_workflow(services) -> PipelineRunner`
- Produces: `WorkflowServices` 显式依赖容器
- Consumes: Tasks 5–10 的全部稳定接口；Task 4 提供验收资产解析

- [ ] **Step 1: 写 mock 后端端到端阶段顺序与定向失效测试**

```python
def test_mvp_workflow_runs_expected_stage_order(project: Project) -> None:
    calls: list[str] = []
    runner = build_mvp_workflow(fake_services(calls), project)
    runner.run(StageName.EXPORT, CancellationToken())
    assert calls == [
        "ingest", "segment", "solve_camera", "map_trajectory",
        "render", "composite", "export",
    ]


def test_changing_target_camera_keeps_ingest_segment_and_solver_cached() -> None:
    project = completed_project()
    invalidate_for_change(project, ChangeKind.TARGET_CAMERA)
    assert project.stages[StageName.SEGMENT].status is StageStatus.SUCCEEDED
    assert project.stages[StageName.SOLVE_CAMERA].status is StageStatus.SUCCEEDED
    assert project.stages[StageName.MAP_TRAJECTORY].status is StageStatus.STALE
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/integration/pipeline -v`

Expected: FAIL，缺少 workflow builder 和失效规则。

- [ ] **Step 3: 为每个阶段定义输入摘要、参数和实现版本**

每个 stage 类只负责一次调用和产物登记。`WorkflowServices` 包含 `media_ingest`、`segmenter`、`camera_solver`、`trajectory_mapper`、`renderer`、`compositor`、`exporter`，测试中全部可替换。预览与最终渲染共享源轨迹、人物 Alpha 和目标相机参数，但使用不同的 render cache namespace。

```python
@dataclass(frozen=True)
class WorkflowServices:
    media_ingest: MediaIngest
    segmenter: ForegroundSegmenter
    camera_solver: CameraSolver
    trajectory_mapper: TrajectoryMapper
    renderer: SceneRenderer
    compositor: Compositor
    exporter: VideoExporter


def build_mvp_workflow(services: WorkflowServices, project: Project, save: SaveProject) -> PipelineRunner:
    stages: dict[StageName, Stage] = {
        StageName.INGEST: IngestStage(services.media_ingest),
        StageName.SEGMENT: SegmentStage(services.segmenter),
        StageName.SOLVE_CAMERA: SolveCameraStage(services.camera_solver),
        StageName.MAP_TRAJECTORY: MapTrajectoryStage(services.trajectory_mapper),
        StageName.RENDER: RenderStage(services.renderer),
        StageName.COMPOSITE: CompositeStage(services.compositor),
        StageName.EXPORT: ExportStage(services.exporter),
    }
    return PipelineRunner(project, stages, save=save)
```

- [ ] **Step 4: 实现变更到下游阶段的精确失效表**

```python
INVALIDATION_ROOT = {
    ChangeKind.SOURCE_VIDEO: (StageName.INGEST,),
    ChangeKind.SUBJECT_PROMPT: (StageName.SEGMENT,),
    ChangeKind.TARGET_CAMERA: (StageName.MAP_TRAJECTORY,),
    ChangeKind.MOTION_SCALE: (StageName.MAP_TRAJECTORY,),
    ChangeKind.EDGE_SETTINGS: (StageName.COMPOSITE,),
    ChangeKind.EXPORT_SETTINGS: (StageName.EXPORT,),
}
```

Run: `python -m pytest tests/integration/pipeline -v`

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add src/gs_video/pipeline tests/integration/pipeline
git commit -m "feat: assemble cached MVP processing workflow"
```

### Task 12: 建立安全、可恢复的 FastAPI 本地服务契约

**Files:**
- Modify: `src/gs_video/app.py`
- Modify: `src/gs_video/__main__.py`
- Create: `src/gs_video/api/auth.py`
- Create: `src/gs_video/api/schemas.py`
- Create: `src/gs_video/api/events.py`
- Create: `src/gs_video/api/routes.py`
- Create: `src/gs_video/api/uploads.py`
- Create: `tests/integration/api/test_projects.py`
- Create: `tests/integration/api/test_tasks.py`
- Create: `tests/security/test_local_api.py`

**Interfaces:**
- Produces: `create_app(settings, services) -> FastAPI`
- Produces: `GET /api/v1/bootstrap`、项目/素材/任务/预览/导出 REST 与 `WS /api/v1/events`
- Consumes: `ProjectRepository`、`EnvironmentDoctor`、`PipelineRunner`

- [ ] **Step 1: 写令牌、origin、任务恢复和 WebSocket 断线测试**

```python
def test_protected_route_rejects_missing_session_token(api_client) -> None:
    response = api_client.get("/api/v1/projects/current")
    assert response.status_code == 401


def test_task_state_is_recoverable_without_websocket(api_client, auth_headers) -> None:
    task_id = api_client.post(
        "/api/v1/tasks", json={"target_stage": "segment"}, headers=auth_headers
    ).json()["id"]
    snapshot = api_client.get(f"/api/v1/tasks/{task_id}", headers=auth_headers).json()
    assert snapshot["status"] in {"queued", "running", "succeeded"}
    assert "revision" in snapshot


def test_browser_upload_rejects_path_traversal_and_oversized_chunk(api_client, auth_headers) -> None:
    response = api_client.put(
        "/api/v1/uploads/u1/chunks/../../project.json",
        content=b"x" * (CHUNK_LIMIT + 1), headers=auth_headers,
    )
    assert response.status_code in {400, 413}
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/integration/api tests/security/test_local_api.py -v`

Expected: FAIL，API app factory 和路由不存在。

- [ ] **Step 3: 定义版本化 DTO、启动认证与严格本地边界**

`ApiSettings` 必须校验 host 只能是 loopback；端口传 `0` 由系统分配。桌面启动令牌使用 `secrets.token_urlsafe(32)`，只通过父子进程私有启动握手传递，不写日志或项目文件。认证后的 `/healthz` 只返回进程存活；`/api/v1/bootstrap` 返回 API 版本、能力、项目快照和环境报告并要求 Bearer token。开发 origin 来自显式 allowlist，不允许 `*`；WebSocket 还要单独验证 `Origin`，不能依赖 CORS 中间件。

```python
def require_session(
    authorization: Annotated[str | None, Header()] = None,
    settings: ApiSettings = Depends(get_settings),
) -> None:
    scheme, _, value = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(value, settings.session_token):
        raise HTTPException(status_code=401, detail="invalid session")
```

禁止 API 接收任意“输出绝对路径”后直接写入。桌面路径导入先经过 `PlatformBridge` 用户选择，再由 API canonicalize 并复制/引用到项目；浏览器上传只能写入服务器分配的 upload ID 目录。所有错误返回稳定的 `code`、`category`、`message`、`retryable`。

- [ ] **Step 4: 实现任务 REST、可重连事件流和 lifespan 清理**

任务创建返回 `202` 与 task ID；`GET /tasks/{id}` 是权威状态。WebSocket 连接建立后必须在 3 秒内把令牌放在首个 `authenticate` 消息中，认证前不订阅也不发送业务事件，避免把令牌放入 URL；认证失败立即以策略错误关闭。事件包含 `task_id`、单调 `revision`、阶段、帧进度和错误摘要，不发送大二进制。客户端认证后发送 `resume(after_revision)`；若内存事件窗口已丢失，服务端发送 `resync_required`，客户端随后 GET 快照。

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    await app.state.task_service.start()
    try:
        yield
    finally:
        await app.state.task_service.cancel_all()
        await app.state.worker_registry.terminate_all()
```

`TaskService` 将同步 `PipelineRunner` 放入受控线程/进程执行器，取消请求映射到 `CancellationToken`。服务关闭时先协作取消，再限时终止剩余 worker 进程树。

- [ ] **Step 5: 实现浏览器分块上传**

`POST /uploads` 预声明文件名、MIME、总大小和 SHA-256；`PUT /uploads/{id}/chunks/{index}` 使用固定上限与偏移验证；`POST /uploads/{id}/complete` 校验块数、总大小和整文件哈希后原子移动到项目输入区。支持查询已上传块和取消，测试中覆盖重复块幂等、断点续传、磁盘不足、哈希不符和取消清理。

Run: `python -m pytest tests/integration/api tests/security -v`

Expected: PASS。

- [ ] **Step 6: 提交**

```powershell
git add src/gs_video/app.py src/gs_video/__main__.py src/gs_video/api tests/integration/api tests/security
git commit -m "feat: expose secure recoverable local API"
```

### Task 13: 建立共享 React/Vite SPA、客户端契约和双平台桥

**Files:**
- Create: `package.json`
- Create: `package-lock.json`
- Create: `tsconfig.base.json`
- Create: `apps/web/package.json`
- Create: `apps/web/vite.config.ts`
- Create: `apps/web/vitest.config.ts`
- Create: `apps/web/index.html`
- Create: `apps/web/src/api/types.ts`
- Create: `apps/web/src/api/backend-client.ts`
- Create: `apps/web/src/api/http-backend-client.ts`
- Create: `apps/web/src/api/task-events.ts`
- Create: `apps/web/src/platform/platform-bridge.ts`
- Create: `apps/web/src/platform/browser-platform-bridge.ts`
- Create: `apps/web/src/platform/tauri-platform-bridge.ts`
- Create: `apps/web/src/test/setup.ts`
- Create: `apps/web/src/api/http-backend-client.test.ts`
- Create: `apps/web/src/platform/platform-boundary.test.ts`

**Interfaces:**
- Produces: `BackendClient`、`TaskEventSource`、`PlatformBridge`
- Produces: 浏览器与 Tauri 两个 composition root，业务组件无平台导入

- [ ] **Step 1: 写客户端认证、重连收敛和平台边界失败测试**

```ts
it('resyncs authoritative task state after an event gap', async () => {
  const client = fakeBackendClient({ task: { id: 't1', revision: 9, status: 'running' } })
  const store = createTaskStore(client)
  store.onEvent({ type: 'resync_required', taskId: 't1', revision: 9 })
  await store.whenIdle()
  expect(client.getTask).toHaveBeenCalledWith('t1')
  expect(store.snapshot().revision).toBe(9)
})

it('keeps tauri imports outside business features', async () => {
  const forbidden = await findImports('apps/web/src/features', /^@tauri-apps\//)
  expect(forbidden).toEqual([])
})
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `npm ci && npm run test:web -- --run`

Expected: FAIL，workspace 和客户端模块不存在。

- [ ] **Step 3: 建立 Vite/React/Vitest 严格工程**

根 workspace 固定 Node 版本与 lockfile。Vite 开发服务器只代理 `/api` 和 `/ws` 到显式本地 API 地址；生产构建使用相对 base 以供 Tauri 加载。Vitest 使用 `jsdom`、setup file、自动恢复 mock 和 V8 coverage；React Testing Library 只测试用户可见行为。CI 一律 `vitest run`，不进入 watch。

```ts
// apps/web/vitest.config.ts
export default defineConfig({
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    clearMocks: true,
    restoreMocks: true,
    coverage: { provider: 'v8', reporter: ['text', 'lcov'] },
  },
})
```

- [ ] **Step 4: 实现 REST/WS 客户端与可恢复 store**

`HttpBackendClient` 统一设置 API 版本、Bearer token、超时、AbortSignal 和稳定错误解析。`TaskEventSource` 在 WebSocket 建立后先发送认证消息，收到 `authenticated` 后才恢复 revision。会话配置只驻留内存；不得放入 URL、localStorage 或日志。任务 store 按 revision 去重；WebSocket 清理在 React effect cleanup 中完成，外部状态订阅使用稳定 subscribe 函数。任何连接状态变化只影响实时提示，不改变 REST 权威状态。

```ts
export interface BackendClient {
  bootstrap(signal?: AbortSignal): Promise<BootstrapDto>
  importLocalPath(kind: AssetKind, path: string): Promise<AssetDto>
  createUpload(input: UploadInit): Promise<UploadSessionDto>
  putUploadChunk(id: string, index: number, data: Blob, signal?: AbortSignal): Promise<void>
  getProject(): Promise<ProjectDto>
  updateProject(patch: ProjectPatch): Promise<ProjectDto>
  startTask(targetStage: StageName): Promise<TaskDto>
  getTask(id: string): Promise<TaskDto>
  cancelTask(id: string): Promise<TaskDto>
}
```

- [ ] **Step 5: 实现 Browser/Tauri PlatformBridge**

```ts
export interface PlatformBridge {
  readonly kind: 'browser' | 'tauri'
  pickInputFile(options: PickFileOptions): Promise<PickedFile | null>
  saveExport(suggestedName: string, source: ExportSource): Promise<void>
  revealPath?(path: string): Promise<void>
  openExternal(url: string): Promise<void>
}
```

浏览器实现返回 `File` 并走分块上传，导出使用受用户手势触发的下载；Tauri 实现动态导入 dialog/opener API，返回本地路径并调用 `importLocalPath`。在线资源目录未来也只能通过 `BackendClient`，不得加入 bridge 或在 WebView 直接下载。

浏览器 composition root 在没有桌面注入配置时显示本地连接页：用户粘贴由 `python -m gs_video --serve` 在交互式终端仅显示一次的端口与会话令牌；连接成功后立即清空输入值，令牌只保存在内存。Tauri composition root 由 Rust 在 WebView 显示前注入同样的内存配置，不出现连接页。该连接动作不计入三次创作关键交互。

Run: `npm run typecheck:web && npm run test:web -- --run && npm run build:web`

Expected: 全部 PASS，生产 bundle 中业务 feature 不含直接 Tauri import。

- [ ] **Step 6: 提交**

```powershell
git add package.json package-lock.json tsconfig.base.json apps/web
git commit -m "feat: add shared browser-ready React client"
```

### Task 14: 实现三次关键交互的 React 向导与场景视口

**Files:**
- Create: `apps/web/src/app/app.tsx`
- Create: `apps/web/src/app/project-store.ts`
- Create: `apps/web/src/features/import/import-page.tsx`
- Create: `apps/web/src/features/subject/subject-page.tsx`
- Create: `apps/web/src/features/camera/camera-page.tsx`
- Create: `apps/web/src/features/camera/scene-viewport.tsx`
- Create: `apps/web/src/features/preview/preview-page.tsx`
- Create: `apps/web/src/features/export/export-page.tsx`
- Create: `apps/web/src/features/workflow/guided-workflow.test.tsx`
- Create: `apps/web/src/features/camera/scene-viewport.test.tsx`

**Interfaces:**
- Produces: 同一 SPA 中完整导入、人物、机位、落脚点、预览与导出流程
- Consumes: Task 12 API 与 Task 13 adapters

- [ ] **Step 1: 写三次交互、页面门控和错误恢复失败测试**

```tsx
it('completes the guided workflow with three creative interactions', async () => {
  const user = userEvent.setup()
  render(<App backend={fakeBackend()} platform={fakePlatform()} />)
  await importPreparedAssets(user)
  await user.click(screen.getByLabelText('人物位置 100,120'))
  await user.click(screen.getByRole('button', { name: '确认初始机位' }))
  await user.click(screen.getByLabelText('场景落脚点 320,180'))
  await user.click(screen.getByRole('button', { name: '生成预览' }))
  expect(await screen.findByRole('button', { name: '导出视频' })).toBeEnabled()
  expect(interactionCounter()).toBe(3)
})
```

导入文件选择、等待任务、调整自动默认值和导出保存不计入三次“创作关键交互”；人物点击、确认机位、落脚点点击各计一次。

- [ ] **Step 2: 运行测试并确认失败**

Run: `npm run test:web -- --run apps/web/src/features/workflow/guided-workflow.test.tsx`

Expected: FAIL，向导页面不存在。

- [ ] **Step 3: 实现以项目快照为状态源的向导**

页面完成度由 API 项目快照和阶段状态推导，不把业务真相藏在组件 state。左侧显示导入、人物、机位、预览、导出步骤；底部提供上一步、下一步、取消任务。导入页显示视频摘要、Gaussian 数量、显存估算和可修复错误。人物页在代理帧上把 CSS 坐标转换为图像像素坐标，提交 prompt 后显示 Alpha 叠加。

任务进度来自事件 store；收到 WS 事件只更新可验证 revision。刷新页面、WS 断开或 Tauri WebView 重载后调用 bootstrap/project/task REST 恢复当前页与任务状态。

- [ ] **Step 4: 实现场景视口、机位和落脚点**

视口显示后端生成的最近预览帧；拖动更新 yaw/pitch，滚轮更新 distance，滑杆调整垂直 FOV。交互期间 150 ms debounce 请求 540p 单帧预览，并以 generation ID + AbortController 忽略旧响应。点击“确认初始机位”才计第二次交互。

落脚点点击向后端发送视口像素、当前相机 revision 和 pick-buffer revision；后端用当前 RGB+ED 单帧深度执行反投影并返回世界坐标。revision 不匹配、深度无效或点击超出内容区时拒绝确认。深度只用于拾取，不进入最终人物遮挡。

```ts
function toImagePoint(event: PointerEvent, bounds: DOMRect, image: ImageSize): Point {
  const scale = Math.min(bounds.width / image.width, bounds.height / image.height)
  const left = bounds.left + (bounds.width - image.width * scale) / 2
  const top = bounds.top + (bounds.height - image.height * scale) / 2
  return { x: (event.clientX - left) / scale, y: (event.clientY - top) / scale }
}
```

- [ ] **Step 5: 实现预览、导出、取消、重试与可访问性**

预览页显示低分辨率合成、焦距、`motion_scale` 和阶段缓存状态；修改参数只调用对应 patch，后端负责定向失效。失败页按错误类别显示重新选人物、降低分辨率、返回机位或重试。导出成功必须来自后端 ffprobe 后验验证。键盘可完成全部按钮/滑杆操作；画布点击提供坐标文本替代输入；任务进度使用 `aria-live="polite"`，错误使用聚焦后的 alert。

Run: `npm run typecheck:web && npm run test:web -- --run && npm run build:web`

Expected: 全部 PASS。

- [ ] **Step 6: 提交**

```powershell
git add apps/web/src
git commit -m "feat: implement guided Gaussian video workflow"
```

### Task 15: 打包 FastAPI sidecar 并实现最小权限 Tauri 2 宿主

**Files:**
- Create: `packaging/gs-video-api.spec`
- Create: `tools/build_sidecar.py`
- Create: `tests/unit/tools/test_build_sidecar.py`
- Create: `apps/desktop/src-tauri/Cargo.toml`
- Create: `apps/desktop/src-tauri/build.rs`
- Create: `apps/desktop/src-tauri/tauri.conf.json`
- Create: `apps/desktop/src-tauri/capabilities/default.json`
- Create: `apps/desktop/src-tauri/src/lib.rs`
- Create: `apps/desktop/src-tauri/src/sidecar.rs`
- Create: `apps/desktop/src-tauri/tests/sidecar_lifecycle.rs`

**Interfaces:**
- Produces: `python tools/build_sidecar.py --target <rust-target-triple>`
- Produces: Tauri 启动握手、健康检查、优雅关闭与强制清理

- [ ] **Step 1: 写 sidecar 命名、握手解析和生命周期失败测试**

```python
def test_sidecar_name_contains_tauri_target_triple(tmp_path: Path) -> None:
    output = plan_sidecar_output(tmp_path, "x86_64-pc-windows-msvc")
    assert output.name == "gs-video-api-x86_64-pc-windows-msvc.exe"
```

Rust 集成测试使用 fake sidecar：输出单行 JSON handshake 后常驻；验证窗口启动前得到 port/token、正常关闭发送终止请求、超时后 kill，且 stdout 日志不会包含 token。

- [ ] **Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/unit/tools/test_build_sidecar.py -v`

Run: `cargo test --manifest-path apps/desktop/src-tauri/Cargo.toml`

Expected: FAIL，打包器和 Tauri crate 不存在。

- [ ] **Step 3: 建立可审查的 PyInstaller onefile 构建**

使用 spec 文件固定 entrypoint、datas、hidden imports 和 exclusions。Tauri bundle 使用 onefile，因为 API sidecar 明确排除 EdgeTAM、SAM 2、gsplat、PyTorch 和 CUDA runtime，体积与启动解压成本可控；这些 GPU 组件仍位于独立可配置 worker 环境。诊断时允许显式 `--mode onedir` 生成可检查目录，但发布门只接受 onefile。构建脚本创建 Tauri 要求的 target-triple 可执行文件，输出 manifest 与 SHA-256。

```python
# packaging/gs-video-api.spec 核心意图
a = Analysis(
    ["src/gs_video/__main__.py"],
    hiddenimports=["uvicorn.logging", "uvicorn.loops.auto", "uvicorn.protocols.http.auto"],
    excludes=["torch"],
)
```

- [ ] **Step 4: 实现 Tauri sidecar 启动与私有握手**

`tauri.conf.json` 的 `externalBin` 只声明 `binaries/gs-video-api`。Tauri 启动 sidecar 时只通过参数传入 loopback 与 port 0，一次性 token 通过继承的私有 stdin pipe 发送，禁止出现在命令行、环境变量、stdout 或日志中；sidecar 绑定成功后向 stdout 写一行 JSON `{port, apiVersion, pid}`。Rust 端验证 schema，以令牌调用认证的 `/healthz`，将 bootstrap 配置注入 WebView 内存后再显示主窗口。若 15 秒未就绪或健康检查失败，显示可诊断启动错误并清理进程。

Tauri capability 只允许固定 sidecar execute、dialog 和 opener 所需动作；不允许 `shell:allow-open` 通配、任意 command args 或宽文件系统 scope。CSP 只允许本地应用资源和当前 loopback API/WS；release 禁用开发者工具。

- [ ] **Step 5: 实现关闭顺序和 worker 树回收**

窗口关闭先调用认证的 API shutdown/cancel，等待 API 清理 GPU workers；超时后 Rust 终止 sidecar job/process tree。崩溃重启检测未完成任务并依赖 `project.json`/REST 状态恢复，不尝试复用旧 token 或旧端口。

Run: `python tools/build_sidecar.py --target x86_64-pc-windows-msvc --dry-run`

Run: `cargo test --manifest-path apps/desktop/src-tauri/Cargo.toml`

Run: `npm run build:web && npm run tauri:build -- --debug`

Expected: 测试通过，debug bundle 中包含命名正确的 sidecar 和 web dist。

- [ ] **Step 6: 提交**

```powershell
git add packaging tools/build_sidecar.py tests/unit/tools apps/desktop package.json package-lock.json
git commit -m "feat: host local API in Tauri sidecar"
```

### Task 16: 建立浏览器与 Tauri 双宿主 E2E 和安全回归门

**Files:**
- Create: `playwright.config.ts`
- Create: `tests/e2e/browser/guided-workflow.spec.ts`
- Create: `tests/e2e/browser/reconnect.spec.ts`
- Create: `tests/e2e/browser/upload-resume.spec.ts`
- Create: `tests/e2e/contracts/desktop-platform.spec.ts`
- Create: `scripts/check_frontend_boundaries.mjs`
- Modify: `package.json`

**Interfaces:**
- Produces: 同一 mock API 下浏览器/桌面 adapter 契约套件
- Produces: 真实浏览器三次交互 E2E 与 sidecar smoke

- [ ] **Step 1: 写浏览器刷新恢复、断点上传与平台契约 E2E**

浏览器测试启动真实 FastAPI（mock pipeline services）和 Vite preview，使用临时项目目录。覆盖：上传中断后续传、WS 断线后 REST 收敛、页面刷新后恢复任务、三次关键交互后导出、取消后无孤儿任务。平台契约用相同案例分别运行 BrowserBridge 与 fake TauriBridge，确保语义一致。

- [ ] **Step 2: 运行测试并确认失败**

Run: `npm run test:e2e:browser`

Expected: FAIL，E2E harness 和边界检查器不存在。

- [ ] **Step 3: 实现确定性的双宿主 harness**

测试 API 使用固定端口仅限测试进程、固定 token 通过 Playwright bootstrap fixture 注入内存；REST 断言使用 Bearer header，WebSocket 仍走真实首消息认证。生产代码使用随机端口和随机 token。mock pipeline 保留真实项目存储、任务 revision、取消和导出后验契约，只替换昂贵 GPU 实现。每次测试结束确认 API、worker、临时文件均清理。

桌面 smoke 不重复全套 WebView UI 自动化：Rust lifecycle 测试验证 sidecar，浏览器 E2E 验证共享 SPA，另用一次 debug bundle smoke 验证 WebView 能 bootstrap、原生对话框 adapter 被调用、关闭后 sidecar 退出。

- [ ] **Step 4: 建立静态边界和安全回归检查**

`check_frontend_boundaries.mjs` 失败条件：`features/` 或 `api/` 直接导入 `@tauri-apps/*`；前端出现硬编码生产端口/token；WebView 直接请求外部资源目录；Tauri capability 出现宽 shell/fs scope。Python 安全测试继续覆盖非 loopback bind、CORS、WS 未认证、上传路径逃逸、符号链接/重解析点和输出路径注入。

Run: `npm run lint:boundaries && npm run test:e2e:browser && npm run test:desktop-smoke`

Expected: 全部 PASS。

- [ ] **Step 5: 提交**

```powershell
git add playwright.config.ts tests/e2e scripts/check_frontend_boundaries.mjs package.json package-lock.json
git commit -m "test: verify browser and Tauri host parity"
```

### Task 17: 建立 8 GB 验收工具、文档和完整发布门

**Files:**
- Create: `scripts/run_acceptance.py`
- Create: `tests/acceptance/cases.json`
- Create: `tests/acceptance/segmentation_backends.json`
- Create: `tests/acceptance/test_cases.py`
- Create: `README.md`
- Create: `docs/development/model-installation.md`
- Create: `docs/development/acceptance.md`

**Interfaces:**
- Produces: `python scripts/run_acceptance.py --cases tests/acceptance/cases.json --report acceptance-report.json`
- Produces: 包含自动闭环率、失败分类、峰值显存、帧数和音视频误差的 JSON 报告

- [ ] **Step 1: 写验收报告聚合失败测试**

```python
def test_report_requires_eighty_percent_automatic_completion() -> None:
    results = [case_result(success=index < 9) for index in range(12)]
    report = AcceptanceReport.from_results(results)
    assert report.completion_rate == pytest.approx(0.75)
    assert report.passed is False
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/acceptance/test_cases.py -v`

Expected: FAIL，缺少 `AcceptanceReport`。

- [ ] **Step 3: 实现固定案例清单与报告器**

`cases.json` 不记录机器相关绝对路径，而是引用 Task 4 的 `asset_id` 和派生素材 ID。DAVIS 帧序列通过固定 FFmpeg 参数重定时为 10 秒、1080p H.264 测试视频，并加入确定性的正弦测试音轨；Graphdeco `train` 与 `truck` PLY 使用固定随机种子生成 250k/500k Gaussian 的派生副本。所有派生产物、转换参数和 SHA-256 写入 lock，保证不同机器得到相同测试输入。

每个案例记录视频 asset ID、场景 asset ID、人物提示、目标相机、落脚点、运动幅度和预期镜头类型。报告器记录：是否导出、是否人工修改中间文件、失败分类、每阶段耗时、每阶段峰值显存、输出帧数、音视频时长误差。缺失素材时标记 `fixture_missing` 并使发布门失败，不得跳过。

```python
class AcceptanceReport(BaseModel):
    case_count: int
    completion_rate: float
    peak_vram_mb: int
    failures: dict[str, int]
    passed: bool

    @classmethod
    def from_results(cls, results: list[CaseResult]) -> "AcceptanceReport":
        successes = sum(result.success and not result.edited_intermediate for result in results)
        completion_rate = successes / len(results) if results else 0.0
        peak_vram_mb = max((result.peak_vram_mb for result in results), default=0)
        failures = Counter(result.failure_code for result in results if not result.success)
        return cls(
            case_count=len(results), completion_rate=completion_rate,
            peak_vram_mb=peak_vram_mb, failures=dict(failures),
            passed=len(results) >= 12 and completion_rate >= 0.8 and peak_vram_mb <= 8192,
        )
```

- [ ] **Step 4: 对比 EdgeTAM 默认后端和 SAM 2.1 对照后端**

`segmentation_backends.json` 选择 DAVIS 的四个人物序列。提示点由首帧真值 Mask 的最大内切位置确定，两个后端接收完全相同的帧和提示。报告平均 IoU、Mask 丢失帧数、耗时和峰值显存。

```python
def backend_gate(edgetam: BackendMetrics, sam2: BackendMetrics) -> bool:
    return (
        edgetam.peak_vram_mb <= 8192
        and edgetam.mean_iou >= 0.75
        and edgetam.missing_mask_frames == 0
        and edgetam.mean_iou >= sam2.mean_iou - 0.07
    )
```

EdgeTAM 是应用默认值；SAM 2.1 只用于显式诊断对照，不做静默自动回退，避免一次任务连续加载两个模型。若上述 gate 失败，发布门失败并要求重新评估默认后端。

- [ ] **Step 5: 写开发、模型和测试素材安装文档**

README 必须包含 Node/Rust/Tauri 前置条件、Python 3.11 venv、`pip install -e ".[dev]"`、`npm ci`、FFmpeg PATH、CUDA/PyTorch 单独安装、EdgeTAM 默认 worker、SAM 2.1 Tiny 对照 worker、gsplat 安装、联网素材下载、`python -m gs_video --doctor --json`、浏览器开发模式、Tauri 开发模式、sidecar 构建和测试命令。模型文档明确记录独立 worker 环境、Windows/WSL 选项、第三方模型许可证和离线 checkpoint 路径。验收文档明确记录 DAVIS 引用、Graphdeco 非商业研究限制、约 14 GB 的一次性归档下载和缓存迁移方式。

```markdown
## Local development

1. Create a Python 3.11 virtual environment.
2. Run `python -m pip install -e ".[dev]"`.
3. Install the pinned Node.js and Rust toolchains, then run `npm ci`.
4. Install the CUDA-enabled PyTorch build that matches the local driver.
5. Install EdgeTAM in the default GPU worker environment and SAM 2.1 Tiny in a separate comparison environment.
6. Install gsplat in the renderer environment and add `ffmpeg`/`ffprobe` to `PATH`.
7. Run `python -m tools.fetch_test_assets fetch --group acceptance` once while online.
8. Run `python -m gs_video --doctor --json` before starting browser or Tauri development.
9. Use `npm run dev:web` for browser UI and `npm run tauri:dev` for the desktop host.
```

- [ ] **Step 6: 运行完整发布门**

Run: `python -m ruff check src tests scripts`

Expected: `All checks passed!`

Run: `python -m mypy src/gs_video`

Expected: `Success: no issues found`。

Run: `python -m pytest tests/unit tests/integration tests/e2e -v -m "not gpu"`

Expected: 全部 PASS。

Run: `npm run lint:boundaries && npm run typecheck:web && npm run test:web -- --run && npm run build:web`

Expected: 全部 PASS，且业务 feature 无直接 Tauri API 依赖。

Run: `npm run test:e2e:browser`

Expected: Chromium 中导入/上传、刷新恢复、WS 重连和三次交互闭环全部 PASS。

Run: `cargo test --manifest-path apps/desktop/src-tauri/Cargo.toml && npm run test:desktop-smoke`

Expected: sidecar 生命周期、最小权限配置和 WebView bootstrap smoke 全部 PASS；退出后无 API/GPU worker 残留。

Run on 8 GB NVIDIA acceptance machine: `python -m pytest tests/integration/scene/test_gsplat_smoke.py -v -m gpu`

Expected: PASS。

Run on 8 GB NVIDIA acceptance machine: `python -m tools.fetch_test_assets fetch --group acceptance --offline`

Expected: PASS；所有网络素材和派生产物均从缓存命中并通过锁文件哈希校验。

Run on 8 GB NVIDIA acceptance machine: `python scripts/run_acceptance.py --segmentation-backends tests/acceptance/segmentation_backends.json --report segmentation-report.json`

Expected: EdgeTAM 峰值显存不超过 8 GB、平均 IoU 至少 0.75、无 Mask 丢失帧，且平均 IoU 不低于 SAM 2.1 Tiny 超过 0.07。

Run on 8 GB NVIDIA acceptance machine: `python scripts/run_acceptance.py --cases tests/acceptance/cases.json --report acceptance-report.json`

Expected: 12 个案例全部执行；自动闭环成功率至少 80%；规定案例峰值显存不超过 8 GB；成功案例音视频时长误差不超过一帧。

- [ ] **Step 7: 提交**

```powershell
git add scripts tests/acceptance README.md docs/development
git commit -m "test: add MVP acceptance and release gate"
```

## 实施顺序与检查点

- Tasks 1–3 完成后：得到可测试的项目状态与阶段框架，进行一次架构审查。
- Task 4 完成后：官方测试素材已下载、审查、锁定并可离线复用，大型二进制未进入 Git。
- Tasks 5–7 完成后：用固定视频验证媒体、PLY 和相机轨迹，不加载分割模型或 gsplat。
- Tasks 8–10 完成后：在 8 GB 基准机分别验证 EdgeTAM/SAM 2、渲染和导出，确认 GPU 阶段串行释放。
- Task 11 完成后：用 mock 服务跑完整管线并审查缓存失效。
- Task 12 完成后：FastAPI 的认证、任务恢复、上传和关闭语义固定，进行一次安全审查。
- Tasks 13–14 完成后：同一 React SPA 在 fake adapters 下跑通三次交互，再接真实 API。
- Tasks 15–16 完成后：验证 Tauri sidecar 生命周期、最小权限和浏览器/Tauri 双宿主一致性。
- Task 17 完成后：运行完整发布门，只有 8 GB 验收报告、浏览器 E2E 和桌面 smoke 同时达标才判定 MVP 验证完成。

## 计划明确不实施的工作

- ViPE/VGGT 真实后端：保留 `CameraSolver` 契约，待 OpenCV 基线和 8 GB 验收结果出来后单独计划。
- 视频深度、Gaussian depth 视频输出和人物遮挡：只允许场景视口使用单帧深度拾取，不进入合成管线。
- Gaussian 场景重建：作为独立 V3 项目，不加入本计划依赖。
- 面向公众的签名安装器、自动更新和代码签名：MVP 只产出内部 Tauri debug/release bundle 与 onefile API sidecar。
- 在线 Gaussian 资源目录与下载：只保留 `BackendClient`、资源元数据和安全下载器的扩展接口，不在 MVP 接入真实目录。
- 云端、账户、协作、节点图、专业时间线和 DCC 集成。
