# GS Video MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个在 Windows 11 与 8 GB NVIDIA 显存环境中运行的内部桌面工具，让用户通过选择人物、确认 Gaussian 场景初始机位和指定落脚点，将 10–30 秒单人单镜头视频自动合成为新的 Gaussian 背景视频。

**Architecture:** 使用单一 Python 3.11 代码库实现 PySide6 桌面向导、版本化项目模型和可恢复的阶段管线。高变动的分割、相机求解和 Gaussian 渲染通过窄接口隔离；GPU 阶段串行运行，并允许在独立 Python 进程中加载模型以可靠释放显存。MVP 相机求解先使用 OpenCV 的受控视频基线，后续可通过同一契约接入 ViPE 或 VGGT。

**Tech Stack:** Python 3.11、PySide6 6、Pydantic 2、NumPy 2、OpenCV 4.13、Pillow、FFmpeg/ffprobe、PyTorch、SAM 2.1 Hiera Tiny、gsplat 1.x、plyfile、pytest、pytest-qt、ruff、mypy。

## Global Constraints

- 运行平台固定为 Windows 11；WSL 只可作为可选 GPU worker 运行方式，桌面应用必须原生运行。
- 基准硬件为 NVIDIA GPU、8 GB 显存；分割、相机求解和渲染不得同时驻留 GPU。
- 源视频限制为 10–30 秒、最高 1920×1080、单镜头、单个主要人物。
- MVP 只接受已重建完成且包含完整 Gaussian 属性的 `.ply` 静态场景。
- 正常案例只要求选择人物、确认初始机位、指定落脚点三次关键交互。
- MVP 只做 RGB Gaussian 背景与人物 Alpha 合成，不做人物深度、遮挡、阴影或重光照。
- 所有原始素材只读；所有产物写入项目目录；运行过程不发起素材上传。
- `project.json` 是版本化状态源；大体积逐帧数据只保存文件引用和内容摘要。
- 每个昂贵阶段必须支持进度、取消、缓存、失败分类和从最近成功阶段重试。
- 代码先写失败测试，再写最小实现；每个任务完成后单独提交。

## 技术决策说明

- PySide6 使用 Qt 官方的主线程 UI + worker signal 模式；后台线程不得直接修改 Qt model 或 widget。
- Pydantic 2 使用 `model_validate_json()` 读取、`model_dump_json()` 写入，并在进入当前模型验证前运行显式字典迁移。
- gsplat 使用 `rasterization(means, quats, scales, opacities, colors, viewmats, Ks, width, height)`；MVP 每次只渲染一个视角以控制显存。
- SAM 2 官方在 Windows 上推荐 WSL，因此分割器通过可配置的外部 worker 命令运行；默认检查 SAM 2.1 Hiera Tiny 以降低显存占用。
- ViPE 能直接输出相机内参、运动和近度量深度，但其 GPU/第三方模型组合尚未在 8 GB Windows 基准机验证，因此不作为首个必须通过的 MVP 后端。`CameraSolver` 契约必须允许后续无 UI 改动地接入 ViPE。
- OpenCV 基线只服务受控输入：固定、纯旋转和轻中度手持。平移由本质矩阵恢复，尺度由目标落脚点和“运动幅度”参数决定。

实施时优先核对以下官方资料，避免复制过时 API：

- [Qt for Python 6 documentation](https://doc.qt.io/qtforpython-6/)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [OpenCV 4.13 documentation](https://docs.opencv.org/4.13.0/)
- [FFmpeg documentation](https://ffmpeg.org/documentation.html)
- [SAM 2 official repository](https://github.com/facebookresearch/sam2)
- [gsplat official repository](https://github.com/nerfstudio-project/gsplat)
- [ViPE official repository](https://github.com/nv-tlabs/vipe)

## 目标文件结构

```text
GS-Video/
├─ pyproject.toml                         # 包元数据、依赖、pytest/ruff/mypy 配置
├─ README.md                              # 开发环境、模型安装、运行与验证命令
├─ src/gs_video/
│  ├─ __init__.py
│  ├─ __main__.py                         # python -m gs_video 入口
│  ├─ app.py                              # QApplication 组装，不含业务规则
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
│  │  ├─ sam2.py                          # 外部 SAM 2 worker 客户端
│  │  └─ worker.py                        # SAM 2.1 视频传播进程入口
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
│  └─ ui/
│     ├─ main_window.py                    # QWizard/QStackedWidget 导航与项目绑定
│     ├─ task_worker.py                    # QThread worker 与 Signal 桥接
│     ├─ viewport.py                       # 场景预览、轨道相机和落脚点交互
│     └─ pages/
│        ├─ import_page.py
│        ├─ subject_page.py
│        ├─ camera_page.py
│        ├─ preview_page.py
│        └─ export_page.py
└─ tests/
   ├─ fixtures/                            # 小型视频、PLY、相机与 Alpha 固定资产
   ├─ unit/
   ├─ integration/
   └─ e2e/
```

---

### Task 1: 建立可测试的应用骨架与环境诊断

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
  "numpy>=2,<3",
  "opencv-python-headless>=4.13,<5",
  "pillow>=11,<12",
  "plyfile>=1.1,<2",
  "pydantic>=2.11,<3",
  "PySide6>=6.9,<7",
]

[project.optional-dependencies]
dev = ["mypy>=1.16,<2", "pytest>=8,<9", "pytest-qt>=4.5,<5", "ruff>=0.12,<1"]

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
    args = parser.parse_args()
    if args.doctor:
        report = EnvironmentDoctor().check()
        print(report.model_dump_json(indent=2) if args.json else report)
        return 0 if report.ready else 2
    from gs_video.app import run
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
```

```python
# src/gs_video/app.py
def run() -> int:
    print("GS Video desktop UI has not been assembled yet. Run with --doctor.")
    return 0
```

Run: `python -m pytest tests/unit/environment/test_doctor.py -v`

Expected: PASS。

Run: `python -m ruff check src tests`

Expected: `All checks passed!`

- [ ] **Step 5: 提交**

```powershell
git add pyproject.toml src/gs_video tests/unit/environment
git commit -m "feat: scaffold application and environment doctor"
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

### Task 4: 实现视频探测、输入限制与代理帧生成

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

### Task 5: 校验 Gaussian PLY、估算显存并建立场景相机数学

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

### Task 6: 实现受控输入的 OpenCV 相机求解与轨迹映射

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

### Task 7: 实现前景分割契约与 SAM 2 外部 worker

**Files:**
- Modify: `src/gs_video/domain/contracts.py`
- Modify: `src/gs_video/environment/doctor.py`
- Create: `src/gs_video/segmentation/sam2.py`
- Create: `src/gs_video/segmentation/worker.py`
- Create: `tests/unit/segmentation/test_sam2_client.py`
- Create: `tests/integration/segmentation/test_worker_protocol.py`

**Interfaces:**
- Produces: `ForegroundSegmenter.segment(frames, prompt, output_dir, emit, token) -> MaskSequence`
- Produces: newline-delimited JSON worker events: `progress`, `result`, `error`
- Consumes: SAM 2.1 checkpoint/config paths and a configurable worker command prefix

- [ ] **Step 1: 写 worker 命令、进度解析和取消测试**

```python
def test_client_parses_progress_and_result(tmp_path: Path) -> None:
    lines = [
        '{"type":"progress","current":1,"total":2}\n',
        '{"type":"result","mask_dir":"masks","frames":2}\n',
    ]
    client = Sam2Segmenter(process_factory=fake_process(lines))
    result = client.segment([tmp_path / "000001.jpg"], Prompt(frame_index=0, x=10, y=20), tmp_path / "masks", lambda event: None, CancellationToken())
    assert result.frame_count == 2
    assert result.mask_dir == tmp_path / "masks"
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/unit/segmentation tests/integration/segmentation -v`

Expected: FAIL，缺少 `Sam2Segmenter`。

- [ ] **Step 3: 实现不经 shell 的外部 worker 客户端**

客户端命令格式固定为：

```text
<worker-prefix> -m gs_video.segmentation.worker
  --frames <proxy-dir>
  --output <mask-dir>
  --frame-index <int>
  --point <x>,<y>
  --config <sam2.1_hiera_t.yaml>
  --checkpoint <sam2.1_hiera_tiny.pt>
```

`worker-prefix` 默认为当前 Python 可执行文件，也可配置为 `wsl.exe -d Ubuntu -- <venv-python>`。取消时先发送 `terminate()`，5 秒未退出再 `kill()`；无论何种退出都读取 stderr 并写入项目日志。

```python
def _start_worker(self, request: SegmentRequest) -> subprocess.Popen[str]:
    command = [
        *self.worker_prefix, "-m", "gs_video.segmentation.worker",
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

- [ ] **Step 4: 实现 SAM 2.1 worker 的单点传播**

Worker 使用官方 `build_sam2_video_predictor`，初始化视频状态后在代表帧加入正点提示，向前传播所有帧，再从代表帧向后传播缺失帧。每帧写 8-bit PNG Alpha，文件名与代理帧编号一致。推理包裹在 `torch.inference_mode()` 与适合 GPU 的 autocast 中；进程退出前删除 predictor 并调用 `torch.cuda.empty_cache()`。

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

Worker 完成后计算每帧非零 Alpha 比例；连续 15 帧低于 `0.001` 时返回 `unsupported_material`，信息为“主要人物长时间不可见”。环境诊断在启动分割前检查 worker 命令、config 和 checkpoint 均存在且可读取。

Run: `python -m pytest tests/unit/segmentation tests/integration/segmentation -v -m "not gpu"`

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add src/gs_video/domain/contracts.py src/gs_video/segmentation tests/unit/segmentation tests/integration/segmentation
git commit -m "feat: add isolated SAM2 video segmentation"
```

### Task 8: 实现 gsplat 串行渲染器和预览降采样

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

### Task 9: 实现 Alpha 合成、音轨恢复和帧精确导出

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

### Task 10: 组装完整工作流、缓存依赖和低分辨率预览

**Files:**
- Modify: `src/gs_video/pipeline/workflow.py`
- Modify: `src/gs_video/pipeline/runner.py`
- Create: `tests/integration/pipeline/test_workflow.py`
- Create: `tests/integration/pipeline/test_invalidation.py`

**Interfaces:**
- Produces: `build_mvp_workflow(services) -> PipelineRunner`
- Produces: `WorkflowServices` 显式依赖容器
- Consumes: Tasks 4–9 的全部稳定接口

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

### Task 11: 建立 PySide6 向导、后台任务桥和人物选择交互

**Files:**
- Modify: `src/gs_video/app.py`
- Create: `src/gs_video/ui/main_window.py`
- Create: `src/gs_video/ui/task_worker.py`
- Create: `src/gs_video/ui/pages/import_page.py`
- Create: `src/gs_video/ui/pages/subject_page.py`
- Create: `tests/unit/ui/test_main_window.py`
- Create: `tests/unit/ui/test_task_worker.py`

**Interfaces:**
- Produces: `run() -> int`, `MainWindow`, `PipelineTaskWorker`
- Consumes: `ProjectRepository`, `EnvironmentDoctor`, `PipelineRunner`

- [ ] **Step 1: 写页面门控和主线程信号测试**

```python
def test_import_page_blocks_next_until_both_inputs_validate(qtbot, window) -> None:
    qtbot.addWidget(window)
    window.import_page.set_video(valid_video_path())
    assert window.next_button.isEnabled() is False
    window.import_page.set_scene(valid_scene_path())
    assert window.next_button.isEnabled() is True


def test_worker_emits_progress_without_touching_widgets(qtbot) -> None:
    worker = PipelineTaskWorker(lambda emit, token: emit(3, 10, "分割"))
    with qtbot.waitSignal(worker.progress, timeout=1000) as signal:
        worker.start()
    assert signal.args == [3, 10, "分割"]
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/unit/ui -v`

Expected: FAIL，UI 模块不存在。

- [ ] **Step 3: 实现 QStackedWidget 向导骨架**

窗口固定包含左侧步骤列表、中央页面、底部“上一步/下一步/取消任务”。页面完成状态来自项目模型而非 widget 临时状态。导入页显示媒体摘要、场景 Gaussian 数量、显存预估和可修复错误。

```python
class MainWindow(QMainWindow):
    def __init__(self, controller: ProjectController) -> None:
        super().__init__()
        self.controller = controller
        self.stack = QStackedWidget()
        self.pages = [ImportPage(controller), SubjectPage(controller)]
        for page in self.pages:
            self.stack.addWidget(page)
            page.completionChanged.connect(self._refresh_navigation)
        self.setCentralWidget(self._build_shell())

    @Slot()
    def _refresh_navigation(self) -> None:
        page = self.pages[self.stack.currentIndex()]
        self.next_button.setEnabled(page.isComplete())
```

- [ ] **Step 4: 实现 QThread worker 与人物点击页**

`PipelineTaskWorker` 继承 `QThread`，只通过 `Signal(int, int, str)`、`Signal(object)` 和 `Signal(str, str)` 返回进度、结果和错误；所有 widget 更新在主线程 slot 中执行。人物页显示代理帧，点击位置转换到原代理图像像素坐标，保存 `Prompt(frame_index, x, y)` 后运行分割并显示 Alpha 叠加。

```python
class PipelineTaskWorker(QThread):
    progress = Signal(int, int, str)
    succeeded = Signal(object)
    failed = Signal(str, str)

    def __init__(self, operation: Callable[[ProgressEmitter, CancellationToken], object]) -> None:
        super().__init__()
        self.operation = operation
        self.token = CancellationToken()

    def run(self) -> None:
        try:
            result = self.operation(lambda a, b, c: self.progress.emit(a, b, c), self.token)
            self.succeeded.emit(result)
        except GsVideoError as error:
            self.failed.emit(error.code, str(error))

    def cancel(self) -> None:
        self.token.cancel()
```

Run: `python -m pytest tests/unit/ui -v`

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add src/gs_video/app.py src/gs_video/ui tests/unit/ui
git commit -m "feat: add guided desktop shell and subject selection"
```

### Task 12: 完成场景视口、落脚点、预览、导出与错误恢复 UI

**Files:**
- Create: `src/gs_video/ui/viewport.py`
- Create: `src/gs_video/ui/pages/camera_page.py`
- Create: `src/gs_video/ui/pages/preview_page.py`
- Create: `src/gs_video/ui/pages/export_page.py`
- Modify: `src/gs_video/ui/main_window.py`
- Create: `tests/unit/ui/test_viewport.py`
- Create: `tests/e2e/test_guided_workflow.py`

**Interfaces:**
- Produces: `SceneViewport.cameraChanged`, `SceneViewport.anchorSelected`
- Produces: 完整三次交互后的预览与导出向导

- [ ] **Step 1: 写轨道相机交互、落脚点和 mock E2E 测试**

```python
def test_viewport_click_emits_world_anchor(qtbot, viewport) -> None:
    with qtbot.waitSignal(viewport.anchorSelected) as signal:
        qtbot.mouseClick(viewport, Qt.MouseButton.LeftButton, pos=QPoint(320, 180))
    assert len(signal.args[0]) == 3


def test_guided_workflow_exports_after_three_interactions(qtbot, app_harness) -> None:
    app_harness.import_assets()
    app_harness.click_subject(100, 120)
    app_harness.confirm_camera()
    app_harness.click_anchor(320, 180)
    app_harness.generate_and_export()
    assert app_harness.output.exists()
    assert app_harness.interaction_count == 3
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/unit/ui/test_viewport.py tests/e2e/test_guided_workflow.py -v`

Expected: FAIL，缺少 viewport 和后续页面。

- [ ] **Step 3: 实现延迟渲染场景视口**

视口显示最近一次 gsplat 预览帧；鼠标拖动更新 yaw/pitch，滚轮更新 distance，焦距使用垂直 FOV slider。交互期间以 150 ms debounce 请求 540p 单帧渲染，旧请求用 generation id 丢弃。用户点击“确认初始机位”才计为第二次关键交互。

落脚点通过当前相机射线与用户选择的场景深度相交。MVP renderer 可额外为单帧交互请求 `render_mode="RGB+ED"`，但最终视频仍只保存 RGB；深度只用于把屏幕点击还原到目标场景坐标，不参与人物遮挡。

```python
def unproject_anchor(x: int, y: int, depth: float, K: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray:
    pixel = np.array([x, y, 1.0], dtype=np.float64)
    camera_point = np.linalg.inv(K) @ pixel * depth
    world = camera_to_world @ np.array([*camera_point, 1.0])
    return world[:3] / world[3]


def accept_pick(self, x: int, y: int) -> None:
    depth = float(self.pick_buffer.expected_depth[y, x])
    if not np.isfinite(depth) or depth <= 0:
        self.pickRejected.emit("该位置没有可用的场景深度")
        return
    self.anchorSelected.emit(unproject_anchor(x, y, depth, self.K, self.camera_to_world))
```

- [ ] **Step 4: 实现预览、导出、取消和重试页面**

预览页显示低分辨率合成视频、焦距和 `motion_scale`；修改后只失效映射及下游阶段。进度页展示阶段、当前帧、总帧数、耗时和 ETA。失败页依据错误分类显示操作：重新选择人物、降低分辨率、返回机位页或重试当前阶段。导出页只在 ffprobe 后验校验通过后显示成功。

```python
@Slot(float)
def _motion_scale_changed(self, value: float) -> None:
    self.controller.update_motion_scale(value)
    self.controller.invalidate(ChangeKind.MOTION_SCALE)


@Slot(str, str)
def _show_failure(self, code: str, message: str) -> None:
    actions = {
        "repairable": self._show_repair_action,
        "unsupported_material": self._show_material_rejection,
        "system_error": self._show_log_action,
    }
    actions.get(code, self._show_log_action)(message)
```

Run: `python -m pytest tests/unit/ui tests/e2e/test_guided_workflow.py -v -m "not gpu"`

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add src/gs_video/ui tests/unit/ui tests/e2e
git commit -m "feat: complete guided preview and export workflow"
```

### Task 13: 建立 8 GB 验收工具、文档和完整发布门

**Files:**
- Create: `scripts/run_acceptance.py`
- Create: `tests/acceptance/cases.json`
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

`cases.json` 每项记录视频、场景、人物提示、目标相机、落脚点、运动幅度和预期镜头类型。报告器记录：是否导出、是否人工修改中间文件、失败分类、每阶段耗时、每阶段峰值显存、输出帧数、音视频时长误差。缺失素材时标记 `fixture_missing` 并使发布门失败，不得跳过。

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

- [ ] **Step 4: 写开发与模型安装文档**

README 必须包含 Python 3.11 venv、`pip install -e ".[dev]"`、FFmpeg PATH、CUDA/PyTorch 单独安装、SAM 2.1 Tiny checkpoint、gsplat 安装、`python -m gs_video --doctor --json` 和测试命令。模型文档明确记录 SAM 2 官方 Windows/WSL建议、第三方模型许可证检查和离线 checkpoint 路径。

```markdown
## Local development

1. Create a Python 3.11 virtual environment.
2. Run `python -m pip install -e ".[dev]"`.
3. Install the CUDA-enabled PyTorch build that matches the local driver.
4. Install gsplat and SAM 2 in the configured GPU worker environment.
5. Add `ffmpeg` and `ffprobe` to `PATH`.
6. Run `python -m gs_video --doctor --json` before opening the desktop app.
```

- [ ] **Step 5: 运行完整发布门**

Run: `python -m ruff check src tests scripts`

Expected: `All checks passed!`

Run: `python -m mypy src/gs_video`

Expected: `Success: no issues found`。

Run: `python -m pytest tests/unit tests/integration tests/e2e -v -m "not gpu"`

Expected: 全部 PASS。

Run on 8 GB NVIDIA acceptance machine: `python -m pytest tests/integration/scene/test_gsplat_smoke.py -v -m gpu`

Expected: PASS。

Run on 8 GB NVIDIA acceptance machine: `python scripts/run_acceptance.py --cases tests/acceptance/cases.json --report acceptance-report.json`

Expected: 12 个案例全部执行；自动闭环成功率至少 80%；规定案例峰值显存不超过 8 GB；成功案例音视频时长误差不超过一帧。

- [ ] **Step 6: 提交**

```powershell
git add scripts tests/acceptance README.md docs/development
git commit -m "test: add MVP acceptance and release gate"
```

## 实施顺序与检查点

- Tasks 1–3 完成后：得到可测试的项目状态与阶段框架，进行一次架构审查。
- Tasks 4–6 完成后：用固定视频验证媒体、PLY 和相机轨迹，不加载 SAM 2 或 gsplat。
- Tasks 7–9 完成后：在 8 GB 基准机分别验证分割、渲染和导出，确认 GPU 阶段串行释放。
- Task 10 完成后：用 mock 服务跑完整管线并审查缓存失效。
- Tasks 11–12 完成后：用 mock 后端验证三次交互的桌面闭环，再接真实后端。
- Task 13 完成后：运行完整发布门，只有验收报告达到 PRD 指标才判定 MVP 验证完成。

## 计划明确不实施的工作

- ViPE/VGGT 真实后端：保留 `CameraSolver` 契约，待 OpenCV 基线和 8 GB 验收结果出来后单独计划。
- 视频深度、Gaussian depth 视频输出和人物遮挡：只允许场景视口使用单帧深度拾取，不进入合成管线。
- Gaussian 场景重建：作为独立 V3 项目，不加入本计划依赖。
- 安装器或单文件 EXE：内部 MVP 先使用 Python 3.11 虚拟环境运行。
- 云端、账户、协作、节点图、专业时间线和 DCC 集成。
