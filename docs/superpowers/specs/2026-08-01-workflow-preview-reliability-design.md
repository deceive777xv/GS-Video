# 工作流媒体与预览可靠性修复设计

日期：2026-08-01

## 背景

当前导入、人物选择和机位设置之间存在多处异步状态竞态。用户在导入阶段刚显示 100% 时进入人物页，可能先于 ingest 阶段权威状态和代表帧产物就绪；返回再进入后才会重新请求。机位页的预览 generation 由会随页面卸载的组件持有，往返导航会复用仍在后端执行的 generation，并触发 `preview_generation_conflict`。后端恢复的相机对象还包含只读 `revision` 字段，直接回传给严格请求模型会触发 `invalid_request`。

此外，FOV 滑动条的函数式状态更新器延后读取 React 事件对象，连续交互时可能访问已失效的 `currentTarget`，造成未捕获异常和整页空白。竖屏代表帧没有受可用视口高度约束，底部任务时间又显示完整浮点数，分别造成页面溢出和进度条横向跳动。

## 目标

- 未取得 ingest 权威成功状态时，不允许从导入页进入人物页。
- 人物页对暂时未就绪的代表帧进行等待和自动重试，不向用户暴露预期内竞态错误。
- 页面往返、快速调节相机和并发中的旧请求不会复用 preview generation。
- 预览请求严格符合 API 契约，不携带响应专用字段。
- FOV 连续交互不会导致 React 崩溃或应用黑屏。
- 横屏和竖屏代表帧都完整包含在可用载入区域内，不引起页面级纵向溢出。
- 时间显示精确到毫秒、宽度稳定，进度条不随浮点文本变化移动。

## 非目标

- 不修改 Gaussian 渲染算法、质量或 540p 预览上限。
- 不放宽后端对相同 generation 冲突、旧 generation 或额外请求字段的校验。
- 不重新设计工作流页面的配色、导航结构或创作交互计数。
- 不把本次修复扩展为通用数据请求框架。

## 方案选择

采用前端状态生命周期修复，并保留后端严格契约。

未采用由后端重新分配 generation 的方案，因为这会扩大 API、类型和集成契约改动；也不采用容忍重复 generation 或额外字段的方案，因为这会削弱防止旧预览覆盖新预览的保护。

## 详细设计

### 1. 人物页准入与代表帧加载

`canVisitStep(project, 'subject')` 除要求 source 和 scene summary 外，还必须要求 ingest 阶段状态为 `succeeded`。因此底部“下一步”和左侧人物步骤会在 ingest 真正提交权威成功状态后才可用。

`SubjectPage` 的 proxy 加载不再只在组件首次挂载时执行。它以 ingest 阶段状态和代表帧权威身份为依赖：

- ingest 未成功时保持“载入代表帧…”状态，不发送媒体请求；
- ingest 成功后立即请求 descriptor 和 artifact；
- `subject_media_not_ready` 视为短暂状态，使用有上限的短间隔重试，并在项目身份变化或组件卸载时取消；
- 其他错误仍通过现有错误横幅报告；
- 成功后继续执行现有 object URL 所有权和释放逻辑。

入口门控是主要保证，页面内等待/重试用于处理 REST、事件流和磁盘产物之间的窄竞态窗口。

### 2. Preview generation 生命周期

generation 必须比 `SceneViewport` 生命周期更长。应用层维护一个单调递增的 preview generation 分配器，并把取号能力传入机位页面和视口。分配器取以下值中的最大值再递增：

- 本次应用会话已经发出的最大 generation；
- 项目当前 preview 的 generation；
- 当前时间形成的安全整数下界。

时间下界确保桌面页面重载后，新会话也不会与仍在后端收尾的旧请求复用小编号；会话内计数器确保同一毫秒的多次请求仍严格递增。generation 保持 JavaScript 安全整数并满足后端 `ge=1` 契约。

组件卸载仍取消本地等待和 artifact 下载，并以 authority 标识忽略旧响应。后端对相同 generation 不同 fingerprint 的 409 校验保持不变。

### 3. 相机请求序列化边界

增加一个纯函数，将 `CameraInput | CameraDto` 显式投影为请求用 `CameraInput`：只复制 `target`、`distance`、`yaw`、`pitch` 和 `fov_y_degrees`。所有 `renderPreview` 调用在提交前都经过该投影。

这样 TypeScript 的结构兼容不会再让运行时对象中的 `revision` 泄漏到 Pydantic `StrictModel`。相机 fingerprint 也继续只使用这五个可编辑字段。

### 4. FOV 交互安全

滑动条 `onChange` 在事件处理器同步阶段读取并校验数值，再把普通 number 传入函数式状态更新器。更新器不捕获 React 事件对象。

数值继续限制在 20–100 度；键盘 Home、End 和方向键行为保持不变。每次有效变化仍使用现有 150 ms debounce 触发预览。

### 5. 竖屏代表帧布局

人物代表帧容器使用明确的、受当前窗口可用高度约束的 block size，同时保留合理最小高度。图像与 Alpha 叠加层都填充同一容器，并使用 `object-fit: contain` 居中显示。

容器尺寸不再由 480×852 等竖屏图像的固有比例反向撑高。点击坐标仍通过 `toImagePoint` 按 contain 后的实际图像区域换算，黑边点击继续被拒绝。

窄屏断点沿用单列布局，并使用更小的受限高度，避免与 sticky footer 共同造成双重滚动。

### 6. 时间格式与底栏稳定性

任务 elapsed 和 ETA 使用单一格式化函数：有限、非负秒数显示为固定三位小数，例如 `43.004 秒`。ETA 为 `null` 时继续隐藏。

时间区域使用等宽数字，并为 elapsed、ETA 预留稳定的最小宽度。进度条保持固定 flex basis，不再由时间字符串的长度推动。进度百分比和后端原始精度不变，本次只调整显示层。

## 错误处理

- `subject_media_not_ready` 在 ingest 已成功后的有限重试窗口内不显示为用户错误；超过窗口仍报告，以免永久隐藏真实故障。
- `preview_generation_conflict` 不做自动降级或相同 generation 重试。正确行为是前端分配新 generation，从源头避免冲突。
- `invalid_request` 仍由全局错误横幅呈现；回归测试保证相机交互不再产生该错误。
- 被 AbortController 取消的请求不显示错误，也不得更新已卸载组件。
- 未捕获渲染错误不作为本次常规恢复路径；FOV 事件生命周期问题在触发源修正并由测试锁定。

## 测试与验证

### 前端回归测试

- summary 已存在但 ingest 为 running 时，“下一步”和人物导航不可用；ingest 变为 succeeded 后自动可用。
- 人物页遇到一次 `subject_media_not_ready` 后自动重试并显示代表帧，不需要返回重进。
- 竖屏 descriptor 使用 contain 布局，容器具有受限高度，坐标换算在图像内容内外均正确。
- 机位页在第一次预览未完成时卸载并重新挂载，新请求 generation 严格更大，不触发相同编号冲突。
- 从包含 `revision` 的 `target_camera` 恢复后调节 FOV，`renderPreview` 请求只包含五个可编辑字段。
- 连续 FOV change 在事件处理结束后仍可完成状态更新，不读取失效事件对象。
- elapsed 和 ETA 固定显示三位小数；时间区域和进度条使用稳定布局类。

### 后端契约回归

现有以下保护必须继续通过：相同 generation 相同 fingerprint 合并、相同 generation 不同 fingerprint 拒绝、旧 generation 拒绝、不同 preview epoch 相互隔离、严格请求拒绝额外字段。

### 完整验证

依次执行前端定向测试、前端全量测试、TypeScript 类型检查、生产构建，以及非 GPU Python 测试中与 API workflow 相关的定向集合。最后用真实浏览器按以下流程验证：竖屏视频导入完成后直接进入人物页，等待代表帧；进入机位页后立即返回再进入；连续拖动 FOV；确认页面无错误横幅、无黑屏、无页面级媒体溢出，且底栏进度条位置稳定。

## 完成标准

五个用户报告的问题都有可在修复前失败、修复后通过的自动化回归信号；严格 API 契约与预览防陈旧保护保持有效；真实浏览器完成竖屏与往返导航流程且无控制台未捕获异常。
