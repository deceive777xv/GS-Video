# GS Video

GS Video 将源视频中的主体与相机运动迁移到真实世界 Gaussian Splatting 场景中，并以视觉可信的透视关系完成合成。

## Language

**探索相机（Exploration Camera）**：
只用于自由浏览 GS 和采集场景标定点的完整 6DoF 相机，不具有最终合成 authority。
_Avoid_: 初始机位、目标相机

**源透视校准（Source Perspective Calibration）**：
源锚定帧中由地平线、垂直方向和内参共同定义的地面透视关系。
_Avoid_: 手动 Pitch、手动 Roll

**主体可见性审计（Subject Visibility Audit）**：
对全片主体 Alpha 和代理帧进行的时间范围分析，用于推荐接触约束或透视约束。
_Avoid_: 脚部检测结果

**局部地面锚（Local Ground Anchor）**：
目标 GS 中由 P0、P1、P2 定义的局部平面；P0 是场景 Pivot，并在接触约束下同时作为人物接触点。
_Avoid_: 世界原点、Orbit Pivot

**接触约束（Contact Constraint）**：
脚底可见时，将源锚定帧的已确认脚底像素与目标 P0 绑定的合成约束。
_Avoid_: 虚拟脚底

**透视约束（Perspective Constraint）**：
不声明人物与目标几何真实接触，只保持源画面和目标局部平面的透视方向一致的合成约束。
_Avoid_: 无约束模式

**场景方位角（Scene Azimuth）**：
合成机位体系绕 P0 的局部地面法线旋转的创作参数，不是相机局部 Yaw。
_Avoid_: Yaw、相机旋转

**人物与场景比例（Subject-to-Scene Scale）**：
在不改变已校准透视关系的前提下，控制人物与目标场景视觉比例的统一尺度参数。
_Avoid_: 相机距离、人物缩放

**合成相机架（Synthesis Camera Rig）**：
由源透视、局部地面、场景方位角和人物与场景比例共同约束的目标相机体系。
_Avoid_: OrbitCamera、自由相机

**目标锚定相机（Target Anchor Camera）**：
源锚定帧在目标 GS 中对应的完整相机位姿和内参，是迁移源相对轨迹的基准。
_Avoid_: 探索相机、确认初始机位
