import type { BackendClient } from '../../api/backend-client'
import type { ProjectDto } from '../../api/types'
import { ConstrainedCameraWorkspace, type CameraPanel } from './constrained-camera-workspace'

interface CameraPageProps {
  backend: BackendClient
  busy: boolean
  project: ProjectDto
  onError(value: unknown): void
  onProjectChange(project: ProjectDto): void
  onRefresh(): Promise<ProjectDto>
  panel: CameraPanel | undefined
}

export function CameraPage({ backend, busy, project, onError, onProjectChange, onRefresh, panel }: CameraPageProps) {
  const workflow = project.workflow
  const ground = workflow.target_ground

  return (
    <section aria-labelledby="camera-title" className="page-grid page-wide camera-workbench-page">
      <div className="page-heading page-heading-row">
        <div>
          <p className="eyebrow">03 · VIPE TRAJECTORY + GS GROUND</p>
          <h2 id="camera-title">对齐自动相机轨迹与 GS 地面</h2>
          <p>ViPE 已从完整视频联合解算逐帧内参、位姿、深度和源地面；这里只需在 GS 中提示并确认目标地面。</p>
        </div>
        <div className="authority-chips">
          <span className={project.stages.solve_camera?.status === 'succeeded' ? 'chip chip-ok' : 'chip'}>ViPE {project.stages.solve_camera?.status === 'succeeded' ? '已解算' : '待解算'}</span>
          <span className={ground?.confirmed === true ? 'chip chip-ok' : 'chip'}>目标地面 {ground === null ? '未拟合' : ground.confirmed ? `r${ground.revision}` : '待确认'}</span>
        </div>
      </div>
      <ConstrainedCameraWorkspace
        backend={backend}
        busy={busy}
        key={`${project.project_id}:${project.source_video_asset_id ?? project.source_video}:${project.scene_ply_asset_id ?? project.scene_ply}:${project.stages.segment?.cache_key ?? ''}`}
        onError={onError}
        onProjectChange={onProjectChange}
        onRefresh={onRefresh}
        panel={panel ?? 'scene'}
        project={project}
      />
      <p className="technical-note">人物是与相机像素坐标锁定的 RGBA 层；任何对人物进行三维平移、旋转或缩放的操作都不属于本工作流。</p>
    </section>
  )
}
