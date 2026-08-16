import type { BackendClient } from '../../api/backend-client'
import type { ProjectDto } from '../../api/types'
import { ConstrainedCameraWorkspace } from './constrained-camera-workspace'

interface CameraPageProps {
  backend: BackendClient
  busy: boolean
  project: ProjectDto
  onError(value: unknown): void
  onProjectChange(project: ProjectDto): void
  onRefresh(): Promise<ProjectDto>
}

export function CameraPage({ backend, busy, project, onError, onProjectChange, onRefresh }: CameraPageProps) {
  const workflow = project.workflow
  const synthesisConfirmed = workflow.synthesis_placement !== null
    && workflow.confirmed_synthesis_placement_revision === workflow.synthesis_placement.revision

  return (
    <section aria-labelledby="camera-title" className="page-grid page-wide">
      <div className="page-heading page-heading-row">
        <div>
          <p className="eyebrow">03 · PERSPECTIVE + LOCAL GROUND</p>
          <h2 id="camera-title">校准透视并放置合成机位</h2>
          <p>自由探索 GS，再用同一冻结视角的三个点定义局部地面；最终机位由源透视约束求解。</p>
        </div>
        <div className="authority-chips">
          <span className={workflow.source_perspective_calibration === null ? 'chip' : 'chip chip-ok'}>源透视 {workflow.source_perspective_calibration === null ? '未校准' : `r${workflow.source_perspective_calibration.revision}`}</span>
          <span className={workflow.local_ground_anchor === null ? 'chip' : 'chip chip-ok'}>局部地面 {workflow.local_ground_anchor === null ? '未标定' : `r${workflow.local_ground_anchor.revision}`}</span>
          <span className={synthesisConfirmed ? 'chip chip-ok' : 'chip'}>合成机位 {synthesisConfirmed ? `r${workflow.synthesis_placement?.revision}` : '未确认'}</span>
        </div>
      </div>
      <ConstrainedCameraWorkspace
        backend={backend}
        busy={busy}
        key={`${project.project_id}:${project.source_video_asset_id ?? project.source_video}:${project.scene_ply_asset_id ?? project.scene_ply}:${project.stages.segment?.cache_key ?? ''}`}
        onError={onError}
        onProjectChange={onProjectChange}
        onRefresh={onRefresh}
        project={project}
      />
      <p className="technical-note">透视一致不等于物理接触正确；纯透视模式不会声明人物与目标几何存在接触或遮挡关系。</p>
    </section>
  )
}
