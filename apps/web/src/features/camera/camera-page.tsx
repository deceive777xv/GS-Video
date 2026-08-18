import { useEffect } from 'react'

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
  const synthesisConfirmed = workflow.synthesis_placement !== null
    && workflow.confirmed_synthesis_placement_revision === workflow.synthesis_placement.revision
  const activePanel: CameraPanel = panel ?? (workflow.source_perspective_calibration === null
    ? 'source'
    : workflow.local_ground_anchor === null ? 'scene' : 'synthesis')
  useEffect(() => {
    if (panel !== undefined) return
    window.location.hash = `#/projects/${encodeURIComponent(project.project_id)}/workflow/camera/${activePanel}`
  }, [activePanel, panel, project.project_id])
  const panelMeta: Record<CameraPanel, { label: string; detail: string; complete: boolean }> = {
    source: { label: '1 · 源透视', detail: '线段证据与可见性', complete: workflow.source_perspective_calibration !== null },
    scene: { label: '2 · GS 场景', detail: '6DoF 与三点局部地面', complete: workflow.local_ground_anchor !== null },
    synthesis: { label: '3 · 合成机位', detail: '受约束放置与微调', complete: synthesisConfirmed },
  }

  return (
    <section aria-labelledby="camera-title" className="page-grid page-wide camera-workbench-page">
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
      <nav aria-label="机位子页面" className="camera-subnav">
        {(Object.keys(panelMeta) as CameraPanel[]).map((item) => <a aria-current={activePanel === item ? 'step' : undefined} className={`${activePanel === item ? 'is-current' : ''} ${panelMeta[item].complete ? 'is-complete' : ''}`} href={`#/projects/${encodeURIComponent(project.project_id)}/workflow/camera/${item}`} key={item}><strong>{panelMeta[item].label}</strong><small>{panelMeta[item].detail}</small></a>)}
      </nav>
      <ConstrainedCameraWorkspace
        backend={backend}
        busy={busy}
        key={`${project.project_id}:${project.source_video_asset_id ?? project.source_video}:${project.scene_ply_asset_id ?? project.scene_ply}:${project.stages.segment?.cache_key ?? ''}`}
        onError={onError}
        onProjectChange={onProjectChange}
        onRefresh={onRefresh}
        panel={activePanel}
        project={project}
      />
      <p className="technical-note">透视一致不等于物理接触正确；纯透视模式不会声明人物与目标几何存在接触或遮挡关系。</p>
    </section>
  )
}
