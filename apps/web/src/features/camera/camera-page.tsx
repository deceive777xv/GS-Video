import { useCallback } from 'react'

import type { BackendClient } from '../../api/backend-client'
import type { CameraInput, PreviewFrameDto, ProjectDto } from '../../api/types'
import { SceneViewport } from './scene-viewport'

const DEFAULT_CAMERA: CameraInput = {
  target: [0, 0, 0],
  distance: 4,
  yaw: 0,
  pitch: 0,
  fov_y_degrees: 50,
}

interface CameraPageProps {
  backend: BackendClient
  project: ProjectDto
  onError(value: unknown): void
  onProjectChange(project: ProjectDto): void
  onRefresh(): Promise<ProjectDto>
}

export function CameraPage({ backend, project, onError, onProjectChange, onRefresh }: CameraPageProps) {
  const workflow = project.workflow
  const camera: CameraInput = workflow.target_camera ?? DEFAULT_CAMERA
  const onPreview = useCallback((_frame: PreviewFrameDto, _camera: CameraInput) => {
    void onRefresh().catch(onError)
  }, [onError, onRefresh])
  const refreshProject = useCallback(async () => {
    try {
      await onRefresh()
    } catch (error) {
      onError(error)
    }
  }, [onError, onRefresh])
  const footPoint = useCallback(() => { void refreshProject() }, [refreshProject])

  return (
    <section aria-labelledby="camera-title" className="page-grid page-wide">
      <div className="page-heading page-heading-row">
        <div>
          <p className="eyebrow">03 · CAMERA + ANCHOR</p>
          <h2 id="camera-title">放置目标镜头</h2>
          <p>拖动旋转，滚轮调整距离；确认机位后，在同一后端帧上选择人物落脚点。</p>
        </div>
        <div className="authority-chips">
          <span className={workflow.confirmed_camera_revision === null ? 'chip' : 'chip chip-ok'}>机位 {workflow.confirmed_camera_revision === null ? '未确认' : `r${workflow.confirmed_camera_revision}`}</span>
          <span className={workflow.foot_point === null ? 'chip' : 'chip chip-ok'}>落脚点 {workflow.foot_point === null ? '未设置' : '已验证'}</span>
        </div>
      </div>
      <SceneViewport
        backend={backend}
        camera={camera}
        confirmedCameraRevision={workflow.confirmed_camera_revision}
        confirmedPreviewArtifactId={workflow.confirmed_preview_artifact_id}
        initialFootPoint={workflow.foot_point}
        initialPreview={workflow.preview}
        onAuthorityStale={refreshProject}
        onError={onError}
        onFootPoint={footPoint}
        onPreview={onPreview}
        onProjectChange={onProjectChange}
      />
      <p className="technical-note">深度仅用于这一帧的反投影拾取，不参与 MVP 最终人物遮挡。</p>
    </section>
  )
}
