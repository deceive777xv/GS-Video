import type { ProjectDto, StageName } from '../api/types'

export const WORKFLOW_STEPS = ['import', 'subject', 'camera', 'preview', 'export'] as const

export type WorkflowStep = (typeof WORKFLOW_STEPS)[number]

export function stageSucceeded(project: ProjectDto, stage: StageName): boolean {
  return project.stages[stage]?.status === 'succeeded'
}

export function creativeInteractionCount(project: ProjectDto): number {
  const workflow = project.workflow
  return Number(workflow.subject_prompt !== null)
    + Number(
      workflow.confirmed_camera_revision !== null
      && workflow.confirmed_preview_artifact_id !== null,
    )
    + Number(workflow.foot_point !== null)
}

export function workflowStepForProject(project: ProjectDto): WorkflowStep {
  const workflow = project.workflow
  if (workflow.source_summary === null || workflow.scene_summary === null) return 'import'
  if (
    workflow.subject_prompt === null
    || !stageSucceeded(project, 'segment')
    || !stageSucceeded(project, 'solve_camera')
  ) return 'subject'
  if (
    workflow.confirmed_camera_revision === null
    || workflow.confirmed_preview_artifact_id === null
    || workflow.foot_point === null
  ) return 'camera'
  if (!stageSucceeded(project, 'composite')) return 'preview'
  return 'export'
}

export function canVisitStep(project: ProjectDto, step: WorkflowStep): boolean {
  const workflow = project.workflow
  switch (step) {
    case 'import': return true
    case 'subject':
      return workflow.source_summary !== null && workflow.scene_summary !== null
    case 'camera':
      return workflow.subject_prompt !== null
        && stageSucceeded(project, 'segment')
        && stageSucceeded(project, 'solve_camera')
    case 'preview':
      return workflow.confirmed_camera_revision !== null && workflow.foot_point !== null
    case 'export':
      return stageSucceeded(project, 'composite')
  }
}

export function canAdvance(project: ProjectDto, step: WorkflowStep): boolean {
  const index = WORKFLOW_STEPS.indexOf(step)
  const next = WORKFLOW_STEPS[index + 1]
  return next !== undefined && canVisitStep(project, next)
}
