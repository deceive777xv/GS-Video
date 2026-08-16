import type { ProjectDto, StageName } from '../api/types'

export const WORKFLOW_STEPS = ['import', 'subject', 'camera', 'preview', 'export'] as const

export type WorkflowStep = (typeof WORKFLOW_STEPS)[number]

export function stageSucceeded(project: ProjectDto, stage: StageName): boolean {
  return project.stages[stage]?.status === 'succeeded'
}

export function creativeInteractionCount(project: ProjectDto): number {
  const workflow = project.workflow
  return Number(workflow.subject_prompt !== null)
    + Number(workflow.source_perspective_calibration !== null)
    + Number(workflow.local_ground_anchor !== null)
    + Number(
      workflow.synthesis_placement !== null
      && workflow.confirmed_synthesis_placement_revision
        === workflow.synthesis_placement.revision,
    )
}

export function workflowStepForProject(project: ProjectDto): WorkflowStep {
  const workflow = project.workflow
  if (
    workflow.source_summary === null
    || workflow.scene_summary === null
    || !stageSucceeded(project, 'ingest')
  ) return 'import'
  if (
    workflow.subject_prompt === null
    || !stageSucceeded(project, 'segment')
    || !stageSucceeded(project, 'solve_camera')
  ) return 'subject'
  if (
    workflow.synthesis_placement === null
    || workflow.confirmed_synthesis_placement_revision
      !== workflow.synthesis_placement.revision
  ) return 'camera'
  if (!stageSucceeded(project, 'composite')) return 'preview'
  return 'export'
}

export function canVisitStep(project: ProjectDto, step: WorkflowStep): boolean {
  const workflow = project.workflow
  switch (step) {
    case 'import': return true
    case 'subject':
      return workflow.source_summary !== null
        && workflow.scene_summary !== null
        && stageSucceeded(project, 'ingest')
    case 'camera':
      return workflow.subject_prompt !== null
        && stageSucceeded(project, 'segment')
        && stageSucceeded(project, 'solve_camera')
    case 'preview':
      return workflow.synthesis_placement !== null
        && workflow.confirmed_synthesis_placement_revision
          === workflow.synthesis_placement.revision
    case 'export':
      return stageSucceeded(project, 'composite')
  }
}

export function canAdvance(project: ProjectDto, step: WorkflowStep): boolean {
  const index = WORKFLOW_STEPS.indexOf(step)
  const next = WORKFLOW_STEPS[index + 1]
  return next !== undefined && canVisitStep(project, next)
}
