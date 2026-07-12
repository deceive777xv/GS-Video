from gs_video.domain.models import Project, StageName, StageStatus


DEPENDENCIES: dict[StageName, tuple[StageName, ...]] = {
    StageName.INGEST: (),
    StageName.SEGMENT: (StageName.INGEST,),
    StageName.SOLVE_CAMERA: (StageName.INGEST,),
    StageName.MAP_TRAJECTORY: (StageName.SOLVE_CAMERA,),
    StageName.RENDER: (StageName.MAP_TRAJECTORY,),
    StageName.COMPOSITE: (StageName.SEGMENT, StageName.RENDER),
    StageName.EXPORT: (StageName.COMPOSITE,),
}


def invalidate_from(project: Project, changed_stage: StageName) -> Project:
    invalidated = {changed_stage}
    pending = [changed_stage]

    while pending:
        dependency = pending.pop()
        for stage, dependencies in DEPENDENCIES.items():
            if dependency in dependencies and stage not in invalidated:
                invalidated.add(stage)
                pending.append(stage)

    for stage in invalidated:
        state = project.stages.get(stage)
        if state is not None:
            state.status = StageStatus.STALE
            state.cache_key = None
            state.error_code = None

    return project
