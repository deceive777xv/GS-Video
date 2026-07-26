class GsVideoError(RuntimeError):
    code = "system_error"
    category = "system"
    retryable = True


class RepairableError(GsVideoError):
    code = "repairable"
    category = "input"
    retryable = True


class UnsupportedMaterialError(GsVideoError):
    code = "unsupported_material"
    category = "subject"
    retryable = False


class CancelledError(GsVideoError):
    code = "cancelled"
    category = "task"
    retryable = False
