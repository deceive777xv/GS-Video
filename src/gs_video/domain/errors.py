class GsVideoError(RuntimeError):
    code = "system_error"


class RepairableError(GsVideoError):
    code = "repairable"


class UnsupportedMaterialError(GsVideoError):
    code = "unsupported_material"


class CancelledError(GsVideoError):
    code = "cancelled"
