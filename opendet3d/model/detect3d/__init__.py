# Lazy imports: SAM3_3D needs sam3 package, GLEE_3D needs glee package,
# GroundingDINO3D needs vis4d_cuda_ops. Only load when requested.
def __getattr__(name):
    if name in ("SAM3_3D", "SAM3_3DOut", "SAM3_3DBatchedInputs", "build_sam3_3d"):
        from . import sam3_3d
        return getattr(sam3_3d, name)
    if name in ("GLEE_3D", "GLEE_3DOut", "GLEE_3DBatchedInputs"):
        from . import glee_3d
        return getattr(glee_3d, name)
    if name == "GroundingDINO3D":
        from .grounding_dino_3d import GroundingDINO3D
        return GroundingDINO3D
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "GroundingDINO3D",
    "SAM3_3D",
    "SAM3_3DOut",
    "SAM3_3DBatchedInputs",
    "build_sam3_3d",
    "GLEE_3D",
    "GLEE_3DOut",
    "GLEE_3DBatchedInputs",
]
