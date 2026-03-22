from .sam3_3d import SAM3_3D, SAM3_3DOut, SAM3_3DBatchedInputs, build_sam3_3d

# Lazy import: GroundingDINO3D pulls in vis4d_cuda_ops which may not be available
def __getattr__(name):
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
]
