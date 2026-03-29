from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from .config import add_glee_config
from .backbone.swin import D2SwinTransformer
from .backbone.eva02 import D2_EVA02


# Lazy imports: GLEE wrapper and data loaders depend on dataset registration
# which requires specific file paths. Import them only when needed.
def __getattr__(name):
    if name == "GLEE":
        from .GLEE import GLEE
        return GLEE
    if name == "build_detection_train_loader":
        from .data import build_detection_train_loader
        return build_detection_train_loader
    if name == "build_detection_test_loader":
        from .data import build_detection_test_loader
        return build_detection_test_loader
    raise AttributeError(f"module 'glee' has no attribute {name!r}")
