"""SAM3_3D data connector configuration."""

from ml_collections import ConfigDict
from vis4d.config import class_config
from vis4d.data.const import CommonKeys as K
from vis4d.engine.connectors import DataConnector, data_key, pred_key


# ============================================================================
# SAM3_3D Specific Connectors
# ============================================================================

# Training connector for SAM3_3D
# Note: SAM3 uses geometric prompts (boxes/points) instead of text
CONN_SAM3_3D_TRAIN = {
    "images": K.images,
    "input_hw": K.input_hw,
    # Geometric prompts (boxes as prompts)
    "prompt_boxes": K.boxes2d,  # Use GT boxes as prompts during training
    "prompt_box_labels": K.boxes2d_classes,
    # Targets
    "boxes2d": K.boxes2d,
    "boxes2d_classes": K.boxes2d_classes,
    "boxes3d": K.boxes3d,
    # Camera
    "intrinsics": K.intrinsics,
    # Depth for geometry backend
    "depth_gt": K.depth_maps,
}

# Test connector for SAM3_3D
CONN_SAM3_3D_TEST = {
    "images": K.images,
    "input_hw": K.input_hw,
    "original_hw": K.original_hw,
    # Geometric prompts (from external detector or user input)
    "prompt_boxes": K.boxes2d,  # External 2D detections as prompts
    # Camera
    "intrinsics": K.intrinsics,
    "padding": "padding",
}

# Loss connector for SAM3_3D
CONN_SAM3_3D_LOSS = {
    # Model outputs
    "pred_logits": pred_key("pred_logits"),
    "pred_boxes_2d": pred_key("pred_boxes_2d"),
    "pred_boxes_3d": pred_key("pred_boxes_3d"),
    "aux_outputs": pred_key("aux_outputs"),
    "geom_losses": pred_key("geom_losses"),
    # Matching indices (computed by model)
    "indices": pred_key("indices"),
    # Targets
    "targets": {
        "boxes": data_key(K.boxes2d),
        "boxes_xyxy": data_key(K.boxes2d),  # Will be converted
        "boxes_3d": data_key(K.boxes3d),
        "num_boxes": data_key("num_boxes"),
    },
    # Camera
    "intrinsics": data_key(K.intrinsics),
}

# Evaluation connector
CONN_SAM3_3D_EVAL = {
    "coco_image_id": data_key(K.sample_names),
    "pred_boxes": pred_key("boxes"),
    "pred_scores": pred_key("scores"),
    "pred_classes": pred_key("class_ids"),
    "pred_boxes3d": pred_key("boxes3d"),
}

# Visualization connector
CONN_SAM3_3D_VIS = {
    "images": data_key(K.original_images),
    "image_names": data_key(K.sample_names),
    "intrinsics": data_key("original_intrinsics"),
    "boxes3d": pred_key("boxes3d"),
    "class_ids": pred_key("class_ids"),
    "scores": pred_key("scores"),
}


def get_sam3_3d_data_connector_cfg() -> tuple[ConfigDict, ConfigDict]:
    """Get SAM3_3D data connector configuration.
    
    Returns:
        Tuple of (train_connector, test_connector).
    """
    train_data_connector = class_config(
        DataConnector, key_mapping=CONN_SAM3_3D_TRAIN
    )
    
    test_data_connector = class_config(
        DataConnector, key_mapping=CONN_SAM3_3D_TEST
    )
    
    return train_data_connector, test_data_connector

