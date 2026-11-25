import math

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def hd95(prediction, target):
    """Symmetric HD95 in pixels for two boolean masks on the valid 2D image grid."""
    prediction_empty, target_empty = not prediction.any(), not target.any()
    if prediction_empty and target_empty:
        return 0.0
    if prediction_empty or target_empty:
        height, width = target.shape
        return math.hypot(height - 1, width - 1)

    # The default erosion uses 4-connectivity and treats the image exterior as background.
    predicted_boundary = prediction & ~binary_erosion(prediction)
    target_boundary = target & ~binary_erosion(target)
    to_target = distance_transform_edt(~target_boundary)[predicted_boundary]
    to_prediction = distance_transform_edt(~predicted_boundary)[target_boundary]
    return float(max(np.percentile(to_target, 95), np.percentile(to_prediction, 95)))
