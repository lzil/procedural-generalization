# code from Adam Gleave

import scipy
import numpy as np

def least_l2_affine(
    source: np.ndarray, target: np.ndarray, shift: bool = True, scale: bool = True
):
    """Finds the squared-error minimizing affine transform.

    Args:
        source: a 1D array consisting of the reward to transform.
        target: a 1D array consisting of the target to match.
        shift: affine includes constant shift.
        scale: affine includes rescale.

    Returns:
        (shift, scale) such that (scale * reward + shift) has minimal squared-error from target.

    Raises:
        ValueError if source or target are not 1D arrays, or if neither shift or scale are True.
    """
    if source.ndim != 1:
        raise ValueError("source must be vector.")
    if target.ndim != 1:
        raise ValueError("target must be vector.")
    if not (shift or scale):
        raise ValueError("At least one of shift and scale must be True.")

    a_vals = []
    if shift:
        # Positive and negative constant.
        # The shift will be the sum of the coefficients of these terms.
        a_vals += [np.ones_like(source), -np.ones_like(source)]
    if scale:
        a_vals += [source]
    a_vals = np.stack(a_vals, axis=1)
    # Find x such that a_vals.dot(x) has least-squared error from target, where x >= 0.
    coefs, _ = scipy.optimize.nnls(a_vals, target)

    shift_param = 0.0
    scale_idx = 0
    if shift:
        shift_param = coefs[0] - coefs[1]
        scale_idx = 2

    scale_param = 1.0
    if scale:
        scale_param = coefs[scale_idx]

    return shift_param, scale_param