"""
Aspect Ratio Correction Module

This module implements the aspect ratio correction based on Zhou & Papautsky (2013)
two-stage focusing model for non-square channels. It provides utility functions
to calculate correction factors for forces in channels with aspect ratios
different from 1.0.
"""

import numpy as np

def calculate_ar_correction_factor(aspect_ratio, reference_ar=1.0, cl_neg=0.5, cl_pos=0.2):
    """
    Calculate the aspect ratio correction factor K(AR) based on Zhou & Papautsky (2013)
    two-stage focusing model.
    
    Parameters:
    - aspect_ratio: Current aspect ratio (H/W)
    - reference_ar: Reference aspect ratio (for which the model is calibrated, typically 1.0)
    - cl_neg: Negative lift coefficient (CL-)
    - cl_pos: Positive lift coefficient (CL+)
    
    Returns:
    - K(AR): Correction factor for scaling forces
    """
    # For the reference case (typically square channel, AR=1)
    if reference_ar >= 1.0:
        h_ref = reference_ar
        w_ref = 1.0
    else:
        h_ref = 1.0
        w_ref = 1.0 / reference_ar
    
    # For the current aspect ratio
    if aspect_ratio >= 1.0:
        h = aspect_ratio
        w = 1.0
    else:
        h = 1.0
        w = 1.0 / aspect_ratio
    
    # Calculate the correction factor using the formula from Zhou & Papautsky
    # K(AR) = (H/W^2 + W/H^2) / (H_ref/W_ref^2 + W_ref/H_ref^2)
    # where H and W are normalized
    
    numerator = (h / (w**2)) + (w / (h**2))
    denominator = (h_ref / (w_ref**2)) + (w_ref / (h_ref**2))
    
    # Calculate correction factor
    k_ar = numerator / denominator
    
    # Adjust with lift coefficients
    k_ar_adjusted = k_ar * (cl_neg / cl_pos)
    
    return k_ar_adjusted

def apply_ar_correction(force, aspect_ratio, reference_ar=1.0, cl_neg=0.5, cl_pos=0.2):
    """
    Apply aspect ratio correction to a force vector.
    
    Parameters:
    - force: Force vector [Fx, Fy]
    - aspect_ratio: Current aspect ratio (H/W)
    - reference_ar: Reference aspect ratio (for which the model is calibrated)
    - cl_neg: Negative lift coefficient (CL-)
    - cl_pos: Positive lift coefficient (CL+)
    
    Returns:
    - Corrected force vector
    """
    # Skip correction if aspect ratio is nearly identical to reference
    if abs(aspect_ratio - reference_ar) < 0.05:
        return force
    
    # Calculate correction factor
    k_ar = calculate_ar_correction_factor(aspect_ratio, reference_ar, cl_neg, cl_pos)
    
    # Apply correction factor to force
    # The scale factor is applied differently to x and y components
    # based on the aspect ratio
    fx, fy = force
    
    if aspect_ratio >= 1.0:
        # Tall channel: stronger correction on y component (height)
        fx_corrected = fx
        fy_corrected = fy * k_ar
    else:
        # Wide channel: stronger correction on x component (width)
        fx_corrected = fx * k_ar
        fy_corrected = fy
    
    return np.array([fx_corrected, fy_corrected])

def get_correction_info(aspect_ratio, reference_ar=1.0, cl_neg=0.5, cl_pos=0.2):
    """
    Get information about the aspect ratio correction for display.
    
    Parameters:
    - aspect_ratio: Current aspect ratio (H/W)
    - reference_ar: Reference aspect ratio (for which the model is calibrated)
    - cl_neg: Negative lift coefficient (CL-)
    - cl_pos: Positive lift coefficient (CL+)
    
    Returns:
    - Dictionary with correction information
    """
    k_ar = calculate_ar_correction_factor(aspect_ratio, reference_ar, cl_neg, cl_pos)
    
    # Determine if we're scaling up or down
    if k_ar > 1.0:
        direction = "amplifying"
    elif k_ar < 1.0:
        direction = "reducing"
    else:
        direction = "not changing"
    
    # Determine which force component is more affected
    if aspect_ratio >= 1.0:
        affected_component = "y (height)"
    else:
        affected_component = "x (width)"
    
    return {
        "correction_factor": k_ar,
        "direction": direction,
        "affected_component": affected_component,
        "original_ar": aspect_ratio,
        "reference_ar": reference_ar,
        "cl_neg": cl_neg,
        "cl_pos": cl_pos
    }