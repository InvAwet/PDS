"""
Stability Analysis Module

This module implements stability analysis for equilibrium positions using the Hessian matrix approach.
It provides functions to determine which equilibrium positions are stable by examining the 
eigenvalues of the Hessian matrix of the force potential.

Based on classical mechanics, an equilibrium point is:
- Stable: if all eigenvalues of the Hessian are positive (energy minimum)
- Unstable: if all eigenvalues are negative (energy maximum)
- Saddle point: if some eigenvalues are positive and some are negative
"""

import numpy as np
import particle_physics as pp
from scipy.optimize import approx_fprime

def calculate_force_potential(position, alpha, Re, ell, rho=1.0, Um=1.0, mu=1.0, 
                             aspect_ratio=1.0, use_ar_correction=True, reference_ar=1.0, 
                             cl_neg=0.5, cl_pos=0.2):
    """
    Calculate the force potential at a given position.
    
    In our system, we don't have a direct potential function, but we can create a pseudo-potential
    where the gradient (negative) equals the force. This allows us to use potential-based stability analysis.
    
    Parameters:
    - position: [x, y] coordinates
    - alpha: Particle size ratio (a/ell)
    - Re: Reynolds number
    - ell: Channel half-width
    - Other parameters same as in pp.lift_force
    
    Returns:
    - Pseudo-potential value
    """
    # Since we don't have an analytical potential, we use the force magnitude as a surrogate
    # A stable equilibrium is where force magnitude is minimized
    x0, y0 = position
    fx, fy = pp.lift_force(x0, y0, alpha, Re, ell, rho, Um, mu, aspect_ratio, 
                           use_ar_correction, reference_ar, cl_neg, cl_pos)
    
    # Return negative of force magnitude squared as our potential
    # This creates a potential where minima occur at force equilibria
    return np.sum(fx**2 + fy**2)

def calculate_hessian(position, alpha, Re, ell, rho=1.0, Um=1.0, mu=1.0, 
                     aspect_ratio=1.0, use_ar_correction=True, reference_ar=1.0, 
                     cl_neg=0.5, cl_pos=0.2, epsilon=1e-6):
    """
    Calculate the Hessian matrix of the force potential at a given position.
    
    The Hessian is a matrix of second derivatives that describes the local curvature
    of the potential function, which determines stability of equilibrium points.
    
    Parameters:
    - position: [x, y] coordinates
    - Other parameters same as in calculate_force_potential
    - epsilon: Step size for finite difference approximation
    
    Returns:
    - 2x2 Hessian matrix
    """
    # Define function that only takes position as input for use with approx_fprime
    def potential_func(pos):
        return calculate_force_potential(pos, alpha, Re, ell, rho, Um, mu, 
                                        aspect_ratio, use_ar_correction, 
                                        reference_ar, cl_neg, cl_pos)
    
    # Calculate gradient of potential (first derivatives)
    grad = approx_fprime(position, potential_func, epsilon)
    
    # Calculate Hessian (second derivatives)
    hess = np.zeros((2, 2))
    
    # d^2f/dx^2
    hess[0, 0] = (potential_func([position[0] + epsilon, position[1]]) - 
                 2 * potential_func(position) + 
                 potential_func([position[0] - epsilon, position[1]])) / epsilon**2
    
    # d^2f/dy^2
    hess[1, 1] = (potential_func([position[0], position[1] + epsilon]) - 
                 2 * potential_func(position) + 
                 potential_func([position[0], position[1] - epsilon])) / epsilon**2
    
    # d^2f/dxdy = d^2f/dydx (mixed partial derivatives)
    hess[0, 1] = hess[1, 0] = (potential_func([position[0] + epsilon, position[1] + epsilon]) - 
                              potential_func([position[0] + epsilon, position[1] - epsilon]) - 
                              potential_func([position[0] - epsilon, position[1] + epsilon]) + 
                              potential_func([position[0] - epsilon, position[1] - epsilon])) / (4 * epsilon**2)
    
    return hess

def analyze_equilibrium_stability(positions, alpha, Re, ell, rho=1.0, Um=1.0, mu=1.0, 
                                 aspect_ratio=1.0, use_ar_correction=True, reference_ar=1.0, 
                                 cl_neg=0.5, cl_pos=0.2):
    """
    Analyze the stability of equilibrium positions.
    
    Parameters:
    - positions: List of [x, y] equilibrium positions
    - Other parameters same as in calculate_hessian
    
    Returns:
    - List of dictionaries with equilibrium positions and their stability properties
    """
    # Group very similar positions (within tolerance)
    grouped_positions = []
    tolerance = 0.05 * ell  # Group positions within 5% of channel half-width
    
    # Group similar positions together
    for pos in positions:
        # Check if this position is close to an existing group
        found_group = False
        for group in grouped_positions:
            # Calculate distance to group center
            group_center = np.mean(group, axis=0)
            distance = np.sqrt((pos[0] - group_center[0])**2 + (pos[1] - group_center[1])**2)
            
            if distance < tolerance:
                # Add to existing group
                group.append(pos)
                found_group = True
                break
                
        if not found_group:
            # Create new group
            grouped_positions.append([pos])
    
    # Generate results for each group's average position
    results = []
    
    for group in grouped_positions:
        # Calculate average position for the group
        avg_pos = np.mean(group, axis=0).tolist()
        
        # Calculate Hessian at this equilibrium position
        hessian = calculate_hessian(avg_pos, alpha, Re, ell, rho, Um, mu, 
                                   aspect_ratio, use_ar_correction, 
                                   reference_ar, cl_neg, cl_pos)
        
        # Calculate eigenvalues to determine stability
        try:
            eigenvalues = np.linalg.eigvals(hessian)
            
            # Determine stability type based on eigenvalues
            if np.all(eigenvalues > 0):
                stability_type = "stable"
            elif np.all(eigenvalues < 0):
                stability_type = "unstable"
            else:
                stability_type = "saddle"
                
            # Calculate normalized position
            if aspect_ratio >= 1.0:
                x_norm = avg_pos[0] / ell
                y_norm = avg_pos[1] / (ell * aspect_ratio)
            else:
                x_norm = avg_pos[0] / (ell / aspect_ratio)
                y_norm = avg_pos[1] / ell
            
            # Calculate potential value
            potential = calculate_force_potential(avg_pos, alpha, Re, ell, rho, Um, mu, 
                                              aspect_ratio, use_ar_correction, 
                                              reference_ar, cl_neg, cl_pos)
            
            # Store results with additional group information
            results.append({
                "position": avg_pos,
                "normalized_position": [x_norm, y_norm],
                "hessian": hessian,
                "eigenvalues": eigenvalues,
                "stability_type": stability_type,
                "potential_value": potential,
                "group_size": len(group),
                "original_positions": group
            })
        except np.linalg.LinAlgError:
            # Handle case where eigenvalue calculation fails
            results.append({
                "position": avg_pos,
                "normalized_position": [avg_pos[0]/ell, avg_pos[1]/ell],
                "stability_type": "undetermined",
                "error": "Could not calculate eigenvalues",
                "group_size": len(group),
                "original_positions": group
            })
    
    # Sort results by stability (stable first, then saddle, then unstable)
    stability_order = {"stable": 0, "saddle": 1, "unstable": 2, "undetermined": 3}
    results.sort(key=lambda x: stability_order.get(x["stability_type"], 4))
    
    return results

def analyze_stability(position, particle_radius, Um, ell, aspect_ratio=1.0, 
                     use_ar_correction=True, reference_ar=1.0, cl_neg=0.5, cl_pos=0.2):
    """
    Analyze the stability of a single equilibrium position.
    
    Parameters:
    - position: [x, y] coordinates of the equilibrium position
    - particle_radius: Radius of the particle relative to channel width
    - Um: Maximum flow velocity
    - ell: Channel half-width
    - aspect_ratio: Ratio of channel height to width (h/w)
    - use_ar_correction: Whether to apply aspect ratio correction
    - reference_ar: Reference aspect ratio (for which the model is calibrated)
    - cl_neg: Negative lift coefficient (CL-)
    - cl_pos: Positive lift coefficient (CL+)
    
    Returns:
    - List with one dictionary containing stability analysis results
    """
    # Convert particle radius to alpha (particle size ratio)
    alpha = particle_radius / ell
    
    # Use Reynolds number of 10 as default (can be adjusted)
    Re = 10.0
    
    # Calculate Hessian at this equilibrium position
    hessian = calculate_hessian(position, alpha, Re, ell, 1.0, Um, 1.0, 
                               aspect_ratio, use_ar_correction, 
                               reference_ar, cl_neg, cl_pos)
    
    # Calculate eigenvalues to determine stability
    try:
        eigenvalues = np.linalg.eigvals(hessian)
        
        # Determine stability type based on eigenvalues
        if np.all(eigenvalues > 0):
            stability_type = "stable"
        elif np.all(eigenvalues < 0):
            stability_type = "unstable"
        else:
            stability_type = "saddle"
            
        # Calculate normalized position
        if aspect_ratio >= 1.0:
            x_norm = position[0] / ell
            y_norm = position[1] / (ell * aspect_ratio)
        else:
            x_norm = position[0] / (ell / aspect_ratio)
            y_norm = position[1] / ell
        
        # Calculate potential value
        potential = calculate_force_potential(position, alpha, Re, ell, 1.0, Um, 1.0, 
                                          aspect_ratio, use_ar_correction, 
                                          reference_ar, cl_neg, cl_pos)
        
        # Store results
        results = [{
            "position": position,
            "normalized_position": [x_norm, y_norm],
            "hessian": hessian,
            "eigenvalues": eigenvalues,
            "stability_type": stability_type,
            "potential_value": potential,
            "group_size": 1,
            "original_positions": [position]
        }]
        
    except np.linalg.LinAlgError:
        # Handle case where eigenvalue calculation fails
        results = [{
            "position": position,
            "normalized_position": [position[0]/ell, position[1]/ell],
            "stability_type": "undetermined",
            "error": "Could not calculate eigenvalues",
            "group_size": 1,
            "original_positions": [position]
        }]
    
    return results