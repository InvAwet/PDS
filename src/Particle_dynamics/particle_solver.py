import numpy as np
from scipy.optimize import minimize
import particle_physics as pp

def calculate_force_on_particle(x0, y0, Up, particle_radius, ell, Um, aspect_ratio=1.0):
    """
    Calculate the force on a particle based on its position and velocity
    
    Parameters:
    - x0, y0: Particle center coordinates
    - Up: Particle velocity vector [Upx, Upy]
    - particle_radius: Radius of the particle
    - ell: Channel half-width
    - Um: Maximum flow velocity
    - aspect_ratio: Ratio of channel height to width (h/w)
    
    Returns:
    - Force vector [Fx, Fy]
    """
    # Get background flow properties at particle center
    beta, gamma_x, gamma_y, delta_xx, delta_xy, delta_yy = pp.flow_derivatives(
        x0, y0, ell, Um, aspect_ratio)
    
    # Calculate background flow velocity at particle position
    u_background = beta
    
    # Calculate slip velocity (difference between particle and background flow)
    slip_velocity = u_background - Up[0]
    
    # Stokes drag calculation (6πμrU)
    # Assuming fluid density of 1.0 for simplicity
    mu = 1.0  # dynamic viscosity
    drag_coefficient = 6 * np.pi * mu * particle_radius
    
    # Force in x-direction due to drag
    Fx = drag_coefficient * slip_velocity
    
    # Calculate lift force using Saffman lift formula
    # For a particle in shear flow, the lift force depends on the gradient
    shear_rate = gamma_y  # du/dy - shear rate
    lift_coefficient = 6.46 * mu * particle_radius**2
    
    # Calculate the sign of the lift (direction)
    lift_sign = 1.0 if shear_rate > 0 else -1.0
    
    # Lift force (perpendicular to flow direction)
    Fy = lift_sign * lift_coefficient * np.sqrt(np.abs(shear_rate)) * slip_velocity
    
    # Apply wall corrections
    distance_to_walls = [
        abs(x0 - ell),  # Distance to right wall
        abs(x0 + ell),  # Distance to left wall
        abs(y0 - ell),  # Distance to top wall
        abs(y0 + ell)   # Distance to bottom wall
    ]
    
    # Apply lubrication corrections for all walls
    for i, dist in enumerate(distance_to_walls):
        # Apply correction factor to the appropriate component
        correction = pp.lubrication_correction(dist, particle_radius)
        if i < 2:  # x-direction walls
            Fx *= correction
        else:      # y-direction walls
            Fy *= correction
    
    return np.array([Fx, Fy])

def find_equilibrium_position(particle_radius, Um, ell, aspect_ratio=1.0, 
                               initial_guess=None, search_resolution=10, search_entire_channel=True,
                               use_ar_correction=True, reference_ar=1.0, cl_neg=0.5, cl_pos=0.2):
    """
    Find the equilibrium position of a particle in a channel flow
    
    Parameters:
    - particle_radius: Radius of the particle
    - Um: Maximum flow velocity
    - ell: Channel half-width
    - aspect_ratio: Ratio of channel height to width (h/w)
    - initial_guess: Initial position guess [x0, y0]
    - search_resolution: Number of initial positions to try
    - search_entire_channel: Whether to search all quadrants of the channel
    - use_ar_correction: Whether to apply aspect ratio correction
    - reference_ar: Reference aspect ratio (for which the model is calibrated)
    - cl_neg: Negative lift coefficient (CL-)
    - cl_pos: Positive lift coefficient (CL+)
    
    Returns:
    - Equilibrium position [x_eq, y_eq]
    - Force field data
    """
    # If no initial guess provided, use grid search
    if initial_guess is None:
        # Create a grid of initial positions within the channel
        # Avoid positions too close to the walls
        safe_margin = 1.5 * particle_radius
        
        # Set up grid points based on whether we're searching the whole channel or just one quadrant
        if search_entire_channel:
            x_points = np.linspace(-ell + safe_margin, ell - safe_margin, search_resolution)
            
            # For aspect ratio calculations
            if aspect_ratio >= 1.0:
                height = ell * aspect_ratio
            else:
                height = ell
                
            y_points = np.linspace(-height + safe_margin, height - safe_margin, search_resolution)
        else:
            # Search only positive quadrant for efficiency
            x_points = np.linspace(0, ell - safe_margin, search_resolution//2)
            
            # For aspect ratio calculations
            if aspect_ratio >= 1.0:
                height = ell * aspect_ratio
            else:
                height = ell
                
            y_points = np.linspace(0, height - safe_margin, search_resolution//2)
        
        best_residual = float('inf')
        best_position = None
        
        force_field = {
            'x': [], 'y': [], 'fx': [], 'fy': [], 'magnitude': []
        }
        
        # Try each grid point as initial guess
        for x in x_points:
            for y in y_points:
                # Skip positions outside the valid channel region
                if aspect_ratio >= 1.0:
                    if abs(x) > ell or abs(y) > ell * aspect_ratio:
                        continue
                else:
                    if abs(x) > ell / aspect_ratio or abs(y) > ell:
                        continue
                
                # Check if particle at this position would intersect with walls
                if (abs(x) + particle_radius > ell or 
                    abs(y) + particle_radius > ell * aspect_ratio):
                    continue
                
                # Calculate force at this position
                # Assuming zero particle velocity at equilibrium
                Up = [0.0, 0.0]
                force = calculate_force_on_particle(x, y, Up, particle_radius, ell, Um, aspect_ratio)
                
                # Store force field data
                force_field['x'].append(x)
                force_field['y'].append(y)
                force_field['fx'].append(force[0])
                force_field['fy'].append(force[1])
                force_field['magnitude'].append(np.linalg.norm(force))
                
                # Update best position if force is closer to zero
                residual = np.sum(force**2)
                if residual < best_residual:
                    best_residual = residual
                    best_position = [x, y]
        
        if best_position is None:
            raise ValueError("Could not find valid initial position")
        
        initial_guess = best_position
    
    # Define objective function to minimize (sum of squared forces)
    def objective(position):
        x, y = position
        # Check if position is within bounds
        if (abs(x) + particle_radius > ell or 
            abs(y) + particle_radius > ell * aspect_ratio):
            return 1e10  # Large penalty for out-of-bounds
        
        # At equilibrium, particle velocity is zero
        Up = [0.0, 0.0]
        force = calculate_force_on_particle(x, y, Up, particle_radius, ell, Um, aspect_ratio)
        return np.sum(force**2)
    
    # Set bounds to keep particle within channel
    bounds = [
        (-ell + particle_radius, ell - particle_radius),  # x bounds
        (-ell * aspect_ratio + particle_radius, ell * aspect_ratio - particle_radius)  # y bounds
    ]
    
    # Run optimization to find equilibrium
    result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
    
    # Check if optimization succeeded
    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")
    
    # If we got here using the grid search approach, force_field is defined
    # Otherwise (when using initial_guess directly), we need to create it
    if 'force_field' not in locals():
        force_field = {
            'x': [], 'y': [], 'fx': [], 'fy': [], 'magnitude': []
        }
    
    # Return equilibrium position and force field data
    return result.x, force_field

def calculate_flow_field(x_eq, y_eq, particle_radius, ell, Um, aspect_ratio=1.0, 
                          grid_resolution=30):
    """
    Calculate flow field around the particle at equilibrium position
    
    Parameters:
    - x_eq, y_eq: Equilibrium position
    - particle_radius: Radius of the particle
    - ell: Channel half-width
    - Um: Maximum flow velocity
    - aspect_ratio: Ratio of channel height to width (h/w)
    - grid_resolution: Resolution of the calculation grid
    
    Returns:
    - Dictionary with flow field data
    """
    # Create grid around particle, focusing on the region near the particle
    margin = 4 * particle_radius
    x_min, x_max = x_eq - margin, x_eq + margin
    y_min, y_max = y_eq - margin, y_eq + margin
    
    # Ensure grid stays within channel bounds
    x_min = max(x_min, -ell + 0.01)
    x_max = min(x_max, ell - 0.01)
    y_min = max(y_min, -ell * aspect_ratio + 0.01)
    y_max = min(y_max, ell * aspect_ratio - 0.01)
    
    x = np.linspace(x_min, x_max, grid_resolution)
    y = np.linspace(y_min, y_max, grid_resolution)
    X, Y = np.meshgrid(x, y)
    
    # Initialize arrays for flow velocities
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    U_background = np.zeros_like(X)
    
    # Calculate background and total flow field at each point
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            xi, yi = X[i, j], Y[i, j]
            
            # Calculate background flow (without particle)
            u_bg = pp.background_flow(xi, yi, ell, Um, aspect_ratio)
            U_background[i, j] = u_bg
            
            # Calculate particle contribution to flow field
            # At equilibrium, particle velocity is zero
            Up = [0.0, 0.0]
            
            # Check if point is inside particle
            dist_to_particle = np.sqrt((xi - x_eq)**2 + (yi - y_eq)**2)
            if dist_to_particle < particle_radius:
                # Inside particle, flow velocity is zero relative to particle
                u_total = Up
            else:
                # Calculate disturbance flow due to particle
                u_disturbance = pp.method_of_reflections_flow(
                    xi, yi, x_eq, y_eq, Up, particle_radius, ell)
                
                # Total flow is background + disturbance
                u_total = np.array([u_bg, 0.0]) + u_disturbance
            
            U[i, j] = u_total[0]
            V[i, j] = u_total[1]
    
    # Return flow field data
    return {
        'X': X, 'Y': Y, 
        'U': U, 'V': V, 
        'U_background': U_background
    }
