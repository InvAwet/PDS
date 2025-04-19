import numpy as np
import matplotlib.pyplot as plt
import particle_solver as solver
import particle_physics as pp
import time

def parameter_sweep(param_name, param_values, particle_radius, Um, ell, aspect_ratio,
                   search_resolution=8, use_ar_correction=True, reference_ar=1.0, 
                   cl_neg=0.5, cl_pos=0.2):
    """
    Perform a parameter sweep to find equilibrium positions for different parameter values
    
    Parameters:
    - param_name: Name of parameter to vary ('aspect_ratio', 'particle_radius', 'Re')
    - param_values: List of parameter values to test
    - particle_radius: Default particle radius
    - Um: Maximum flow velocity
    - ell: Channel half-width
    - aspect_ratio: Default aspect ratio
    - search_resolution: Resolution for equilibrium search
    - use_ar_correction: Whether to apply aspect ratio correction
    - reference_ar: Reference aspect ratio (for which the model is calibrated)
    - cl_neg: Negative lift coefficient (CL-)
    - cl_pos: Positive lift coefficient (CL+)
    
    Returns:
    - Dictionary with sweep results
    """
    # Initialize results dictionary
    results = {
        'param_name': param_name,
        'param_values': param_values,
        'positions': [],
        'force_fields': []
    }
    
    # Loop through parameter values
    for value in param_values:
        # Set parameters based on which one we're varying
        if param_name == 'aspect_ratio':
            # Vary aspect ratio, keep particle radius constant
            ar = value
            pr = particle_radius
            re = None  # Reynolds number not used directly
        elif param_name == 'particle_radius':
            # Vary particle radius, keep aspect ratio constant
            ar = aspect_ratio
            pr = value
            re = None  # Reynolds number not used directly
        elif param_name == 'Re':
            # Re doesn't directly affect our solver, but we'll record it
            ar = aspect_ratio
            pr = particle_radius
            re = value
        else:
            raise ValueError(f"Unknown parameter name: {param_name}")
        
        # Skip invalid combinations (e.g., particle too large for channel)
        if pr >= ell or (ar >= 1.0 and pr >= ell * ar) or (ar < 1.0 and pr >= ell / ar):
            # Add None values to maintain index correspondence
            results['positions'].append(None)
            results['force_fields'].append(None)
            continue
        
        # Find equilibrium position with the current parameter value
        try:
            eq_position, force_field = solver.find_equilibrium_position(
                pr, Um, ell, ar, 
                search_resolution=search_resolution,
                search_entire_channel=True,
                use_ar_correction=use_ar_correction,
                reference_ar=reference_ar,
                cl_neg=cl_neg,
                cl_pos=cl_pos
            )
            
            # Store results
            results['positions'].append(eq_position)
            results['force_fields'].append(force_field)
            
        except Exception as e:
            print(f"Error for {param_name}={value}: {e}")
            # Add None values to maintain index correspondence
            results['positions'].append(None)
            results['force_fields'].append(None)
    
    return results

def simulate_multi_particles(num_particles, particle_radius, Um, ell, aspect_ratio,
                            search_resolution=8, use_ar_correction=True, reference_ar=1.0, 
                            cl_neg=0.5, cl_pos=0.2):
    """
    Simulate multiple particles and find their equilibrium positions
    
    Parameters:
    - num_particles: Number of particles to simulate
    - particle_radius: Radius of the particles
    - Um: Maximum flow velocity
    - ell: Channel half-width
    - aspect_ratio: Ratio of channel height to width (h/w)
    - search_resolution: Resolution for equilibrium search
    - use_ar_correction: Whether to apply aspect ratio correction
    - reference_ar: Reference aspect ratio (for which the model is calibrated)
    - cl_neg: Negative lift coefficient (CL-)
    - cl_pos: Positive lift coefficient (CL+)
    
    Returns:
    - Dictionary with multi-particle results
    """
    # Initialize results dictionary
    results = {
        'num_particles': num_particles,
        'positions': [],
        'force_fields': []
    }
    
    # Calculate channel dimensions
    if aspect_ratio >= 1.0:
        height = ell * aspect_ratio * 2
        width = ell * 2
    else:
        height = ell * 2
        width = ell / aspect_ratio * 2
    
    # Define starting positions for particles
    # We'll use different quadrants and positions to spread them out
    start_positions = []
    
    # Determine safe distance from walls
    safe_margin = 1.5 * particle_radius
    
    # Create a grid of possible starting positions
    x_points = np.linspace(-ell + safe_margin, ell - safe_margin, 4)
    y_points = np.linspace(-ell * aspect_ratio + safe_margin, ell * aspect_ratio - safe_margin, 4)
    
    # Generate all possible positions from grid
    all_positions = []
    for x in x_points:
        for y in y_points:
            all_positions.append((x, y))
    
    # Take the first num_particles positions
    start_positions = all_positions[:num_particles]
    
    # If we need more positions than our grid provides, add some random ones
    if num_particles > len(all_positions):
        for i in range(num_particles - len(all_positions)):
            # Generate random position within safe margins
            x = np.random.uniform(-ell + safe_margin, ell - safe_margin)
            y = np.random.uniform(-ell * aspect_ratio + safe_margin, ell * aspect_ratio - safe_margin)
            start_positions.append((x, y))
    
    # Find equilibrium for each starting position
    for start_pos in start_positions:
        try:
            # Use starting position as initial guess
            eq_position, force_field = solver.find_equilibrium_position(
                particle_radius, Um, ell, aspect_ratio, 
                initial_guess=start_pos,
                search_resolution=search_resolution,
                search_entire_channel=True,
                use_ar_correction=use_ar_correction,
                reference_ar=reference_ar,
                cl_neg=cl_neg,
                cl_pos=cl_pos
            )
            
            # Check if this position is too close to any existing position
            too_close = False
            for pos in results['positions']:
                dist = np.sqrt((eq_position[0] - pos[0])**2 + (eq_position[1] - pos[1])**2)
                if dist < 2 * particle_radius:
                    too_close = True
                    break
            
            # Only add if not too close to existing positions
            # But always add the first particle
            if not too_close or len(results['positions']) == 0:
                results['positions'].append(eq_position)
                results['force_fields'].append(force_field)
            else:
                # If too close, try a different starting position
                # Generate random position within safe margins
                x = np.random.uniform(-ell + safe_margin, ell - safe_margin)
                y = np.random.uniform(-ell * aspect_ratio + safe_margin, ell * aspect_ratio - safe_margin)
                
                # Try again with new position
                eq_position, force_field = solver.find_equilibrium_position(
                    particle_radius, Um, ell, aspect_ratio, 
                    initial_guess=(x, y),
                    search_resolution=search_resolution//2,
                    search_entire_channel=True,
                    use_ar_correction=use_ar_correction,
                    reference_ar=reference_ar,
                    cl_neg=cl_neg,
                    cl_pos=cl_pos
                )
                
                results['positions'].append(eq_position)
                results['force_fields'].append(force_field)
            
        except Exception as e:
            print(f"Error for particle at {start_pos}: {e}")
            # Skip this particle if there's an error
            continue
        
        # Stop if we've found enough particles
        if len(results['positions']) >= num_particles:
            break
    
    return results
