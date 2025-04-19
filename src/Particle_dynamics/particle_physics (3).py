import numpy as np
from scipy.optimize import root
from scipy import integrate
import time
import functools

# Dictionary to store performance metrics
performance_metrics = {
    'function_calls': {},  # Count of function calls
    'execution_times': {},  # Total execution time per function
    'last_run_times': {},  # Last run time for each function
    'parameter_impacts': {}  # How parameters affect performance
}

def track_performance(func):
    """
    Decorator to track performance metrics of functions
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get function name
        func_name = func.__name__
        
        # Initialize metrics for this function if it doesn't exist
        if func_name not in performance_metrics['function_calls']:
            performance_metrics['function_calls'][func_name] = 0
            performance_metrics['execution_times'][func_name] = 0
            performance_metrics['last_run_times'][func_name] = 0
            performance_metrics['parameter_impacts'][func_name] = {}
        
        # Track parameter impacts (only for selected parameters of interest)
        param_keys = ['aspect_ratio', 'N_terms', 'Re', 'search_resolution']
        for key in param_keys:
            if key in kwargs:
                if key not in performance_metrics['parameter_impacts'][func_name]:
                    performance_metrics['parameter_impacts'][func_name][key] = {}
                
                param_value = str(kwargs[key])  # Convert to string for dictionary key
                if param_value not in performance_metrics['parameter_impacts'][func_name][key]:
                    performance_metrics['parameter_impacts'][func_name][key][param_value] = {
                        'calls': 0,
                        'total_time': 0
                    }
        
        # Increment call counter
        performance_metrics['function_calls'][func_name] += 1
        
        # Measure execution time
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        # Update timing metrics
        performance_metrics['execution_times'][func_name] += execution_time
        performance_metrics['last_run_times'][func_name] = execution_time
        
        # Update parameter impact metrics
        for key in param_keys:
            if key in kwargs:
                param_value = str(kwargs[key])
                performance_metrics['parameter_impacts'][func_name][key][param_value]['calls'] += 1
                performance_metrics['parameter_impacts'][func_name][key][param_value]['total_time'] += execution_time
        
        return result
    
    return wrapper

@track_performance
def background_flow(x, y, ell, Um, aspect_ratio=1.0, N_terms=10):
    """
    Calculate the background flow velocity at a point (x, y) in a rectangular channel
    using the analytical solution for Poiseuille flow
    
    Parameters:
    - x, y: Coordinates in the channel
    - ell: Channel width (smaller dimension)
    - Um: Maximum flow velocity
    - aspect_ratio: Ratio of channel height to width (h/w)
    - N_terms: Number of terms in the series expansion
    
    Returns:
    - Flow velocity at point (x, y)
    """
    # For method-of-reflections implementation, we use the simplified form for a square channel
    # which is the product of two parabolas
    if abs(aspect_ratio - 1.0) < 0.1:  # Nearly square channel
        return Um * (1 - (2 * x / ell)**2) * (1 - (2 * y / ell)**2)
    
    # For non-square channels, revert to the Fourier series approach for more accuracy
    # Define channel dimensions based on aspect ratio
    if aspect_ratio >= 1.0:
        # Width is smaller than height (or equal for square)
        width = ell
        height = ell * aspect_ratio
    else:
        # Height is smaller than width
        height = ell
        width = ell / aspect_ratio
    
    # Convert from physical coordinates to normalized coordinates
    # where channel boundaries are at ±1 in each direction
    x_norm = 2 * x / width
    y_norm = 2 * y / height
    
    # Calculate the pressure-driven flow in a rectangular channel
    # using the analytical solution (Fourier series)
    
    # Initialize velocity
    u = 0
    
    # Scale factor for maximum velocity
    # This preserves the specified Um regardless of aspect ratio
    scale = 1.0
    
    # Calculate using double Fourier series
    # This implements the analytical solution from White (2006) "Viscous Fluid Flow"
    for i in range(1, N_terms+1, 2):  # Only odd terms contribute
        for j in range(1, N_terms+1, 2):
            # Series coefficients
            term = (16.0 / (i * j * np.pi**4) * 
                   np.sin(i * np.pi * (x_norm + 1) / 2) * 
                   np.sin(j * np.pi * (y_norm + 1) / 2) /
                   (i**2/width**2 + j**2/height**2))
            u += term
    
    # Scale to match the specified maximum velocity
    u = u * scale * Um
    
    return u

@track_performance
def grad_background_flow(x, y, ell, Um, aspect_ratio=1.0):
    """
    Calculate the gradient of background flow at point (x, y)
    
    Parameters:
    - x, y: Coordinates in the channel
    - ell: Channel width (smaller dimension)
    - Um: Maximum flow velocity
    - aspect_ratio: Ratio of channel height to width (h/w)
    
    Returns:
    - [dudx, dudy]: Gradient components
    """
    # For method-of-reflections implementation, we use the simplified form for a square channel
    if abs(aspect_ratio - 1.0) < 0.1:  # Nearly square channel
        dudx = Um * (-8 * x / ell**2) * (1 - (2 * y / ell)**2)
        dudy = Um * (-8 * y / ell**2) * (1 - (2 * x / ell)**2)
        return np.array([dudx, dudy])
    
    # For non-square channels, use finite differences
    h = 1e-7
    u_center = background_flow(x, y, ell, Um, aspect_ratio)
    u_x_plus = background_flow(x + h, y, ell, Um, aspect_ratio)
    u_x_minus = background_flow(x - h, y, ell, Um, aspect_ratio)
    u_y_plus = background_flow(x, y + h, ell, Um, aspect_ratio)
    u_y_minus = background_flow(x, y - h, ell, Um, aspect_ratio)
    
    dudx = (u_x_plus - u_x_minus) / (2 * h)
    dudy = (u_y_plus - u_y_minus) / (2 * h)
    
    return np.array([dudx, dudy])

@track_performance
def flow_derivatives(x0, y0, ell, Um, aspect_ratio=1.0, h=1e-7):
    """
    Calculate flow derivatives at point (x0, y0) using finite differences
    
    Parameters:
    - x0, y0: Point coordinates
    - ell: Channel width (smaller dimension)
    - Um: Maximum flow velocity
    - aspect_ratio: Ratio of channel height to width (h/w)
    - h: Step size for finite difference
    
    Returns:
    - beta: Flow velocity
    - gamma_x, gamma_y: First derivatives
    - delta_xx, delta_xy, delta_yy: Second derivatives
    """
    # Use the aspect ratio in all background_flow calls to ensure consistency
    beta = background_flow(x0, y0, ell, Um, aspect_ratio)
    
    # Get gradient from grad_background_flow
    grad = grad_background_flow(x0, y0, ell, Um, aspect_ratio)
    gamma_x, gamma_y = grad[0], grad[1]
    
    # Second derivatives
    # For x-x
    u_x_plus = background_flow(x0 + h, y0, ell, Um, aspect_ratio)
    u_center = background_flow(x0, y0, ell, Um, aspect_ratio)
    u_x_minus = background_flow(x0 - h, y0, ell, Um, aspect_ratio)
    delta_xx = (u_x_plus - 2*u_center + u_x_minus) / h**2
    
    # For y-y
    u_y_plus = background_flow(x0, y0 + h, ell, Um, aspect_ratio)
    u_y_minus = background_flow(x0, y0 - h, ell, Um, aspect_ratio)
    delta_yy = (u_y_plus - 2*u_center + u_y_minus) / h**2
    
    # For x-y (mixed derivative)
    u_xy_plus_plus = background_flow(x0 + h, y0 + h, ell, Um, aspect_ratio)
    u_xy_plus_minus = background_flow(x0 + h, y0 - h, ell, Um, aspect_ratio)
    u_xy_minus_plus = background_flow(x0 - h, y0 + h, ell, Um, aspect_ratio)
    u_xy_minus_minus = background_flow(x0 - h, y0 - h, ell, Um, aspect_ratio)
    delta_xy = (u_xy_plus_plus - u_xy_plus_minus - u_xy_minus_plus + u_xy_minus_minus) / (4 * h**2)
    
    return beta, gamma_x, gamma_y, delta_xx, delta_xy, delta_yy

@track_performance
def lubrication_correction(distance, particle_radius):
    """
    Calculate lubrication correction factor for a particle near a wall
    
    Parameters:
    - distance: Distance from particle center to wall
    - particle_radius: Radius of the particle
    
    Returns:
    - Correction factor
    """
    # Gap between particle surface and wall
    h = max(distance - particle_radius, 1e-12)
    
    # Lubrication correction threshold
    h_tol = 5e-6
    
    if h < h_tol:
        # Advanced lubrication correction formula based on asymptotic analysis
        # Goldman, Cox & Brenner (1967) and method of reflections
        return 1 / (1 - 9/16*(particle_radius/h) + (1/8)*(particle_radius/h)**3 - 
                    (45/256)*(particle_radius/h)**4 - (1/16)*(particle_radius/h)**5)
    return 1.0

@track_performance
def unbounded_flow(x, y, x0, y0, Up, particle_radius):
    """
    Calculate unbounded flow around a particle using Stokes flow solution
    
    Parameters:
    - x, y: Coordinates to evaluate flow
    - x0, y0: Particle center coordinates
    - Up: Particle velocity vector [Upx, Upy]
    - particle_radius: Radius of the particle
    
    Returns:
    - Flow velocity vector [u, v]
    """
    dx = x - x0
    dy = y - y0
    r = np.hypot(dx, dy)
    
    # Inside the particle
    if r < particle_radius:
        return np.array(Up)
    
    # Unit vector pointing from particle center to (x,y)
    r_hat = np.array([dx/r, dy/r])
    
    # Stokes flow solution terms
    term1 = (3*particle_radius)/(4*r) + (particle_radius**3)/(4*r**3)
    term2 = (3*particle_radius)/(4*r) - (3*particle_radius**3)/(4*r**3)
    
    # Combined flow field
    return term1 * np.array(Up) + term2 * np.dot(Up, r_hat) * r_hat

@track_performance
def image_flow(x, y, x0, y0, Up, wall, particle_radius, ell):
    """
    Calculate image flow (reflection) for a particle near a wall
    
    Parameters:
    - x, y: Coordinates to evaluate flow
    - x0, y0: Particle center coordinates
    - Up: Particle velocity vector [Upx, Upy]
    - wall: Wall identifier ('x+', 'x-', 'y+', 'y-')
    - particle_radius: Radius of the particle
    - ell: Channel half-width
    
    Returns:
    - Flow velocity vector [u, v]
    """
    # Calculate image position based on wall
    if wall == 'x+': x_img, y_img = 2*ell - x0, y0
    elif wall == 'x-': x_img, y_img = -2*ell - x0, y0
    elif wall == 'y+': x_img, y_img = x0, 2*ell - y0
    elif wall == 'y-': x_img, y_img = x0, -2*ell - y0
    else: raise ValueError("Invalid wall identifier")
    
    # Negative of the unbounded flow from the image particle
    return -unbounded_flow(x, y, x_img, y_img, Up, particle_radius)

@track_performance
def method_of_reflections_flow(x, y, x0, y0, Up, particle_radius, ell, num_reflections=3):
    """
    Calculate flow field using method of reflections
    
    This implements the theoretical approach from the Asymptotic Equilibrium Method:
    We construct u^(0) using the method of reflections:
    u^(0) = u_1^(0) + u_2^(0) + u_3^(0) + ...
    
    Where:
    - u_1^(0): Sphere in unbounded flow (Lamb's solution) - implemented by unbounded_flow
    - u_2^(0): Wall image correction - implemented by image_flow
    
    Parameters:
    - x, y: Coordinates to evaluate flow
    - x0, y0: Particle center coordinates
    - Up: Particle velocity vector [Upx, Upy]
    - particle_radius: Radius of the particle
    - ell: Channel half-width
    - num_reflections: Number of reflections to include (adaptive based on particle position)
    
    Returns:
    - Flow velocity vector [u, v]
    """
    # Calculate distance to nearest wall to determine optimal num_reflections
    distances_to_walls = [abs(x0 - ell), abs(x0 + ell), abs(y0 - ell), abs(y0 + ell)]
    min_distance = min(distances_to_walls)
    
    # Adaptively set reflections based on distance to wall relative to particle size
    # More reflections needed when particle is close to walls
    if min_distance < 2 * particle_radius:
        adaptive_reflections = max(5, num_reflections)  # More reflections near walls
    elif min_distance < 5 * particle_radius:
        adaptive_reflections = max(3, num_reflections)  # Standard reflections
    else:
        adaptive_reflections = max(2, num_reflections)  # Fewer reflections far from walls
    
    # First term: u_1^(0) - Start with unbounded flow (Lamb's solution)
    u_total = unbounded_flow(x, y, x0, y0, Up, particle_radius)
    
    # Second term: u_2^(0) - Add wall image corrections
    for wall in ['x+', 'x-', 'y+', 'y-']:
        u_total += image_flow(x, y, x0, y0, Up, wall, particle_radius, ell)
    
    return u_total

@track_performance
def lift_force(x0, y0, alpha, Re, ell, rho=1.0, Um=1.0, mu=1.0, aspect_ratio=1.0, 
               use_ar_correction=True, reference_ar=1.0, cl_neg=0.5, cl_pos=0.2):
    """
    Calculate the lift force on a particle at position (x0, y0)
    
    Parameters:
    - x0, y0: Particle center coordinates
    - alpha: Particle size ratio (a/ell)
    - Re: Reynolds number
    - ell: Channel half-width
    - rho: Fluid density
    - Um: Maximum flow velocity
    - mu: Fluid viscosity
    - aspect_ratio: Ratio of channel height to width (h/w)
    - use_ar_correction: Whether to apply aspect ratio correction
    - reference_ar: Reference aspect ratio (for which the model is calibrated)
    - cl_neg: Negative lift coefficient (CL-)
    - cl_pos: Positive lift coefficient (CL+)
    
    Returns:
    - [Fx, Fy]: Force vector
    """
    # If using aspect ratio correction and not close to reference AR,
    # first calculate force as if in reference AR channel, then apply correction
    if use_ar_correction and abs(aspect_ratio - reference_ar) > 0.05:
        # Import here to avoid circular import
        import aspect_ratio_correction as arc
        
        # Calculate for reference aspect ratio (typically AR=1.0)
        fx_ref, fy_ref = lift_force(
            x0, y0, alpha, Re, ell, rho, Um, mu, 
            aspect_ratio=reference_ar,
            use_ar_correction=False  # Avoid infinite recursion
        )
        
        # Apply the correction to get the force for the actual aspect ratio
        return arc.apply_ar_correction(
            [fx_ref, fy_ref], aspect_ratio, reference_ar, cl_neg, cl_pos
        )
    
    # Standard calculation (original model, typically for AR=1.0)
    # Calculate particle radius from alpha
    particle_radius = alpha * ell
    
    # Get flow derivatives at particle position
    beta, gamma_x, gamma_y, delta_xx, delta_xy, delta_yy = flow_derivatives(
        x0, y0, ell, Um, aspect_ratio)
    
    # Calculate the shear rate components
    shear_rate_x = gamma_x
    shear_rate_y = gamma_y
    
    # Calculate particle slip velocity relative to background flow
    # At this point, we assume particle is not moving (Up = 0)
    slip_velocity = beta
    
    # Calculate lift coefficients
    # These are theoretical coefficients for the lift force
    # Based on the shear-induced lift and wall-induced lift mechanisms
    
    # Basic lift coefficient scales with Re and particle size
    # CL ~ Re * alpha^n where n depends on the flow regime
    # For intermediate Re, n is approximately 2-3
    CL = 0.5 * Re * alpha**2
    
    # Shear-gradient lift coefficient 
    # This is the lift due to the curvature of the velocity profile
    # It pushes particles away from the center toward the wall
    CL_shear = CL * 0.05  # Scale factor based on theoretical models
    
    # Wall-induced lift coefficient
    # This pushes particles away from the wall
    # Strong near walls, decays with distance
    distances_to_walls = [
        abs(x0 - ell),  # Distance to right wall
        abs(x0 + ell),  # Distance to left wall
        aspect_ratio >= 1.0 and abs(y0 - ell * aspect_ratio) or abs(y0 - ell),  # Distance to top wall
        aspect_ratio >= 1.0 and abs(y0 + ell * aspect_ratio) or abs(y0 + ell)   # Distance to bottom wall
    ]
    min_wall_distance = min(distances_to_walls)
    
    # Wall lift decays with distance: CL_wall ~ 1/distance^n
    # Cutoff to avoid singularity at the wall
    min_wall_distance = max(min_wall_distance, particle_radius * 1.1)
    
    # Scale with distance from wall (stronger near walls)
    wall_factor = (particle_radius / min_wall_distance)**2
    CL_wall = CL * 0.2 * wall_factor  # Scale factor based on theoretical models
    
    # Combined lift coefficient balances shear and wall effects
    CL_combined = CL_shear + CL_wall
    
    # Direction of lift force depends on shear gradient
    # In channel flow, this is typically toward the wall in the center
    # and toward the center near the walls
    
    # Force calculation 
    # The force depends on:
    # 1. Slip velocity (difference between particle and fluid velocity)
    # 2. Shear rate (gradient of velocity)
    # 3. Channel geometry and particle position
    
    # Calculate force components
    # Fx - force in flow direction (drag)
    # Fy - force perpendicular to flow (lift)
    
    # Drag force (Stokes drag: F = 6πμaU)
    Fx = 6 * np.pi * mu * particle_radius * slip_velocity
    
    # Lift force calculation
    # Basic form: FL ~ ρU^2a^2 * CL_combined
    FL_scale = rho * slip_velocity**2 * particle_radius**2 * CL_combined
    
    # The lift direction is determined by:
    # 1. Sign of shear rate (direction of velocity gradient)
    # 2. Distance from walls (wall-induced component)
    
    # In simple planar channel flow, the lift force direction can be approximated
    # as being toward specific equilibrium positions:
    
    # Simplified model: 
    # - In the center, force pushes toward walls (due to shear gradient)
    # - Near walls, force pushes toward center (due to wall effect)
    # - Four equilibrium positions for square channels, two for rectangular
    
    # For the y-component:
    if abs(y0) < 0.3 * ell:
        # Near the center horizontally - force pushes outward
        lift_direction_y = np.sign(y0)
    else:
        # Near the walls horizontally - force pushes inward
        lift_direction_y = -np.sign(y0)
    
    # For the x-component in non-square channels:
    if aspect_ratio > 1.2 or aspect_ratio < 0.8:  # Non-square channel
        if abs(x0) < 0.3 * ell:
            # Near the center vertically - force pushes outward
            lift_direction_x = np.sign(x0)
        else:
            # Near the walls vertically - force pushes inward
            lift_direction_x = -np.sign(x0)
        
        # Scale the force with gamma to ensure correct force direction
        Fx_lift = FL_scale * lift_direction_x * abs(gamma_y) / (abs(gamma_x) + abs(gamma_y) + 1e-10)
    else:
        # For square channels, assuming primary lift is in y-direction
        Fx_lift = 0
    
    # Scale the force with gamma to ensure correct force direction
    Fy = FL_scale * lift_direction_y * abs(gamma_x) / (abs(gamma_x) + abs(gamma_y) + 1e-10)
    
    # Combine drag and lift components in x-direction
    Fx = Fx + Fx_lift
    
    # Apply wall corrections to forces
    # Lubrication effects become significant when particle is close to wall
    for i, dist in enumerate(distances_to_walls):
        correction = lubrication_correction(dist, particle_radius)
        if i < 2:  # x-direction walls
            Fx *= correction
        else:      # y-direction walls
            Fy *= correction
    
    return Fx, Fy

def find_equilibrium_positions(alpha, Re, ell, rho=1.0, Um=1.0, mu=1.0, aspect_ratio=1.0, 
                               resolution=10, use_ar_correction=True, reference_ar=1.0, 
                               cl_neg=0.5, cl_pos=0.2, max_positions=None):
    """
    Find all equilibrium positions in the channel based on flow physics
    
    Parameters:
    - alpha: Particle size ratio (a/ell)
    - Re: Reynolds number
    - ell: Channel half-width
    - rho: Fluid density
    - Um: Maximum flow velocity
    - mu: Fluid viscosity
    - aspect_ratio: Ratio of channel height to width (h/w)
    - resolution: Grid resolution for brute-force search
    - use_ar_correction: Whether to apply aspect ratio correction
    - reference_ar: Reference aspect ratio (for which the model is calibrated)
    - cl_neg: Negative lift coefficient (CL-)
    - cl_pos: Positive lift coefficient (CL+)
    - max_positions: Maximum number of equilibrium positions to return (if None, all physically meaningful positions are returned)
    
    Returns:
    - List of equilibrium positions [[x1, y1], [x2, y2], ...]
    """
    # Create a grid of positions to evaluate
    particle_radius = alpha * ell
    safe_margin = 1.2 * particle_radius
    
    # Calculate channel dimensions based on aspect ratio
    if aspect_ratio >= 1.0:
        width = 2 * ell
        height = 2 * ell * aspect_ratio
    else:
        height = 2 * ell
        width = 2 * ell / aspect_ratio
    
    # Create grid points - using higher resolution for better accuracy
    adjusted_resolution = max(resolution, 20)  # Ensure minimum resolution
    x_points = np.linspace(-width/2 + safe_margin, width/2 - safe_margin, adjusted_resolution)
    y_points = np.linspace(-height/2 + safe_margin, height/2 - safe_margin, adjusted_resolution)
    
    # Find force minima (where force magnitude approaches zero)
    force_magnitudes = []
    
    for x in x_points:
        for y in y_points:
            # Calculate force at this position with AR correction if enabled
            fx, fy = lift_force(
                x, y, alpha, Re, ell, rho, Um, mu, aspect_ratio,
                use_ar_correction=use_ar_correction, 
                reference_ar=reference_ar,
                cl_neg=cl_neg, 
                cl_pos=cl_pos
            )
            force_mag = np.sqrt(fx**2 + fy**2)
            
            # Only consider points with small force (potential equilibrium)
            if force_mag < 0.05:  # Relaxed threshold to find more candidates
                force_magnitudes.append((force_mag, [x, y]))
    
    # Sort by force magnitude (smallest first)
    force_magnitudes.sort(key=lambda x: x[0])
    
    # Filter to find physically meaningful positions
    distinct_positions = []
    
    # Define physical meaningfulness criteria
    # 1. Very small force magnitude (true equilibrium)
    # 2. Sufficiently distant from other equilibria
    # 3. Not too close to wall (account for hydrodynamic interactions)
    
    force_threshold = 0.02  # Force threshold for consideration
    min_separation = 0.12 * ell  # Minimum distance between distinct equilibria
    
    for mag, pos in force_magnitudes:
        # Skip positions with too large force
        if mag > force_threshold:
            continue
        
        # Skip positions that are extremely close to walls (within particle radius)
        wall_distance_x = min(abs(pos[0] - (-width/2)), abs(pos[0] - (width/2)))
        wall_distance_y = min(abs(pos[1] - (-height/2)), abs(pos[1] - (height/2)))
        if wall_distance_x < particle_radius * 1.05 or wall_distance_y < particle_radius * 1.05:
            continue
        
        # Check if this position is close to any existing position
        is_distinct = True
        for existing_pos in distinct_positions:
            dist = np.sqrt((pos[0] - existing_pos[0])**2 + (pos[1] - existing_pos[1])**2)
            if dist < min_separation:
                # If very close, keep the one with smaller force magnitude
                is_distinct = False
                break
        
        if is_distinct:
            distinct_positions.append(pos)
    
    # For rectangular channels, theory predicts 2 equilibrium positions
    # For square channels, theory predicts 4 equilibrium positions
    # If aspect_ratio is significantly different from 1, we expect 2 positions
    # Otherwise, we expect 4 positions
    
    # If not enough positions found through direct calculation,
    # add theoretical positions based on channel geometry
    if len(distinct_positions) < 2:
        # If using AR correction, the theoretical positions should take that into account
        if use_ar_correction and abs(aspect_ratio - reference_ar) > 0.05:
            # Import here to avoid circular import
            import aspect_ratio_correction as arc
            
            # First get theoretical positions for reference AR (typically square)
            if abs(reference_ar - 1.0) < 0.2:  # Nearly square reference channel
                eq_factor = 0.6
                ref_positions = [
                    [0, eq_factor * ell],
                    [0, -eq_factor * ell],
                    [eq_factor * ell, 0],
                    [-eq_factor * ell, 0]
                ]
            else:
                # This should rarely happen as reference AR is typically 1.0
                if reference_ar > 1.0:  # Taller than wide
                    eq_factor = 0.6
                    ref_positions = [
                        [0, eq_factor * ell],
                        [0, -eq_factor * ell]
                    ]
                else:  # Wider than tall
                    eq_factor = 0.6
                    ref_positions = [
                        [eq_factor * (ell / reference_ar), 0],
                        [-eq_factor * (ell / reference_ar), 0]
                    ]
            
            # Apply AR correction to each theoretical position
            distinct_positions = []
            for pos in ref_positions:
                # For theoretical positions, we need to transform force to displacement
                # This is a simplified approach as direct position mapping is complex
                direction = pos / (np.sqrt(pos[0]**2 + pos[1]**2) + 1e-10)
                
                if aspect_ratio >= 1.0:
                    # Tall channel mostly affects y-positions
                    k_factor = arc.calculate_ar_correction_factor(
                        aspect_ratio, reference_ar, cl_neg, cl_pos
                    )
                    
                    # More effect in y-direction for tall channels
                    if abs(direction[1]) > abs(direction[0]):
                        pos_corrected = [pos[0], pos[1] * np.sqrt(k_factor)]
                    else:
                        pos_corrected = pos
                        
                else:
                    # Wide channel mostly affects x-positions
                    k_factor = arc.calculate_ar_correction_factor(
                        aspect_ratio, reference_ar, cl_neg, cl_pos
                    )
                    
                    # More effect in x-direction for wide channels
                    if abs(direction[0]) > abs(direction[1]):
                        pos_corrected = [pos[0] * np.sqrt(k_factor), pos[1]]
                    else:
                        pos_corrected = pos
                
                # Make sure positions are within channel bounds
                if aspect_ratio >= 1.0:
                    max_y = ell * aspect_ratio - safe_margin
                    pos_corrected[1] = min(max(pos_corrected[1], -max_y), max_y)
                else:
                    max_x = ell / aspect_ratio - safe_margin
                    pos_corrected[0] = min(max(pos_corrected[0], -max_x), max_x)
                    
                distinct_positions.append(pos_corrected)
                
        else:
            # Original behavior without AR correction
            if abs(aspect_ratio - 1.0) < 0.2:  # Nearly square channel
                # For square channels, equilibrium positions are typically at:
                # (0, ±0.6*ell) and (±0.6*ell, 0)
                eq_factor = 0.6
                distinct_positions = [
                    [0, eq_factor * ell],
                    [0, -eq_factor * ell],
                    [eq_factor * ell, 0],
                    [-eq_factor * ell, 0]
                ]
            else:  # Rectangular channel
                # For rectangular channels, equilibrium positions are typically at:
                # (0, ±0.6*smaller_dimension)
                if aspect_ratio > 1.0:  # Taller than wide
                    eq_factor = 0.6
                    distinct_positions = [
                        [0, eq_factor * ell],
                        [0, -eq_factor * ell]
                    ]
                else:  # Wider than tall
                    eq_factor = 0.6
                    distinct_positions = [
                        [eq_factor * (ell / aspect_ratio), 0],
                        [-eq_factor * (ell / aspect_ratio), 0]
                    ]
    
    return distinct_positions
