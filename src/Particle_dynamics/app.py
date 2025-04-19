import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

# Import local modules
import particle_solver as solver
import particle_physics as pp
import particle_migration as migration
import parameter_sweep as sweep
import visualization as viz
import utils
import stability_analysis as stability

# Set page configuration
st.set_page_config(
    page_title="Particle Dynamics Simulator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application title and description
st.title("Particle Dynamics in Microfluidic Channels")
st.markdown("""
This application simulates the dynamics of particles in microfluidic channels, 
calculating equilibrium positions, flow fields, and particle migration trajectories.
It models inertial focusing phenomena observed in microfluidic systems.
""")

# Create an expander for the additional physics information
with st.expander("Learn About Inertial Focusing Physics"):
    st.markdown("""
    ## Inertial Focusing in Microchannels
    
    Inertial focusing is a passive particle manipulation technique that relies on inertial forces 
    in microchannels to migrate particles to specific equilibrium positions.
    
    ### Physical Mechanism
    
    When particles flow through microchannels, they experience several forces:
    
    1. **Shear-Induced Lift Force** (Fs): Acts down the shear gradient toward the channel walls
    2. **Wall-Induced Lift Force** (Fw): Pushes particles away from walls toward the channel center
    3. **Rotation-Induced Lift Force**: Affects focusing behavior in non-circular channels
    4. **Drag Force**: Acts parallel to the flow direction
    
    Particles reach equilibrium positions where these forces balance. The focusing behavior typically occurs in two stages:
    - First, particles migrate to the wall-centered planes
    - Then, they migrate along those planes to their final equilibrium positions
    
    ### Applications
    
    Inertial focusing has numerous applications in:
    - Cell/particle separation
    - Flow cytometry
    - Sample preparation
    - Filtration
    - Cell enrichment
    
    ### Limitations of This Model
    
    While this simulation captures key aspects of inertial focusing, it has several limitations:
    
    1. **Reynolds Number Range**: Optimized for low to moderate Reynolds numbers (Re < 100)
    2. **Particle Size Effects**: Different scaling laws may apply for different particle size regimes
    3. **Simplified Forces**: Some complex force interactions may be simplified
    4. **Channel Geometry**: Optimized for rectangular channels; curved channels exhibit additional Dean flows
    5. **Particle Interactions**: Model assumes dilute suspensions with no particle-particle interactions
    6. **Neutrally Buoyant Assumption**: Particles are assumed to have the same density as the fluid
    7. **Near-Wall Corrections**: Lubrication effects near walls are approximated
    
    ### Relevant Dimensionless Numbers
    
    - **Reynolds Number (Re)** = ρUDh/μ: Ratio of inertial to viscous forces
    - **Particle Reynolds Number (Rep)** = Re × (a/Dh)²: Characterizes particle-scale inertial effects
    - **Aspect Ratio**: Ratio of channel height to width, affects focusing pattern
    """)

# Sidebar for parameters
st.sidebar.header("Simulation Parameters")

# Add units information at the top
st.sidebar.info("""
**About Units:**
All parameters are in dimensionless form. This means:
- Length scales are relative to channel width (W)
- Velocities are relative to maximum flow velocity (Um)
- Forces are scaled by ρUm²W² (where ρ is fluid density)
- Reynolds number is defined as Re = ρUmW/μ (where μ is viscosity)
""")

# Parameter inputs
with st.sidebar:
    st.subheader("Flow Parameters")
    channel_width_um = st.number_input(
        "Channel Width (μm)", 
        min_value=10.0,
        max_value=1000.0,
        value=100.0,
        step=10.0,
        help="Width of the channel in micrometers"
    )
    
    height_um = st.number_input(
        "Channel Height (μm)",
        min_value=10.0,
        max_value=1000.0,
        value=100.0,
        step=10.0,
        help="Height of the channel in micrometers"
    )
    
    aspect_ratio = height_um / channel_width_um
    st.write(f"Aspect Ratio: {aspect_ratio:.2f}")
    
    Um_mps = st.number_input(
        "Maximum Flow Velocity (m/s)",
        min_value=0.001,
        max_value=10.0,
        value=0.1,
        step=0.001,
        format="%.3f",
        help="Maximum velocity of the background flow"
    )
    
    # Convert to simulation units
    channel_width = 1.0  # Reference length
    Um = 1.0  # Reference velocity
    
    # Fluid properties inputs
    st.write("Fluid Properties:")
    density = st.number_input(
        "Fluid Density (kg/m³)",
        min_value=1.0,
        max_value=2000.0,
        value=1000.0,
        step=1.0,
        help="Density of the fluid in kg/m³"
    )
    
    viscosity = st.number_input(
        "Fluid Viscosity (Pa·s)",
        min_value=1e-4,
        max_value=1.0,
        value=1e-3,
        format="%.6f",
        step=1e-4,
        help="Dynamic viscosity of the fluid in Pa·s"
    )
    
    # Calculate and display Reynolds number
    Re_calculated = density * Um_mps * channel_width_um * 1e-6 / viscosity
    st.write(f"Calculated Reynolds number: {Re_calculated:.2f}")

with st.sidebar.expander("Particle Parameters", expanded=True):
    num_particles = st.number_input(
        "Number of Particles",
        min_value=1, 
        max_value=10, 
        value=1,
        help="Number of particles to simulate"
    )
    
    particle_radius = st.number_input(
        "Particle Radius (relative to channel width)", 
        min_value=0.01, 
        max_value=0.45, 
        value=0.1, 
        step=0.01,
        help="Radius of the particle relative to channel width. Must be smaller than half the channel width."
    )
    
    # Add visualization of particle size relative to channel
    st.caption("Particle size relative to channel:")
    size_ratio = particle_radius / 0.5
    st.progress(size_ratio)
    st.caption(f"Particle occupies {size_ratio*100:.1f}% of channel half-width")
    
    # Calculate alpha (particle size ratio)
    alpha = particle_radius / 0.5  # a/ell

with st.sidebar.expander("Solver Parameters", expanded=True):
    search_resolution = st.slider(
        "Search Resolution", 
        min_value=5, 
        max_value=30, 
        value=10, 
        step=5,
        help="Number of initial positions to check in each dimension. Higher values give more accurate results but take longer."
    )
    
    grid_resolution = st.slider(
        "Grid Resolution for Flow Field", 
        min_value=10, 
        max_value=50, 
        value=30, 
        step=5,
        help="Resolution of the calculation grid for flow visualization. Higher values show more detail but require more computation."
    )
    
    # Add option to search entire channel
    search_entire_channel = st.checkbox(
        "Search Entire Channel", 
        value=True,
        help="When enabled, searches all four quadrants of the channel. When disabled, uses symmetry to search only one quadrant (faster)."
    )
    
    # Note about equilibrium positions
    st.info("""
    **About Equilibrium Positions**: 
    The simulation will find all physically meaningful equilibrium positions based on:
    1. Points where force magnitude approaches zero
    2. Points sufficiently distant from other equilibria
    3. Positions not too close to walls (hydrodynamic considerations)
    
    For rectangular channels, theory predicts 2 primary positions;
    for square channels, theory predicts 4 positions.
    """)
    
    migration_simulation_time = st.slider(
        "Migration Simulation Time", 
        min_value=5.0, 
        max_value=1000.0,  # Changed from 100 to 1000
        value=50.0, 
        step=5.0,
        help="Maximum time for particle migration simulation."
    )
    
    convergence_threshold = st.number_input(
        "Convergence Threshold", 
        min_value=1e-8, 
        max_value=1e-2, 
        value=1e-6, 
        format="%.1e",
        help="Velocity magnitude threshold to consider a particle converged to equilibrium."
    )

with st.sidebar.expander("Aspect Ratio Correction Parameters", expanded=True):
    st.markdown("""
    ### Zhou & Papautsky Correction
    Parameters for aspect ratio correction based on Zhou & Papautsky (2013) two-stage focusing model.
    Controls the scaling of lift forces for non-square channels (AR≠1).
    """)
    
    use_ar_correction = st.checkbox(
        "Apply Aspect Ratio Correction", 
        value=True,
        help="Apply Zhou & Papautsky correction to scale forces for non-square channels."
    )
    
    cl_neg = st.number_input(
        "CL- (Negative Lift Coefficient)", 
        min_value=0.01, 
        max_value=10.0, 
        value=0.5, 
        step=0.01,
        help="Coefficient for wall-migration forces (Stage I focusing)"
    )
    
    cl_pos = st.number_input(
        "CL+ (Positive Lift Coefficient)", 
        min_value=0.01, 
        max_value=10.0, 
        value=0.2, 
        step=0.01,
        help="Coefficient for rotation-induced forces (Stage II focusing)"
    )
    
    reference_ar = st.number_input(
        "Reference Aspect Ratio", 
        min_value=0.1, 
        max_value=10.0, 
        value=1.0, 
        step=0.1,
        help="Reference aspect ratio for which the simulator is calibrated"
    )

# Convert channel width to half-width (ell)
ell = 0.5  # channel_width / 2.0 (already normalized)

# Validate parameters
is_valid, error_message = utils.validate_parameters(particle_radius, ell, aspect_ratio)
if not is_valid:
    st.error(f"Invalid parameters: {error_message}")
    st.stop()

# Create tabs for different visualizations
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Equilibrium Position", 
    "Flow Field", 
    "Force Contours",
    "Background Flow", 
    "Particle Migration",
    "Parameter Sweep"
])

# Button to run simulation
if st.sidebar.button("Calculate Equilibrium Position", type="primary"):
    with st.spinner("Calculating equilibrium position..."):
        try:
            # Find equilibrium position
            start_time = time.time()
            eq_position, force_field = solver.find_equilibrium_position(
                particle_radius, Um, ell, aspect_ratio, 
                search_resolution=search_resolution,
                search_entire_channel=search_entire_channel,
                use_ar_correction=use_ar_correction,
                reference_ar=reference_ar,
                cl_neg=cl_neg,
                cl_pos=cl_pos
            )
            x_eq, y_eq = eq_position
            calc_time = time.time() - start_time
            
            # Store results in session state
            st.session_state.equilibrium_found = True
            st.session_state.x_eq = x_eq
            st.session_state.y_eq = y_eq
            st.session_state.force_field = force_field
            st.session_state.calc_time = calc_time
            st.session_state.ar_correction_used = use_ar_correction
            st.session_state.ar_correction_params = {
                "reference_ar": reference_ar,
                "cl_neg": cl_neg,
                "cl_pos": cl_pos
            }
            
            # Calculate flow field around equilibrium position
            with st.spinner("Calculating flow field..."):
                flow_field = solver.calculate_flow_field(
                    x_eq, y_eq, particle_radius, ell, Um, 
                    aspect_ratio, grid_resolution
                )
            st.session_state.flow_field = flow_field
            
            # Setup the particle migration simulator for trajectory calculations
            with st.spinner("Setting up migration simulator..."):
                migration_simulator = migration.ParticleMigrationSimulator(
                    alpha, Re_calculated, ell, density, Um, viscosity, aspect_ratio
                )
                
                # Run migration simulation for a single particle
                progress_bar, progress_text, update_progress = utils.create_progress_bar(100, "Simulating particle migration")
                
                initial_pos = [0.0, 0.3]  # Start position away from equilibrium
                trajectory = migration_simulator.simulate_migration(
                    initial_pos, 
                    t_max=migration_simulation_time,
                    convergence_threshold=convergence_threshold
                )
                
                progress_bar.progress(1.0)
                progress_text.text("Simulation complete!")
                
                st.session_state.migration_simulator = migration_simulator
                st.session_state.trajectory = trajectory
            
        except Exception as e:
            st.error(f"Error calculating equilibrium: {e}")
            st.session_state.equilibrium_found = False

# Run parameter sweep if requested
if st.sidebar.button("Run Particle Radius Sweep"):
    num_points = st.sidebar.slider(
        "Number of Points", 
        min_value=3, 
        max_value=15, 
        value=8,
        help="Number of particle radius values to test"
    )
    
    with st.spinner("Running particle radius sweep..."):
        # Sweep particle radius from 0.05 to 0.4
        param_values = np.linspace(0.05, 0.4, num_points)
        sweep_results = sweep.parameter_sweep(
            "particle_radius", param_values, particle_radius, Um, ell, aspect_ratio,
            search_resolution=search_resolution
        )
        
        st.session_state.sweep_results = sweep_results

# Display results if equilibrium was found
if 'equilibrium_found' in st.session_state and st.session_state.equilibrium_found:
    # Get stored results
    x_eq = st.session_state.x_eq
    y_eq = st.session_state.y_eq
    force_field = st.session_state.force_field
    calc_time = st.session_state.calc_time
    flow_field = st.session_state.flow_field
    
    # Display equilibrium position tab
    with tab1:
        st.subheader("Equilibrium Position")
        st.write(f"Equilibrium position found at: x = {x_eq:.4f}, y = {y_eq:.4f}")
        st.write(f"Calculation time: {calc_time:.2f} seconds")
        
        # Plot equilibrium position
        fig_eq = viz.plot_equilibrium_position(x_eq, y_eq, particle_radius, ell, aspect_ratio, force_field)
        st.pyplot(fig_eq)
        
        # Create the stability analysis if not already done
        if 'stability_results' not in st.session_state:
            with st.spinner("Performing stability analysis..."):
                # Perform stability analysis
                stability_results = stability.analyze_stability(
                    [x_eq, y_eq], particle_radius, Um, ell, 
                    aspect_ratio=aspect_ratio,
                    use_ar_correction=use_ar_correction,
                    reference_ar=reference_ar,
                    cl_neg=cl_neg,
                    cl_pos=cl_pos
                )
                st.session_state.stability_results = stability_results
        else:
            stability_results = st.session_state.stability_results
        
        # Filter for stable positions only
        stable_positions = [result for result in stability_results if result["stability_type"] == "stable"]
        
        # Show summary statistics
        st.write(f"Total equilibrium positions found: **{len(stability_results)}**")
        st.write(f"Stable equilibrium positions: **{len(stable_positions)}**")
        
        # Display normalized position
        if aspect_ratio >= 1.0:
            x_norm = x_eq / ell
            y_norm = y_eq / (ell * aspect_ratio)
        else:
            x_norm = x_eq / (ell / aspect_ratio)
            y_norm = y_eq / ell
            
        st.write(f"Normalized position: x/width = {x_norm:.4f}, y/height = {y_norm:.4f}")
        
        # Add stability analysis
        st.subheader("Stability Analysis")
        
        with st.spinner("Performing stability analysis..."):
            # Find all potential equilibrium positions
            try:
                # Find all equilibrium positions
                equilibrium_positions = pp.find_equilibrium_positions(
                    alpha, Re_calculated, ell, density, Um, viscosity, aspect_ratio,
                    resolution=search_resolution,
                    use_ar_correction=use_ar_correction,
                    reference_ar=reference_ar,
                    cl_neg=cl_neg,
                    cl_pos=cl_pos
                )
                
                # Analyze stability of all equilibrium positions
                stability_results = stability.analyze_equilibrium_stability(
                    equilibrium_positions, alpha, Re_calculated, ell, 
                    density, Um, viscosity, aspect_ratio,
                    use_ar_correction=use_ar_correction,
                    reference_ar=reference_ar,
                    cl_neg=cl_neg,
                    cl_pos=cl_pos
                )
                
                # Display stability results
                st.write("### Equilibrium Positions and Stability:")
                
                for i, result in enumerate(stability_results):
                    pos = result["position"]
                    norm_pos = result["normalized_position"]
                    stability_type = result["stability_type"]
                    
                    # Highlight the current equilibrium position
                    is_current = np.allclose(np.array(pos), np.array([x_eq, y_eq]), atol=0.05)
                    prefix = "✓" if is_current else ""
                    highlight = "**" if is_current else ""
                    
                    # Choose color based on stability type
                    if stability_type == "stable":
                        color = "green"
                    elif stability_type == "saddle":
                        color = "orange"
                    elif stability_type == "unstable":
                        color = "red"
                    else:
                        color = "gray"
                    
                    # Get group size if available
                    group_size = result.get("group_size", 1)
                    group_info = f" (representing {group_size} nearby points)" if group_size > 1 else ""
                    
                    # Create description with position, normalized position, and stability
                    st.markdown(
                        f"{prefix} {highlight}Position {i+1}{highlight}: x,y = [{pos[0]:.4f}, {pos[1]:.4f}], "
                        f"normalized = [{norm_pos[0]:.4f}, {norm_pos[1]:.4f}], "
                        f"stability: :{color}[{stability_type}]{group_info}"
                    )
                    
                    # Show the individual points in the group if there are multiple
                    if group_size > 1 and st.checkbox(f"Show all {group_size} points in group {i+1}", value=False):
                        original_positions = result.get("original_positions", [])
                        st.write("Individual points in this group:")
                        for j, orig_pos in enumerate(original_positions):
                            # Calculate normalized position for each original position
                            if aspect_ratio >= 1.0:
                                orig_x_norm = orig_pos[0] / ell
                                orig_y_norm = orig_pos[1] / (ell * aspect_ratio)
                            else:
                                orig_x_norm = orig_pos[0] / (ell / aspect_ratio)
                                orig_y_norm = orig_pos[1] / ell
                                
                            st.write(f"  Point {j+1}: x,y = [{orig_pos[0]:.4f}, {orig_pos[1]:.4f}], "
                                    f"normalized = [{orig_x_norm:.4f}, {orig_y_norm:.4f}]")
                    
                    # Show eigenvalues for the tech-savvy users
                    if stability_type != "undetermined" and st.checkbox(f"Show eigenvalues for Position {i+1}", value=False):
                        eigenvalues = result["eigenvalues"]
                        st.write(f"Eigenvalues: [{eigenvalues[0]:.4e}, {eigenvalues[1]:.4e}]")
                        
                        # Show Hessian matrix
                        st.write("Hessian matrix:")
                        hessian = result["hessian"]
                        st.write(pd.DataFrame(hessian, columns=["d²/dx²", "d²/dxdy"], index=["d²/dx²", "d²/dydx"]))
                
                # Add explanation
                st.info("""
                **Stability Analysis**: 
                - **Stable** equilibrium positions (green) are where particles will naturally settle
                - **Saddle** points (orange) are stable in one direction but unstable in another
                - **Unstable** positions (red) will repel particles
                
                The stability is determined using the Hessian matrix of the force potential, which is the mathematical equivalent of analyzing the "energy landscape" in the channel.
                """)
                
            except Exception as e:
                st.error(f"Error in stability analysis: {e}")
                st.warning("Couldn't perform full stability analysis. Try increasing the search resolution or adjusting parameters.")
        
        # Display force field statistics
        if force_field and len(force_field['x']) > 0:
            magnitudes = np.array(force_field['magnitude'])
            st.write(f"Maximum force magnitude: {np.max(magnitudes):.4e}")
            st.write(f"Minimum force magnitude: {np.min(magnitudes):.4e}")
            
            # Display histogram of force magnitudes
            fig, ax = plt.subplots()
            ax.hist(magnitudes, bins=20)
            ax.set_xlabel('Force Magnitude')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Force Magnitudes')
            ax.grid(True)
            st.pyplot(fig)
    
    # Display flow field tab
    with tab2:
        st.subheader("Flow Field Around Particle")
        
        # Plot flow field
        fig_flow = viz.plot_flow_field(flow_field, [x_eq, y_eq], particle_radius, ell, aspect_ratio)
        st.pyplot(fig_flow)
        
        # Display flow field statistics
        speed = np.sqrt(flow_field['U']**2 + flow_field['V']**2)
        st.write(f"Maximum flow speed: {np.max(speed):.4f}")
        st.write(f"Minimum flow speed: {np.min(speed):.4f}")
        
        # Create a slider to show the flow at different y-positions
        # Calculate min, max, and default values with proper step alignment
        y_min = float(flow_field['Y'].min())
        y_max = float(flow_field['Y'].max())
        y_default = float(y_eq)
        step = 0.01
        
        # Ensure values align with step
        y_min = np.floor(y_min / step) * step
        y_max = np.ceil(y_max / step) * step
        y_default = np.round(y_default / step) * step
        
        y_slice = st.slider("Y position for x-direction velocity profile", 
                           min_value=y_min, max_value=y_max, 
                           value=y_default, step=step)
        
        # Find the closest y-index
        y_indices = np.abs(flow_field['Y'][:, 0] - y_slice).argmin()
        
        # Plot horizontal velocity profile at selected y-position
        fig, ax = plt.subplots()
        ax.plot(flow_field['X'][y_indices, :], flow_field['U'][y_indices, :], 'b-', label='Total flow')
        ax.plot(flow_field['X'][y_indices, :], flow_field['U_background'][y_indices, :], 'g--', label='Background flow')
        ax.axvline(x=x_eq, color='r', linestyle='--', label='Particle position')
        ax.set_xlabel('x position')
        ax.set_ylabel('Velocity')
        ax.set_title(f'Velocity Profile at y = {y_slice:.3f}')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
    
    # Display force contours tab
    with tab3:
        st.subheader("Force Contours")
        
        # Create a uniform grid for force calculation
        if 'force_field' in st.session_state and st.session_state.force_field:
            force_field = st.session_state.force_field
            
            # Check if we have enough force data points for contour plot
            if len(force_field['x']) >= 16:  # Need at least a 4x4 grid
                # Extract unique x and y points
                x_unique = np.unique(np.array(force_field['x']))
                y_unique = np.unique(np.array(force_field['y']))
                
                if len(x_unique) >= 3 and len(y_unique) >= 3:
                    # Reshape the force magnitudes into a grid
                    x_grid, y_grid = np.meshgrid(x_unique, y_unique)
                    
                    # Get force magnitudes in correct format for contour plot
                    force_mag = np.array(force_field['magnitude'])
                    
                    # Plot contours
                    fig_contour = viz.plot_force_contours(
                        x_unique, y_unique, force_mag, 
                        aspect_ratio, ell, particle_radius, 
                        equilibrium_position=[x_eq, y_eq]
                    )
                    st.pyplot(fig_contour)
                    
                    # Add explanatory text
                    st.write("""
                    The contour plot shows the magnitude of forces on a particle throughout the channel.
                    Darker regions represent areas with lower forces, where particles are likely to equilibrate.
                    The equilibrium position is marked with a red circle.
                    """)
                else:
                    st.warning("Not enough unique x and y points for contour plotting.")
            else:
                st.warning("Insufficient force field data for contour plotting. Try increasing search resolution.")
        else:
            st.info("Force field data not available. Run the calculation first.")
    
    # Display background flow tab
    with tab4:
        st.subheader("Background Flow Profile")
        
        # Plot background flow
        fig_bg = viz.plot_background_flow(ell, aspect_ratio, resolution=grid_resolution, Um=Um)
        st.pyplot(fig_bg)
        
        # Add explanation
        st.write("""
        This shows the background flow velocity distribution in the channel without particles.
        For pressure-driven flow in a rectangular channel, the velocity follows a parabolic profile,
        with maximum velocity at the center and zero velocity at the walls due to the no-slip condition.
        """)
        
        # Add interactive exploration of background flow
        st.subheader("Explore Background Flow")
        col1, col2 = st.columns(2)
        
        x_point = col1.slider("X coordinate", -ell, ell, 0.0, 0.01)
        y_point = col2.slider("Y coordinate", -ell * aspect_ratio, ell * aspect_ratio, 0.0, 0.01)
        
        # Calculate flow at the selected point
        flow_at_point = pp.background_flow(x_point, y_point, ell, Um, aspect_ratio)
        gradient = pp.grad_background_flow(x_point, y_point, ell, Um, aspect_ratio)
        
        st.write(f"Flow velocity at ({x_point:.3f}, {y_point:.3f}): {flow_at_point:.4f}")
        st.write(f"Velocity gradient at this point: ∂u/∂x = {gradient[0]:.4f}, ∂u/∂y = {gradient[1]:.4f}")
            
    # Display particle migration tab
    with tab5:
        st.subheader("Particle Migration Simulation")
        
        if 'migration_simulator' in st.session_state and 'trajectory' in st.session_state:
            simulator = st.session_state.migration_simulator
            trajectory = st.session_state.trajectory
            
            # Display migration results
            if trajectory:
                # Show convergence status
                if trajectory['converged']:
                    st.success(f"Particle converged in {trajectory['convergence_time']:.2f} time units")
                else:
                    st.warning(f"Particle did not converge within the simulation time ({migration_simulation_time} time units)")
                
                # Plot complete migration results
                fig_migration = viz.plot_migration_results(simulator, trajectory)
                st.pyplot(fig_migration)
                
                # Option to show animation
                if st.checkbox("Show Migration Animation", value=False):
                    st.write("Animation of particle migration:")
                    animation_speed = st.slider("Animation Speed", 5, 50, 20, 5)
                    fig, anim = simulator.visualize_migration(trajectory, animation_speed=animation_speed)
                    
                    # Use st.pyplot for the static figure part
                    st.pyplot(fig)
                    st.info("Note: The interactive animation is displayed statically in this view. For a full animation, you can download and run the application locally.")
                
                # Show final position and compare to calculated equilibrium
                st.subheader("Migration Analysis")
                col1, col2 = st.columns(2)
                
                initial_pos = trajectory['initial_position']
                final_pos = trajectory['final_position']
                
                col1.metric("Initial x position", f"{initial_pos[0]:.4f}", 
                           f"{final_pos[0] - initial_pos[0]:.4f}")
                col1.metric("Initial y position", f"{initial_pos[1]:.4f}", 
                           f"{final_pos[1] - initial_pos[1]:.4f}")
                
                col2.metric("Final x position", f"{final_pos[0]:.4f}", 
                           f"{final_pos[0] - x_eq:.4f} from equilibrium")
                col2.metric("Final y position", f"{final_pos[1]:.4f}", 
                           f"{final_pos[1] - y_eq:.4f} from equilibrium")
                
                # Display migration time statistics
                st.write(f"Migration distance: {np.sqrt((final_pos[0]-initial_pos[0])**2 + (final_pos[1]-initial_pos[1])**2):.4f}")
                st.write(f"Final distance to equilibrium: {np.sqrt((final_pos[0]-x_eq)**2 + (final_pos[1]-y_eq)**2):.4f}")
                
                # Plot velocity magnitude vs time
                fig, ax = plt.subplots()
                ax.semilogy(trajectory['times'], trajectory['velocity_magnitudes'])
                ax.set_xlabel('Time')
                ax.set_ylabel('Velocity Magnitude (log scale)')
                ax.set_title('Velocity Magnitude During Migration')
                ax.grid(True)
                st.pyplot(fig)
            else:
                st.error("Error in trajectory calculation. Please try different parameters.")
        else:
            st.info("Migration simulation data not available. Run the calculation first.")
    
    # Display parameter sweep tab
    with tab6:
        st.subheader("Particle Radius Sweep Analysis")
        
        if 'sweep_results' in st.session_state:
            sweep_results = st.session_state.sweep_results
            
            # Display parameter sweep info
            st.write(f"Parameter swept: Particle Radius")
            st.write(f"Number of values tested: {len(sweep_results['param_values'])}")
            
            # Plot sweep results
            fig_sweep = viz.plot_parameter_sweep_results(sweep_results, ell, aspect_ratio)
            st.pyplot(fig_sweep)
            
            # Create a table of results
            sweep_data = []
            for i, value in enumerate(sweep_results['param_values']):
                if sweep_results['positions'][i] is not None:
                    pos = sweep_results['positions'][i]
                    sweep_data.append({
                        "Particle Radius": f"{value:.3f}",
                        "X Position": f"{pos[0]:.4f}",
                        "Y Position": f"{pos[1]:.4f}",
                        "Valid": "Yes"
                    })
                else:
                    sweep_data.append({
                        "Particle Radius": f"{value:.3f}",
                        "X Position": "N/A",
                        "Y Position": "N/A",
                        "Valid": "No"
                    })
            
            st.dataframe(sweep_data)
            
            # Add analysis text
            st.write("""
            **Analysis of Particle Size Effect:**
            
            Particle size affects both the magnitude and balance of forces in inertial focusing. Larger 
            particles experience stronger lift forces (scaling with a² or a³ depending on the regime) and 
            typically reach equilibrium positions that are closer to the channel center. Very small 
            particles may focus poorly or require longer channels for complete focusing due to weaker 
            inertial effects.
            """)
        else:
            st.info("Particle radius sweep data not available. Click 'Run Particle Radius Sweep' in the sidebar to perform a sweep analysis.")
else:
    # Instructions if no calculation has been run yet
    with tab1:
        st.info("Click 'Calculate Equilibrium Position' in the sidebar to run the simulation.")

# Footer with application information
st.markdown("---")
st.markdown("""
**About this simulator:**

This application implements a physics-based model for particle dynamics in microfluidic channels, 
focusing on inertial focusing phenomena. It uses the method of reflections and asymptotic analysis 
to calculate forces on particles and predict their equilibrium positions.

For more details about the underlying physics, expand the 'Learn About Inertial Focusing Physics' section at the top.
""")