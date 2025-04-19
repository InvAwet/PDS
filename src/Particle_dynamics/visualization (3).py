"""
Visualization Module for Particle Equilibrium Solver

This module provides visualization functions for displaying simulation results,
including equilibrium positions, flow fields, and particle migration.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import matplotlib.colors as colors
from matplotlib.ticker import FormatStrFormatter
import io
import base64

def plot_equilibrium_position(x_eq, y_eq, particle_radius, ell, aspect_ratio, force_field=None):
    """
    Plot the equilibrium position of a particle in a channel
    
    Parameters:
    - x_eq, y_eq: Equilibrium coordinates
    - particle_radius: Radius of the particle
    - ell: Channel half-width
    - aspect_ratio: Ratio of channel height to width (h/w)
    - force_field: Optional force field data for visualization
    
    Returns:
    - Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10 * aspect_ratio))
    
    # Calculate channel dimensions
    if aspect_ratio >= 1.0:
        width = 2 * ell
        height = 2 * ell * aspect_ratio
    else:
        height = 2 * ell
        width = 2 * ell / aspect_ratio
    
    # Draw channel
    rect = Rectangle((-width/2, -height/2), width, height, 
                     facecolor='none', edgecolor='black', linestyle='-')
    ax.add_patch(rect)
    
    # Draw centerlines
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # Plot force field if provided
    if force_field and len(force_field['x']) > 0:
        # Calculate force magnitudes for scaling arrow size
        magnitudes = np.array(force_field['magnitude'])
        max_mag = np.max(magnitudes)
        
        # Normalize arrow sizes
        scale_factor = 0.2 * min(width, height) / max_mag
        
        # Plot force arrows
        ax.quiver(force_field['x'], force_field['y'], 
                 np.array(force_field['fx']) * scale_factor, 
                 np.array(force_field['fy']) * scale_factor,
                 magnitudes, alpha=0.6, cmap='viridis',
                 scale=1.0, scale_units='xy')
        
        # Optional: Add colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis', 
                                  norm=plt.Normalize(vmin=0, vmax=max_mag))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Force Magnitude')
    
    # Draw particle at equilibrium position
    circle = Circle((x_eq, y_eq), particle_radius, 
                   facecolor='red', edgecolor='black', alpha=0.7)
    ax.add_patch(circle)
    
    # Add marker for equilibrium position
    ax.plot(x_eq, y_eq, 'k+', markersize=10, markeredgewidth=2)
    
    # Set axis limits with some margin
    margin = 0.1 * max(width, height)
    ax.set_xlim(-width/2 - margin, width/2 + margin)
    ax.set_ylim(-height/2 - margin, height/2 + margin)
    
    # Set labels and title
    ax.set_xlabel('x position')
    ax.set_ylabel('y position')
    ax.set_title(f'Equilibrium Position at ({x_eq:.4f}, {y_eq:.4f})')
    ax.set_aspect('equal')
    
    return fig

def plot_flow_field(flow_field, particle_position, particle_radius, ell, aspect_ratio):
    """
    Plot the flow field around a particle
    
    Parameters:
    - flow_field: Dictionary with flow field data
    - particle_position: [x, y] coordinates of particle center
    - particle_radius: Radius of the particle
    - ell: Channel half-width
    - aspect_ratio: Ratio of channel height to width (h/w)
    
    Returns:
    - Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract data from flow field dictionary
    X, Y = flow_field['X'], flow_field['Y']
    U, V = flow_field['U'], flow_field['V']
    
    # Calculate speed for color mapping
    speed = np.sqrt(U**2 + V**2)
    
    # Create streamplot of the flow field
    strm = ax.streamplot(X, Y, U, V, color=speed, cmap='viridis',
                        linewidth=1, density=1.5, arrowsize=1.5)
    
    # Add colorbar
    cbar = fig.colorbar(strm.lines, ax=ax)
    cbar.set_label('Flow Speed')
    
    # Draw particle
    x_eq, y_eq = particle_position
    circle = Circle((x_eq, y_eq), particle_radius, 
                   facecolor='red', edgecolor='black', alpha=0.7)
    ax.add_patch(circle)
    
    # Calculate channel dimensions
    if aspect_ratio >= 1.0:
        width = 2 * ell
        height = 2 * ell * aspect_ratio
    else:
        height = 2 * ell
        width = 2 * ell / aspect_ratio
    
    # Draw channel outline
    rect = Rectangle((-width/2, -height/2), width, height, 
                     facecolor='none', edgecolor='black', linestyle='-')
    ax.add_patch(rect)
    
    # Set labels and title
    ax.set_xlabel('x position')
    ax.set_ylabel('y position')
    ax.set_title('Flow Field Around Particle')
    ax.set_aspect('equal')
    
    return fig

def plot_force_contours(x_points, y_points, force_mag, aspect_ratio, ell, 
                        particle_radius, equilibrium_position=None):
    """
    Plot contours of force magnitude to visualize equilibrium regions
    
    Parameters:
    - x_points, y_points: Arrays of coordinates where forces were evaluated
    - force_mag: Force magnitude at each point
    - aspect_ratio: Ratio of channel height to width (h/w)
    - ell: Channel half-width
    - particle_radius: Radius of the particle
    - equilibrium_position: Optional [x_eq, y_eq] coordinates
    
    Returns:
    - Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 10 * aspect_ratio))
    
    # Create grid for contour plot
    X, Y = np.meshgrid(x_points, y_points)
    Z = np.reshape(force_mag, X.shape)
    
    # Calculate channel dimensions
    if aspect_ratio >= 1.0:
        width = 2 * ell
        height = 2 * ell * aspect_ratio
    else:
        height = 2 * ell
        width = 2 * ell / aspect_ratio
    
    # Create contour plot with log scale for better visualization
    Z_log = np.log10(Z + 1e-10)  # Add small value to avoid log(0)
    
    # Plot contour
    contour = ax.contourf(X, Y, Z_log, levels=20, cmap='viridis')
    
    # Add colorbar
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('Log10(Force Magnitude)')
    
    # Draw channel boundary
    rect = Rectangle((-width/2, -height/2), width, height, 
                     facecolor='none', edgecolor='black', linestyle='-')
    ax.add_patch(rect)
    
    # Mark equilibrium position if provided
    if equilibrium_position is not None:
        x_eq, y_eq = equilibrium_position
        circle = Circle((x_eq, y_eq), particle_radius, 
                       facecolor='red', edgecolor='black', alpha=0.5)
        ax.add_patch(circle)
        ax.plot(x_eq, y_eq, 'k+', markersize=10, markeredgewidth=2)
    
    # Set axis limits
    ax.set_xlim(-width/2, width/2)
    ax.set_ylim(-height/2, height/2)
    
    # Set labels and title
    ax.set_xlabel('x position')
    ax.set_ylabel('y position')
    ax.set_title('Force Magnitude Contour Map')
    ax.set_aspect('equal')
    
    return fig

def plot_background_flow(ell, aspect_ratio, resolution=50, Um=1.0):
    """
    Plot the background flow profile in the channel
    
    Parameters:
    - ell: Channel half-width
    - aspect_ratio: Ratio of channel height to width (h/w)
    - resolution: Resolution of the calculated grid
    - Um: Maximum flow velocity
    
    Returns:
    - Matplotlib figure
    """
    # Import here to avoid circular imports
    import particle_physics as pp
    
    # Calculate channel dimensions
    if aspect_ratio >= 1.0:
        width = 2 * ell
        height = 2 * ell * aspect_ratio
    else:
        height = 2 * ell
        width = 2 * ell / aspect_ratio
    
    # Create grid for flow calculation
    x = np.linspace(-width/2, width/2, resolution)
    y = np.linspace(-height/2, height/2, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Calculate background flow at each grid point
    U = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            U[i, j] = pp.background_flow(X[i, j], Y[i, j], ell, Um, aspect_ratio)
    
    # Create figure for 2D colormap of velocity
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Colormap plot
    cm = ax1.pcolormesh(X, Y, U, cmap='viridis', shading='auto')
    ax1.set_title('Background Flow Velocity')
    ax1.set_xlabel('x position')
    ax1.set_ylabel('y position')
    ax1.set_aspect('equal')
    cbar1 = fig.colorbar(cm, ax=ax1)
    cbar1.set_label('Velocity')
    
    # Draw channel boundary
    rect1 = Rectangle((-width/2, -height/2), width, height, 
                     facecolor='none', edgecolor='black', linestyle='-')
    ax1.add_patch(rect1)
    
    # Add velocity profile along y-axis (x=0)
    center_x_idx = resolution // 2
    ax2.plot(U[:, center_x_idx], y, 'b-', linewidth=2)
    ax2.set_title('Velocity Profile at Channel Center (x=0)')
    ax2.set_xlabel('Velocity')
    ax2.set_ylabel('y position')
    ax2.grid(True)
    
    # Add velocity profile along x-axis (y=0)
    center_y_idx = resolution // 2
    ax2.plot(x, U[center_y_idx, :], 'r--', linewidth=2, 
            label='x-axis profile (y=0)')
    ax2.legend()
    
    plt.tight_layout()
    return fig

def plot_parameter_sweep_results(sweep_results, ell, aspect_ratio):
    """
    Plot equilibrium positions as a function of the swept parameter
    
    Parameters:
    - sweep_results: Dictionary with parameter sweep results
    - ell: Channel half-width
    - aspect_ratio: Default aspect ratio
    
    Returns:
    - Matplotlib figure
    """
    param_name = sweep_results['param_name']
    param_values = sweep_results['param_values']
    positions = sweep_results['positions']
    
    # Extract valid results (removing None values)
    valid_indices = [i for i, pos in enumerate(positions) if pos is not None]
    valid_values = [param_values[i] for i in valid_indices]
    valid_positions = [positions[i] for i in valid_indices]
    
    # Extract x and y coordinates
    x_positions = [pos[0] for pos in valid_positions]
    y_positions = [pos[1] for pos in valid_positions]
    
    # Normalize positions by channel dimensions
    if aspect_ratio >= 1.0:
        x_normalized = [x / ell for x in x_positions]
        y_normalized = [y / (ell * aspect_ratio) for y in y_positions]
    else:
        x_normalized = [x / (ell / aspect_ratio) for x in x_positions]
        y_normalized = [y / ell for y in y_positions]
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot x position vs parameter
    ax1.plot(valid_values, x_positions, 'o-', color='blue')
    ax1.set_xlabel(param_name)
    ax1.set_ylabel('x position')
    ax1.set_title(f'x position vs {param_name}')
    ax1.grid(True)
    
    # Plot y position vs parameter
    ax2.plot(valid_values, y_positions, 'o-', color='red')
    ax2.set_xlabel(param_name)
    ax2.set_ylabel('y position')
    ax2.set_title(f'y position vs {param_name}')
    ax2.grid(True)
    
    # Plot normalized positions in channel
    for i, value in enumerate(valid_values):
        ax3.plot(x_normalized[i], y_normalized[i], 'o', 
                markersize=8, alpha=0.7,
                label=f'{param_name}={value:.3f}')
    
    # Add channel boundary
    rect = Rectangle((-1, -1), 2, 2, facecolor='none', 
                    edgecolor='black', linestyle='-')
    ax3.add_patch(rect)
    
    # Add crosshairs at origin
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    ax3.set_xlim(-1.1, 1.1)
    ax3.set_ylim(-1.1, 1.1)
    ax3.set_xlabel('Normalized x position')
    ax3.set_ylabel('Normalized y position')
    ax3.set_title('Equilibrium Positions in Channel')
    ax3.set_aspect('equal')
    ax3.legend(loc='upper right', fontsize='small')
    
    plt.tight_layout()
    return fig

def plot_migration_results(simulator, trajectory):
    """
    Plot complete results of a particle migration simulation
    
    Parameters:
    - simulator: ParticleMigrationSimulator instance
    - trajectory: Dictionary with trajectory data
    
    Returns:
    - Matplotlib figure
    """
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Define grid for plots: 2 rows, 3 columns
    gs = fig.add_gridspec(2, 3)
    
    # Plot 1: Trajectory in channel
    ax1 = fig.add_subplot(gs[0, :2])  # Top left, spans 2 columns
    
    # Draw channel
    rect = Rectangle(
        (-simulator.width/2, -simulator.height/2), 
        simulator.width, simulator.height,
        facecolor='none', edgecolor='black', linestyle='-'
    )
    ax1.add_patch(rect)
    
    # Draw centerlines
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # Plot trajectory
    positions = trajectory['positions']
    ax1.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.7)
    
    # Mark initial and final positions
    ax1.plot(positions[0, 0], positions[0, 1], 'go', markersize=8, label='Initial')
    ax1.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=8, label='Final')
    
    # Plot equilibrium positions
    if simulator.equilibrium_positions:
        eq_x = [pos[0] for pos in simulator.equilibrium_positions]
        eq_y = [pos[1] for pos in simulator.equilibrium_positions]
        ax1.scatter(eq_x, eq_y, s=100, c='red', marker='*', 
                   edgecolor='white', zorder=3, label='Equilibrium')
    
    # Draw particle at final position
    circle = Circle(
        (positions[-1, 0], positions[-1, 1]), 
        simulator.particle_radius, 
        fc='blue', ec='black', alpha=0.3
    )
    ax1.add_patch(circle)
    
    # Set labels and title
    ax1.set_xlabel('x position')
    ax1.set_ylabel('y position')
    ax1.set_title('Particle Trajectory')
    ax1.legend()
    ax1.set_aspect('equal')
    
    # Set axis limits with margin
    margin = 0.1 * max(simulator.width, simulator.height)
    ax1.set_xlim(-simulator.width/2 - margin, simulator.width/2 + margin)
    ax1.set_ylim(-simulator.height/2 - margin, simulator.height/2 + margin)
    
    # Plot 2: Position vs. time
    ax2 = fig.add_subplot(gs[0, 2])  # Top right
    times = trajectory['times']
    ax2.plot(times, positions[:, 0], 'b-', label='x position')
    ax2.plot(times, positions[:, 1], 'r-', label='y position')
    
    # Mark convergence time if available
    if trajectory['converged']:
        converge_time = trajectory['convergence_time']
        ax2.axvline(x=converge_time, color='gray', linestyle='--')
        ax2.text(converge_time, ax2.get_ylim()[1]*0.9, 
                f'Converged: {converge_time:.2f}', 
                rotation=90, verticalalignment='top')
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Position')
    ax2.set_title('Position vs Time')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Velocity magnitude vs. time
    ax3 = fig.add_subplot(gs[1, 0])  # Bottom left
    velocity_mags = trajectory['velocity_magnitudes']
    ax3.semilogy(times, velocity_mags, 'g-')
    
    # Mark convergence threshold
    if 'convergence_threshold' in trajectory:
        threshold = trajectory['convergence_threshold']
    else:
        threshold = 1e-6  # Default value
    ax3.axhline(y=threshold, color='r', linestyle='--')
    ax3.text(times[-1]*0.5, threshold*1.5, f'Threshold: {threshold:.2e}')
    
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Velocity Magnitude (log scale)')
    ax3.set_title('Velocity Magnitude vs Time')
    ax3.grid(True)
    
    # Plot 4: Distance to closest equilibrium vs. time
    ax4 = fig.add_subplot(gs[1, 1])  # Bottom middle
    
    # Calculate distance to closest equilibrium at each time step
    if simulator.equilibrium_positions:
        distances = []
        for pos in positions:
            min_dist = float('inf')
            for eq_pos in simulator.equilibrium_positions:
                dist = np.sqrt((pos[0] - eq_pos[0])**2 + (pos[1] - eq_pos[1])**2)
                min_dist = min(min_dist, dist)
            distances.append(min_dist)
        
        ax4.plot(times, distances, 'm-')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Distance to Equilibrium')
        ax4.set_title('Distance to Closest Equilibrium')
        ax4.grid(True)
    else:
        ax4.text(0.5, 0.5, 'No equilibrium positions defined', 
                ha='center', va='center', transform=ax4.transAxes)
    
    # Plot 5: Velocity vectors
    ax5 = fig.add_subplot(gs[1, 2])  # Bottom right
    
    # Create a grid and calculate velocity at each point
    x_points = np.linspace(-simulator.width/2, simulator.width/2, 10)
    y_points = np.linspace(-simulator.height/2, simulator.height/2, 10)
    X, Y = np.meshgrid(x_points, y_points)
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            pos = [X[i, j], Y[i, j]]
            
            # Skip points outside channel or inside walls
            if (abs(pos[0]) + simulator.particle_radius >= simulator.width/2 or 
                abs(pos[1]) + simulator.particle_radius >= simulator.height/2):
                U[i, j] = 0
                V[i, j] = 0
                continue
                
            vel = simulator.calculate_velocity(pos)
            U[i, j] = vel[0]
            V[i, j] = vel[1]
    
    # Calculate magnitudes for coloring
    speed = np.sqrt(U**2 + V**2)
    
    # Draw velocity field
    ax5.quiver(X, Y, U, V, speed, cmap='viridis', pivot='mid')
    
    # Draw channel
    rect2 = Rectangle(
        (-simulator.width/2, -simulator.height/2), 
        simulator.width, simulator.height,
        facecolor='none', edgecolor='black', linestyle='-'
    )
    ax5.add_patch(rect2)
    
    # Add particle trajectory
    ax5.plot(positions[:, 0], positions[:, 1], 'k-', alpha=0.5)
    
    ax5.set_xlabel('x position')
    ax5.set_ylabel('y position')
    ax5.set_title('Velocity Field')
    ax5.set_aspect('equal')
    
    plt.tight_layout()
    return fig
