"""
Particle Migration Simulation Module

This module simulates the time-dependent migration of particles from initial positions
to their equilibrium positions within microfluidic channels under inertial focusing conditions.
It implements the full trajectory computation based on forces calculated from the 
underlying physics model.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import time
from scipy.integrate import solve_ivp

# Import local modules
import particle_physics as pp

class ParticleMigrationSimulator:
    """
    Simulator for particle migration under inertial forces in microfluidic channels.
    
    This class simulates the time-based migration of particles to their equilibrium positions,
    accounting for inertial lift forces, wall effects, and flow profile.
    """
    
    def __init__(self, alpha, Re, ell, rho=1.0, Um=1.0, mu=1.0, aspect_ratio=1.0):
        """
        Initialize the particle migration simulator.
        
        Parameters:
        - alpha: Particle size ratio (a/ell, where a is particle radius)
        - Re: Reynolds number
        - ell: Channel half-width
        - rho: Fluid density
        - Um: Maximum flow velocity
        - mu: Fluid viscosity
        - aspect_ratio: Ratio of channel height to width (h/w)
        """
        self.alpha = alpha
        self.Re = Re
        self.ell = ell
        self.rho = rho
        self.Um = Um
        self.mu = mu
        self.aspect_ratio = aspect_ratio
        
        # Calculate particle radius and channel dimensions
        self.particle_radius = alpha * ell
        if aspect_ratio >= 1.0:
            self.height = 2 * ell * aspect_ratio
            self.width = 2 * ell
        else:
            self.height = 2 * ell
            self.width = 2 * ell / aspect_ratio
            
        # Get stable equilibrium positions for reference
        self.equilibrium_positions = pp.find_equilibrium_positions(
            alpha, Re, ell, rho, Um, mu, aspect_ratio
        )
        
        # Physics constants
        # Stokes drag coefficient: 6πμa for a sphere
        self.drag_coefficient = 6 * np.pi * mu * self.particle_radius
        
        # Time scale factors
        self.characteristic_velocity = Um
        self.characteristic_length = ell
        self.characteristic_time = self.characteristic_length / self.characteristic_velocity
        
        # Mobility factor (inverse of drag coefficient, normalized)
        # Relates force to velocity: v = M * F
        self.mobility = 1.0 / self.drag_coefficient
        
        # Results storage
        self.trajectories = []
        self.migration_times = []
        
    def generate_random_initial_positions(self, num_particles, safe_margin=None):
        """
        Generate random initial positions for particles within the channel.
        
        Parameters:
        - num_particles: Number of particles to place
        - safe_margin: Margin from walls (if None, uses particle radius)
        
        Returns:
        - List of [x, y] positions
        """
        if safe_margin is None:
            safe_margin = 1.2 * self.particle_radius
            
        # Available space for particle centers
        x_min = -self.width/2 + safe_margin
        x_max = self.width/2 - safe_margin
        y_min = -self.height/2 + safe_margin
        y_max = self.height/2 - safe_margin
        
        # Generate random positions
        positions = []
        for _ in range(num_particles):
            # Use uniform distribution for unbiased sampling
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            positions.append([x, y])
            
        return positions
    
    def calculate_velocity(self, position):
        """
        Calculate particle velocity at a given position based on forces.
        
        Parameters:
        - position: [x, y] coordinates
        
        Returns:
        - [vx, vy]: Velocity vector
        """
        # Calculate net force at position
        fx, fy = pp.lift_force(
            position[0], position[1], 
            self.alpha, self.Re, self.ell,
            self.rho, self.Um, self.mu, 
            self.aspect_ratio
        )
        
        # Convert force to velocity using mobility (F = m·a, v = F/drag)
        vx = fx * self.mobility
        vy = fy * self.mobility
        
        return np.array([vx, vy])
    
    def particle_ode(self, t, y):
        """
        ODE function for particle motion.
        
        Parameters:
        - t: Time (not used explicitly, but required for solve_ivp)
        - y: State vector [x, y]
        
        Returns:
        - [dx/dt, dy/dt]: Derivatives for ODE solver
        """
        position = y  # y is [x, y]
        # Skip calculation if position is outside valid range
        x, y = position
        if (abs(x) + self.particle_radius >= self.width/2 or 
            abs(y) + self.particle_radius >= self.height/2):
            return np.zeros(2)  # No motion if at or beyond walls
            
        return self.calculate_velocity(position)
    
    def simulate_migration(self, initial_position, t_max=50.0, dt=0.1, 
                          convergence_threshold=1e-6, progress_callback=None):
        """
        Simulate migration of a single particle from initial position.
        
        Parameters:
        - initial_position: [x, y] starting coordinates
        - t_max: Maximum simulation time
        - dt: Time step for output
        - convergence_threshold: Threshold velocity magnitude to consider converged
        - progress_callback: Optional callback function for progress updates
        
        Returns:
        - Dictionary with trajectory information
        """
        # Time points for solution
        t_span = (0, t_max)
        t_eval = np.arange(0, t_max, dt)
        
        # Solve the ODE
        try:
            # Use adaptive step size integration for stability and accuracy
            solution = solve_ivp(
                self.particle_ode, 
                t_span, 
                initial_position,
                t_eval=t_eval,
                method='RK45',  # 4th order Runge-Kutta with adaptive step size
                rtol=1e-4,      # Relative tolerance
                atol=1e-6       # Absolute tolerance
            )
            
            # Extract results
            times = solution.t
            positions = solution.y.T  # Transpose to get [time, position] format
            
            # Calculate velocities at each step
            velocities = np.zeros_like(positions)
            for i, pos in enumerate(positions):
                velocities[i] = self.calculate_velocity(pos)
                
            # Calculate velocity magnitudes
            velocity_magnitudes = np.sqrt(velocities[:, 0]**2 + velocities[:, 1]**2)
            
            # Determine convergence time
            # Consider converged if velocity drops below threshold
            converged_indices = np.where(velocity_magnitudes < convergence_threshold)[0]
            if len(converged_indices) > 0:
                convergence_time = times[converged_indices[0]]
                converged = True
            else:
                convergence_time = t_max
                converged = False
                
            # Find closest equilibrium position
            final_position = positions[-1]
            closest_eq = None
            min_dist = float('inf')
            
            for eq_pos in self.equilibrium_positions:
                dist = np.sqrt((final_position[0] - eq_pos[0])**2 + 
                              (final_position[1] - eq_pos[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_eq = eq_pos
            
            # Store results
            result = {
                'times': times,
                'positions': positions,
                'velocities': velocities,
                'velocity_magnitudes': velocity_magnitudes,
                'initial_position': initial_position,
                'final_position': final_position,
                'converged': converged,
                'convergence_time': convergence_time,
                'closest_equilibrium': closest_eq,
                'distance_to_equilibrium': min_dist if closest_eq is not None else None
            }
            
            return result
            
        except Exception as e:
            print(f"Error in simulation: {e}")
            return None
    
    def simulate_multiple_particles(self, initial_positions, t_max=50.0, dt=0.1,
                                  convergence_threshold=1e-6, progress_callback=None):
        """
        Simulate migration of multiple particles.
        
        Parameters:
        - initial_positions: List of [x, y] starting coordinates
        - Other parameters same as simulate_migration
        
        Returns:
        - List of trajectory dictionaries
        """
        self.trajectories = []
        self.migration_times = []
        
        # Total particles for progress tracking
        total_particles = len(initial_positions)
        
        # Simulate each particle
        for i, init_pos in enumerate(initial_positions):
            # Update progress if callback provided
            if progress_callback:
                progress = i / total_particles
                progress_callback(progress)
                
            # Run simulation for this particle
            trajectory = self.simulate_migration(
                init_pos, t_max, dt, convergence_threshold
            )
            
            if trajectory:
                self.trajectories.append(trajectory)
                self.migration_times.append(trajectory['convergence_time'])
        
        # Final progress update
        if progress_callback:
            progress_callback(1.0)
            
        return self.trajectories
    
    def visualize_migration(self, trajectory, show_equilibrium=True, animation_speed=20):
        """
        Create an animation of particle migration.
        
        Parameters:
        - trajectory: Trajectory dictionary from simulate_migration
        - show_equilibrium: Whether to show equilibrium positions
        - animation_speed: Number of steps to skip per frame (higher = faster)
        
        Returns:
        - Animation object
        """
        positions = trajectory['positions']
        times = trajectory['times']
        
        # Setup figure and axes
        fig, ax = plt.subplots(figsize=(10, 10 * self.height/self.width))
        
        # Draw channel boundary
        rect = plt.Rectangle(
            (-self.width/2, -self.height/2), 
            self.width, self.height,
            facecolor='none', edgecolor='black', linestyle='-'
        )
        ax.add_patch(rect)
        
        # Add centerlines
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        # Plot trajectory path
        ax.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.3)
        
        # Plot initial position
        ax.plot(positions[0, 0], positions[0, 1], 'go', markersize=8, label='Initial position')
        
        # Plot equilibrium positions
        if show_equilibrium and self.equilibrium_positions:
            eq_x = [pos[0] for pos in self.equilibrium_positions]
            eq_y = [pos[1] for pos in self.equilibrium_positions]
            ax.scatter(eq_x, eq_y, s=100, c='red', marker='*', 
                      edgecolor='white', zorder=3, label='Equilibrium positions')
        
        # Create particle object (circle) for animation
        particle = Circle(
            (positions[0, 0], positions[0, 1]), 
            self.particle_radius, 
            fc='blue', ec='black', alpha=0.7
        )
        ax.add_patch(particle)
        
        # Add time display text
        time_text = ax.text(
            0.02, 0.95, '', transform=ax.transAxes,
            fontsize=12, verticalalignment='top'
        )
        
        # Set axis limits with margin
        margin = 0.1 * max(self.width, self.height)
        ax.set_xlim(-self.width/2 - margin, self.width/2 + margin)
        ax.set_ylim(-self.height/2 - margin, self.height/2 + margin)
        
        # Set labels and title
        ax.set_xlabel('x position (dimensionless)')
        ax.set_ylabel('y position (dimensionless)')
        ax.set_title(f'Particle Migration (a/ell={self.alpha:.2f}, Re={self.Re:.1f})')
        ax.set_aspect('equal')
        ax.legend(loc='upper right')
        
        # Animation function
        def update(frame):
            # Update particle position
            idx = min(frame * animation_speed, len(positions) - 1)
            particle.center = (positions[idx, 0], positions[idx, 1])
            
            # Update time text
            time_text.set_text(f'Time: {times[idx]:.2f}')
            
            return particle, time_text
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, update, frames=len(positions) // animation_speed + 1,
            interval=50, blit=True
        )
        
        return fig, anim
    
    def plot_trajectory(self, trajectory, ax=None, color='blue', show_equilibrium=True):
        """
        Plot a single particle trajectory.
        
        Parameters:
        - trajectory: Trajectory dictionary from simulate_migration
        - ax: Matplotlib axis to plot on (creates new if None)
        - color: Color for the trajectory
        - show_equilibrium: Whether to show equilibrium positions
        
        Returns:
        - Matplotlib axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8 * self.height/self.width))
        
        positions = trajectory['positions']
        initial_pos = trajectory['initial_position']
        final_pos = trajectory['final_position']
        
        # Draw channel boundary
        rect = plt.Rectangle(
            (-self.width/2, -self.height/2), 
            self.width, self.height,
            facecolor='none', edgecolor='black', linestyle='-'
        )
        ax.add_patch(rect)
        
        # Add centerlines
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        # Plot trajectory
        ax.plot(positions[:, 0], positions[:, 1], '-', color=color, alpha=0.7)
        ax.plot(initial_pos[0], initial_pos[1], 'o', color='green', markersize=8)
        ax.plot(final_pos[0], final_pos[1], 'o', color='red', markersize=8)
        
        # Plot particle at final position
        particle = Circle(
            (final_pos[0], final_pos[1]), 
            self.particle_radius, 
            fc=color, ec='black', alpha=0.3
        )
        ax.add_patch(particle)
        
        # Plot equilibrium positions
        if show_equilibrium and self.equilibrium_positions:
            eq_x = [pos[0] for pos in self.equilibrium_positions]
            eq_y = [pos[1] for pos in self.equilibrium_positions]
            ax.scatter(eq_x, eq_y, s=100, c='red', marker='*', 
                     edgecolor='white', zorder=3)
        
        # Set labels and limits
        ax.set_xlabel('x position (dimensionless)')
        ax.set_ylabel('y position (dimensionless)')
        
        # Set axis limits with margin
        margin = 0.1 * max(self.width, self.height)
        ax.set_xlim(-self.width/2 - margin, self.width/2 + margin)
        ax.set_ylim(-self.height/2 - margin, self.height/2 + margin)
        ax.set_aspect('equal')
        
        return ax
    
    def plot_multiple_trajectories(self, trajectories=None, colors=None):
        """
        Plot multiple particle trajectories on a single plot.
        
        Parameters:
        - trajectories: List of trajectory dictionaries (uses self.trajectories if None)
        - colors: List of colors for each trajectory (generates if None)
        
        Returns:
        - Matplotlib figure and axis
        """
        if trajectories is None:
            trajectories = self.trajectories
            
        if not trajectories:
            raise ValueError("No trajectories to plot")
            
        if colors is None:
            # Generate colors using a colormap
            import matplotlib.cm as cm
            cmap = cm.get_cmap('tab10')
            colors = [cmap(i % 10) for i in range(len(trajectories))]
        
        fig, ax = plt.subplots(figsize=(10, 10 * self.height/self.width))
        
        # Plot each trajectory
        for i, traj in enumerate(trajectories):
            self.plot_trajectory(traj, ax=ax, color=colors[i], 
                              show_equilibrium=(i==0))  # Only show equilibrium once
        
        # Add legend for initial and final positions
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                  markersize=10, label='Initial positions'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                  markersize=10, label='Final positions'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='red', 
                  markersize=15, label='Equilibrium positions')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.set_title(f'Particle Migration Trajectories (a/ell={self.alpha:.2f}, Re={self.Re:.1f})')
        
        return fig, ax
    
    def analyze_convergence_times(self):
        """
        Analyze the convergence times of multiple particle trajectories.
        
        Returns:
        - Dictionary with convergence statistics
        """
        if not self.trajectories:
            raise ValueError("No trajectories to analyze")
            
        # Extract convergence times and distances
        times = []
        distances = []
        positions = []
        
        for traj in self.trajectories:
            if traj['converged']:
                times.append(traj['convergence_time'])
                if traj['distance_to_equilibrium'] is not None:
                    distances.append(traj['distance_to_equilibrium'])
                positions.append(traj['final_position'])
        
        # Calculate statistics
        stats = {
            'count': len(times),
            'mean_time': np.mean(times) if times else None,
            'std_time': np.std(times) if times else None,
            'min_time': np.min(times) if times else None,
            'max_time': np.max(times) if times else None,
            'mean_distance': np.mean(distances) if distances else None,
            'std_distance': np.std(distances) if distances else None,
            'positions': positions
        }
        
        return stats
