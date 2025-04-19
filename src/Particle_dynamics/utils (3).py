import numpy as np
import streamlit as st
import time

def validate_parameters(particle_radius, ell, aspect_ratio):
    """
    Validate input parameters to ensure they are physically meaningful
    
    Parameters:
    - particle_radius: Radius of the particle
    - ell: Channel half-width
    - aspect_ratio: Ratio of channel height to width (h/w)
    
    Returns:
    - True if valid, False otherwise
    - Error message if invalid
    """
    # Check that particle radius is positive
    if particle_radius <= 0:
        return False, "Particle radius must be positive"
    
    # Check that channel width is positive
    if ell <= 0:
        return False, "Channel width must be positive"
    
    # Check that aspect ratio is positive
    if aspect_ratio <= 0:
        return False, "Aspect ratio must be positive"
    
    # Check that particle fits in the channel
    if particle_radius >= ell:
        return False, "Particle is too large for the channel width"
    
    # For non-square channels, check height constraint too
    if aspect_ratio >= 1.0:
        if particle_radius >= ell * aspect_ratio:
            return False, "Particle is too large for the channel height"
    else:
        if particle_radius >= ell / aspect_ratio:
            return False, "Particle is too large for the channel height"
    
    return True, ""

def create_progress_bar(total_steps, label="Calculating"):
    """
    Create a progress bar for long-running calculations
    
    Parameters:
    - total_steps: Total number of steps in calculation
    - label: Text label for the progress bar
    
    Returns:
    - progress_bar: Streamlit progress bar object
    - update_func: Function to update the progress
    """
    progress_text = st.empty()
    progress_bar = st.progress(0)
    progress_text.text(f"{label}... 0%")
    
    step = 0
    
    def update_progress():
        nonlocal step
        step += 1
        progress = min(step / total_steps, 1.0)
        progress_bar.progress(progress)
        progress_text.text(f"{label}... {int(progress * 100)}%")
    
    return progress_bar, progress_text, update_progress

def timer(func):
    """
    Decorator to time the execution of a function and display a spinner
    
    Parameters:
    - func: Function to time
    
    Returns:
    - Wrapped function that times execution
    """
    def wrapped(*args, **kwargs):
        spinner_text = f"Running {func.__name__}..."
        with st.spinner(spinner_text):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            # Display execution time if it took more than 0.5 seconds
            if end_time - start_time > 0.5:
                st.info(f"⏱️ {func.__name__} completed in {end_time - start_time:.2f} seconds")
                
        return result
    
    return wrapped
