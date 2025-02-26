import os
import yaml
import json
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from main import load_config, main as run_simulation
import threading
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['CONFIG_FILE'] = 'config/config.yaml'

# Ensure required directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Global variables to track simulation status
simulation_running = False
simulation_progress = 0
simulation_message = ""

@app.route('/')
def index():
    """Render the main dashboard page"""
    config = load_config(app.config['CONFIG_FILE'])
    return render_template('index.html', config=config)

@app.route('/config', methods=['GET', 'POST'])
def config_page():
    """Handle configuration page"""
    if request.method == 'POST':
        # Handle form submission to update config
        updated_config = process_config_form(request.form)
        
        # Save the updated config
        with open(app.config['CONFIG_FILE'], 'w') as f:
            yaml.dump(updated_config, f)
            
        return redirect(url_for('index'))
    
    # Load and display current config
    config = load_config(app.config['CONFIG_FILE'])
    return render_template('config.html', config=config)

def process_config_form(form_data):
    """Process form data and update configuration"""
    # Load existing config
    config = load_config(app.config['CONFIG_FILE'])
    
    # Update grid parameters
    config['simulation']['grid']['x_points'] = int(form_data.get('x_points', 50))
    config['simulation']['grid']['y_points'] = int(form_data.get('y_points', 50))
    config['simulation']['dimension'] = int(form_data.get('dimension', 3))
    
    if config['simulation']['dimension'] == 3:
        config['simulation']['grid']['z_points'] = int(form_data.get('z_points', 50))
        config['simulation']['grid']['z_length'] = float(form_data.get('z_length', 1.0))
    
    config['simulation']['grid']['x_length'] = float(form_data.get('x_length', 1.0))
    config['simulation']['grid']['y_length'] = float(form_data.get('y_length', 1.0))
    
    # Update time parameters
    config['simulation']['time']['dt'] = float(form_data.get('dt', 0.001))
    config['simulation']['time']['total_time'] = float(form_data.get('total_time', 1.0))
    
    # Update material parameters
    config['simulation']['material']['epsilon'] = float(form_data.get('epsilon', 1.0))
    config['simulation']['material']['mu'] = float(form_data.get('mu', 1.0))
    
    # Update boundary conditions
    config['simulation']['boundary_conditions']['type'] = form_data.get('bc_type', 'dirichlet')
    config['simulation']['boundary_conditions']['impedance'] = float(form_data.get('impedance', 0.1))
    
    # Update source parameters
    config['simulation']['source']['enabled'] = 'source_enabled' in form_data
    config['simulation']['source']['amplitude'] = float(form_data.get('source_amplitude', 1.0))
    config['simulation']['source']['frequency'] = float(form_data.get('source_frequency', 50.0))
    config['simulation']['source']['modulation'] = form_data.get('source_modulation', 'amplitude')
    
    # Parse source center as a list
    center_str = form_data.get('source_center', '0.5, 0.5, 0.5')
    try:
        center = [float(x.strip()) for x in center_str.split(',')]
        config['simulation']['source']['center'] = center
    except ValueError:
        # Keep default if parsing fails
        pass
    
    config['simulation']['source']['sigma'] = float(form_data.get('source_sigma', 0.05))
    
    # Update nonlinearity
    config['simulation']['nonlinearity']['enabled'] = 'nonlinearity_enabled' in form_data
    config['simulation']['nonlinearity']['strength'] = float(form_data.get('nonlinearity_strength', 0.01))
    
    # Update visualization
    config['simulation']['visualization']['slice_axis'] = form_data.get('slice_axis', 'z')
    config['simulation']['visualization']['slice_position'] = float(form_data.get('slice_position', 0.5))
    config['simulation']['visualization']['volume_rendering'] = 'volume_rendering' in form_data
    
    # Update output
    config['simulation']['output']['snapshot_interval'] = int(form_data.get('snapshot_interval', 50))
    config['simulation']['output']['output_dir'] = form_data.get('output_dir', 'outputs')
    
    return config

@app.route('/run-simulation', methods=['POST'])
def run_simulation_route():
    """Start simulation in a background thread"""
    global simulation_running, simulation_progress, simulation_message
    
    if simulation_running:
        return jsonify({'status': 'error', 'message': 'Simulation already running'})
    
    # Set initial status
    simulation_running = True
    simulation_progress = 0
    simulation_message = "Initializing simulation..."
    
    # Start simulation in a background thread
    threading.Thread(target=simulation_thread).start()
    
    return jsonify({'status': 'started'})

def simulation_thread():
    """Background thread to run the simulation"""
    global simulation_running, simulation_progress, simulation_message
    
    try:
        # Update progress messages
        simulation_message = "Setting up simulation parameters..."
        simulation_progress = 5
        time.sleep(0.5)  # Small delay to allow progress updates
        
        simulation_message = "Initializing grid and fields..."
        simulation_progress = 10
        time.sleep(0.5)
        
        # Run the simulation
        simulation_message = "Running simulation..."
        run_simulation()
        
        # Simulation completed successfully
        simulation_message = "Simulation completed successfully!"
        simulation_progress = 100
        
    except Exception as e:
        # Handle any errors
        simulation_message = f"Error: {str(e)}"
        simulation_progress = 0
    
    finally:
        # Always mark the simulation as finished
        simulation_running = False

@app.route('/simulation-status')
def simulation_status():
    """Return the current simulation status as JSON"""
    global simulation_running, simulation_progress, simulation_message
    
    return jsonify({
        'running': simulation_running,
        'progress': simulation_progress,
        'message': simulation_message
    })

@app.route('/results')
def results():
    """Display simulation results"""
    # Get a list of output files (animations and energy plot only)
    output_dir = 'outputs'
    animation_file = 'simulation_animation.mp4' if os.path.exists(os.path.join(output_dir, 'simulation_animation.mp4')) else None
    volume_file = 'volume_animation.mp4' if os.path.exists(os.path.join(output_dir, 'volume_animation.mp4')) else None
    energy_file = 'energy_time_series.png' if os.path.exists(os.path.join(output_dir, 'energy_time_series.png')) else None
    
    return render_template(
        'results.html', 
        animation=animation_file,
        volume_animation=volume_file,
        energy_file=energy_file
    )

@app.route('/output/<path:filename>')
def serve_output(filename):
    """Serve files from the output directory"""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/documentation')
def documentation():
    """Display documentation page"""
    return render_template('documentation.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)