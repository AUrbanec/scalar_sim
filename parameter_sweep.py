# parameter_sweep.py

import yaml
import copy
import numpy as np
import csv
import os
from modules.grid import Grid
from modules.initialization import FieldInitialization
from modules.solver import Solver
from analysis import compute_total_energies

def run_simulation_with_config(config):
    """
    Run a simulation with a specific configuration
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
        
    Returns
    -------
    tuple
        (snapshots, scalar_energy_max, vector_energy_max)
    """
    # Set up the simulation
    grid = Grid(config)
    initializer = FieldInitialization(config, grid)
    phi, A = initializer.initialize_fields()
    solver = Solver(config, grid, phi, A)
    
    # Run the simulation
    snapshots = solver.run()
    
    # Calculate energy metrics
    dimension = config.get('simulation', {}).get('dimension', 3)
    if dimension == 2:
        dx, dy = grid.get_spacing()
        times, scalar_energy, vector_energy = compute_total_energies(
            snapshots, solver.dt, solver.c, dx, dy
        )
    else:  # 3D case
        dx, dy, dz = grid.get_spacing()
        times, scalar_energy, vector_energy = compute_total_energies(
            snapshots, solver.dt, solver.c, dx, dy, dz
        )
    
    # Extract maximum values
    max_phi = max(np.max(abs(snap['phi'])) for snap in snapshots)
    scalar_energy_max = np.max(scalar_energy)
    vector_energy_max = np.max(vector_energy)
    
    return snapshots, max_phi, scalar_energy_max, vector_energy_max

def cavity_shape_parameter_sweep():
    """
    Run a parameter sweep for different 3D cavity shapes to determine optimal resonant structures
    """
    # Load base configuration
    with open("config/config.yaml", 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Ensure we're in 3D mode
    base_config['simulation']['dimension'] = 3
    
    # Define shapes and parameters to test
    cavity_shapes = [
        {"type": "box"},
        {"type": "sphere", "parameters": {"radius": 0.4}},
        {"type": "cylinder", "parameters": {"radius": 0.3, "height": 0.8, "axis": "z"}},
        {"type": "cylinder", "parameters": {"radius": 0.3, "height": 0.8, "axis": "x"}},
        {"type": "cylinder", "parameters": {"radius": 0.3, "height": 0.8, "axis": "y"}},
        {"type": "toroid", "parameters": {"major_radius": 0.3, "minor_radius": 0.1}}
    ]
    
    # Set output directory for this sweep
    sweep_output_dir = "outputs/cavity_sweep"
    os.makedirs(sweep_output_dir, exist_ok=True)
    
    # Results container
    results = []
    
    # Run simulations for each shape
    for i, cavity_config in enumerate(cavity_shapes):
        print(f"Running simulation {i+1}/{len(cavity_shapes)}: {cavity_config['type']} cavity")
        
        # Create a copy of the base config and update it
        config = copy.deepcopy(base_config)
        config['simulation']['cavity'] = cavity_config
        
        # Set unique output directory for this run
        run_output_dir = os.path.join(sweep_output_dir, f"{cavity_config['type']}_{i}")
        os.makedirs(run_output_dir, exist_ok=True)
        config['simulation']['output']['output_dir'] = run_output_dir
        
        # Run the simulation
        snapshots, max_phi, scalar_energy_max, vector_energy_max = run_simulation_with_config(config)
        
        # Calculate scalar/vector energy ratio (higher is better for scalar waves)
        if vector_energy_max > 0:
            scalar_vector_ratio = scalar_energy_max / vector_energy_max
        else:
            scalar_vector_ratio = float('inf')
        
        # Save results
        result = {
            'cavity_type': cavity_config['type'],
            'max_phi': max_phi,
            'scalar_energy_max': scalar_energy_max,
            'vector_energy_max': vector_energy_max,
            'scalar_vector_ratio': scalar_vector_ratio
        }
        
        # Add specific shape parameters to results
        if 'parameters' in cavity_config:
            for param, value in cavity_config['parameters'].items():
                result[f'param_{param}'] = value
                
        results.append(result)
        print(f"  Results: Max phi = {max_phi:.6f}, Scalar/Vector ratio = {scalar_vector_ratio:.6f}")
    
    # Write results to CSV
    output_file = "cavity_shape_results.csv"
    fieldnames = list(results[0].keys())
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"Parameter sweep complete. Results saved to {output_file}")

def impedance_parameter_sweep():
    """
    Run a parameter sweep for boundary impedance values
    """
    with open("config/config.yaml", 'r') as f:
        base_config = yaml.safe_load(f)
    
    impedance_values = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
    results = []
    
    for imp in impedance_values:
        print(f"Testing impedance value: {imp}")
        config = copy.deepcopy(base_config)
        config['simulation']['boundary_conditions']['impedance'] = imp
        snapshots, max_phi, scalar_energy_max, vector_energy_max = run_simulation_with_config(config)
        
        # Calculate scalar/vector energy ratio
        if vector_energy_max > 0:
            scalar_vector_ratio = scalar_energy_max / vector_energy_max
        else:
            scalar_vector_ratio = float('inf')
            
        results.append({
            'impedance': imp, 
            'max_phi': max_phi,
            'scalar_energy_max': scalar_energy_max,
            'vector_energy_max': vector_energy_max,
            'scalar_vector_ratio': scalar_vector_ratio
        })
        print(f"  Results: Max phi = {max_phi:.6f}, Scalar/Vector ratio = {scalar_vector_ratio:.6f}")
    
    with open("impedance_sweep_results.csv", 'w', newline='') as csvfile:
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print("Parameter sweep complete. Results saved to impedance_sweep_results.csv")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run parameter sweep for scalar wave simulation')
    parser.add_argument('--type', choices=['impedance', 'cavity'], default='cavity',
                        help='Type of parameter sweep to run')
    args = parser.parse_args()
    
    if args.type == 'impedance':
        impedance_parameter_sweep()
    else:  # cavity
        cavity_shape_parameter_sweep()
