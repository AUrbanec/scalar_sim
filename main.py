# main.py

import os
import yaml
from modules.grid import Grid
from modules.initialization import FieldInitialization
from modules.solver import Solver
from modules.visualization import Visualization
from modules.data_output import DataOutput

def load_config(config_file="config/config.yaml"):
    """
    Load the simulation configuration from a YAML file.

    Parameters:
        config_file (str): Path to the configuration YAML file.

    Returns:
        dict: The configuration dictionary.
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Load simulation configuration.
    config = load_config()

    # Create the simulation grid.
    grid = Grid(config)
    
    # Initialize scalar and vector potentials.
    initializer = FieldInitialization(config, grid)
    phi, A = initializer.initialize_fields()

    # Create the solver instance and run the simulation.
    solver = Solver(config, grid, phi, A)
    print("Starting simulation...")
    snapshots = solver.run()
    print(f"Simulation complete. Collected {len(snapshots)} snapshots.")

    # Save simulation snapshots and final state using the data output module.
    data_out = DataOutput(config)
    data_out.save_all_snapshots(snapshots)

    # Save a checkpoint of the final simulation state.
    state = {
        'phi': solver.phi_current,
        'Ax': solver.Ax_current,
        'Ay': solver.Ay_current,
        'step': solver.num_steps
    }
    data_out.save_checkpoint(state, checkpoint_name="final_checkpoint.npz")

    # Visualize snapshots and create an animation.
    viz = Visualization(config, grid)
    # Save individual snapshot images.
    for snapshot in snapshots:
        viz.save_snapshot(snapshot)
    
    # Create and save an animation of the simulation.
    viz.animate_snapshots(snapshots, save_filename="simulation_animation.mp4")
    print("Data output and visualization complete.")

if __name__ == "__main__":
    main()
