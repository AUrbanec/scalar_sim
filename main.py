# main.py

import os
import yaml
from modules.grid import Grid
from modules.initialization import FieldInitialization
from modules.solver import Solver
from modules.visualization import Visualization
from modules.data_output import DataOutput
from analysis import compute_total_energies

def load_config(config_file="config/config.yaml"):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    config = load_config()
    grid = Grid(config)
    initializer = FieldInitialization(config, grid)
    phi, A = initializer.initialize_fields()
    solver = Solver(config, grid, phi, A)
    
    print("Starting simulation...")
    snapshots = solver.run()
    print(f"Simulation complete. Collected {len(snapshots)} snapshots.")

    dx, dy = grid.get_spacing()
    times, scalar_energy, vector_energy = compute_total_energies(snapshots, solver.dt, solver.c, dx, dy)

    data_out = DataOutput(config)
    data_out.save_all_snapshots(snapshots)
    state = {
        'phi': solver.phi_current,
        'Ax': solver.Ax_current,
        'Ay': solver.Ay_current,
        'step': solver.num_steps
    }
    data_out.save_checkpoint(state, checkpoint_name="final_checkpoint.npz")
    data_out.save_energy_data(times, scalar_energy, vector_energy)

    viz = Visualization(config, grid)
    for snapshot in snapshots:
        viz.save_snapshot(snapshot)
    viz.animate_snapshots(snapshots, save_filename="simulation_animation.mp4")
    
    fig_energy = viz.plot_energy_time_series(times, scalar_energy, vector_energy)
    energy_fig_path = os.path.join(config['simulation']['output']['output_dir'], "energy_time_series.png")
    fig_energy.savefig(energy_fig_path)
    print("Data output and visualization complete.")

if __name__ == "__main__":
    main()
