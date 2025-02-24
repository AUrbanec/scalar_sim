# parameter_sweep.py

import yaml
import copy
import numpy as np
import csv
from modules.grid import Grid
from modules.initialization import FieldInitialization
from modules.solver import Solver

def run_simulation_with_config(config):
    from modules.geometry import Geometry  # local import
    grid = Grid(config)
    initializer = FieldInitialization(config, grid)
    phi, A = initializer.initialize_fields()
    solver = Solver(config, grid, phi, A)
    snapshots = solver.run()
    return snapshots

def parameter_sweep():
    with open("config/config.yaml", 'r') as f:
        base_config = yaml.safe_load(f)
    
    impedance_values = [0.05, 0.1, 0.2]
    results = []
    
    for imp in impedance_values:
        config = copy.deepcopy(base_config)
        config['simulation']['boundary_conditions']['impedance'] = imp
        snapshots = run_simulation_with_config(config)
        max_phi = max(np.max(abs(snap['phi'])) for snap in snapshots)
        results.append({'impedance': imp, 'max_phi': max_phi})
        print(f"Impedance: {imp}, Max phi: {max_phi}")
    
    with open("parameter_sweep_results.csv", 'w', newline='') as csvfile:
        fieldnames = ['impedance', 'max_phi']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print("Parameter sweep complete. Results saved to parameter_sweep_results.csv")

if __name__ == "__main__":
    parameter_sweep()
