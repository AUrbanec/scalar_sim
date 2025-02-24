# modules/data_output.py

import os
import numpy as np

class DataOutput:
    def __init__(self, config):
        self.config = config
        self.output_dir = config.get('simulation', {}).get('output', {}).get('output_dir', 'outputs')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def save_snapshot(self, snapshot):
        step = snapshot.get('step', 0)
        filename = os.path.join(self.output_dir, f"snapshot_{step:05d}.npz")
        np.savez_compressed(filename, step=step, time=snapshot.get('time', 0.0), 
                            phi=snapshot['phi'], Ax=snapshot['Ax'], Ay=snapshot['Ay'],
                            energy=snapshot.get('energy', None))
        print(f"Saved snapshot at step {step} to {filename}")

    def save_all_snapshots(self, snapshots):
        for snapshot in snapshots:
            self.save_snapshot(snapshot)

    def save_checkpoint(self, state, checkpoint_name="checkpoint.npz"):
        filename = os.path.join(self.output_dir, checkpoint_name)
        np.savez_compressed(filename, **state)
        print(f"Saved checkpoint to {filename}")

    def save_energy_data(self, times, scalar_energy, vector_energy, filename="energy_data.npz"):
        filepath = os.path.join(self.output_dir, filename)
        np.savez_compressed(filepath, times=times, scalar_energy=scalar_energy, vector_energy=vector_energy)
        print(f"Saved energy data to {filepath}")
