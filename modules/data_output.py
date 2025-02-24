# modules/data_output.py

import os
import numpy as np

class DataOutput:
    def __init__(self, config):
        """
        Initialize the data output module using the configuration.

        Parameters:
            config (dict): Configuration dictionary containing simulation settings.
                           Expected to have an 'output' section with an 'output_dir' key.
        """
        self.config = config
        self.output_dir = config.get('simulation', {}).get('output', {}).get('output_dir', 'outputs')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def save_snapshot(self, snapshot):
        """
        Save a single simulation snapshot to an NPZ file.
        Each snapshot contains the step index, and the scalar potential (phi) and
        vector potential components (Ax, Ay).

        Parameters:
            snapshot (dict): A dictionary with keys 'step', 'phi', 'Ax', and 'Ay'.
        """
        step = snapshot.get('step', 0)
        filename = os.path.join(self.output_dir, f"snapshot_{step:05d}.npz")
        np.savez_compressed(filename, step=step, phi=snapshot['phi'], Ax=snapshot['Ax'], Ay=snapshot['Ay'])
        print(f"Saved snapshot at step {step} to {filename}")

    def save_all_snapshots(self, snapshots):
        """
        Save all simulation snapshots to the output directory.

        Parameters:
            snapshots (list): List of snapshot dictionaries.
        """
        for snapshot in snapshots:
            self.save_snapshot(snapshot)

    def save_checkpoint(self, state, checkpoint_name="checkpoint.npz"):
        """
        Save a checkpoint of the current simulation state.
        The checkpoint can include current field values and other simulation parameters
        that may be used to restart the simulation.

        Parameters:
            state (dict): Dictionary containing the simulation state variables.
            checkpoint_name (str): Filename for the checkpoint.
        """
        filename = os.path.join(self.output_dir, checkpoint_name)
        np.savez_compressed(filename, **state)
        print(f"Saved checkpoint to {filename}")
