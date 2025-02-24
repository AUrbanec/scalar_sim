# modules/data_output.py

import os
import numpy as np

class DataOutput:
    def __init__(self, config):
        """
        Initialize data output functionality for the simulation
        
        Parameters
        ----------
        config : dict
            Configuration dictionary containing simulation parameters
        """
        self.config = config
        self.dimension = config.get('simulation', {}).get('dimension', 3)
        self.output_dir = config.get('simulation', {}).get('output', {}).get('output_dir', 'outputs')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def save_snapshot(self, snapshot):
        """
        Save a simulation snapshot to a compressed NPZ file
        
        Parameters
        ----------
        snapshot : dict
            Snapshot dictionary containing field data
        """
        step = snapshot.get('step', 0)
        filename = os.path.join(self.output_dir, f"snapshot_{step:05d}.npz")
        
        if self.dimension == 2:
            np.savez_compressed(filename, 
                                step=step, 
                                time=snapshot.get('time', 0.0), 
                                phi=snapshot['phi'], 
                                Ax=snapshot['Ax'], 
                                Ay=snapshot['Ay'],
                                energy=snapshot.get('energy', None))
        else:  # 3D case
            np.savez_compressed(filename, 
                                step=step, 
                                time=snapshot.get('time', 0.0), 
                                phi=snapshot['phi'], 
                                Ax=snapshot['Ax'], 
                                Ay=snapshot['Ay'],
                                Az=snapshot['Az'],
                                energy=snapshot.get('energy', None))
        
        print(f"Saved snapshot at step {step} to {filename}")

    def save_all_snapshots(self, snapshots):
        """
        Save all snapshots to files
        
        Parameters
        ----------
        snapshots : list
            List of snapshot dictionaries
        """
        for snapshot in snapshots:
            self.save_snapshot(snapshot)

    def save_checkpoint(self, state, checkpoint_name="checkpoint.npz"):
        """
        Save a simulation checkpoint
        
        Parameters
        ----------
        state : dict
            Dictionary containing simulation state
        checkpoint_name : str
            Name of the checkpoint file
        """
        filename = os.path.join(self.output_dir, checkpoint_name)
        np.savez_compressed(filename, **state)
        print(f"Saved checkpoint to {filename}")

    def save_energy_data(self, times, scalar_energy, vector_energy, filename="energy_data.npz"):
        """
        Save energy time series data
        
        Parameters
        ----------
        times : ndarray
            Array of time values
        scalar_energy : ndarray
            Array of scalar energy values
        vector_energy : ndarray
            Array of vector energy values
        filename : str
            Name of the output file
        """
        filepath = os.path.join(self.output_dir, filename)
        np.savez_compressed(filepath, times=times, scalar_energy=scalar_energy, vector_energy=vector_energy)
        print(f"Saved energy data to {filepath}")
        
    def load_snapshot(self, filename):
        """
        Load a snapshot from a file
        
        Parameters
        ----------
        filename : str
            Path to the snapshot file
            
        Returns
        -------
        dict
            Snapshot dictionary containing field data
        """
        data = np.load(filename)
        snapshot = {}
        for key in data.files:
            snapshot[key] = data[key]
        return snapshot
    
    def load_energy_data(self, filename="energy_data.npz"):
        """
        Load energy time series data
        
        Parameters
        ----------
        filename : str
            Path to the energy data file
            
        Returns
        -------
        tuple
            (times, scalar_energy, vector_energy)
        """
        filepath = os.path.join(self.output_dir, filename)
        data = np.load(filepath)
        return data['times'], data['scalar_energy'], data['vector_energy']
