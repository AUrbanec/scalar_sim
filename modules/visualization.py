# modules/visualization.py

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

class Visualization:
    def __init__(self, config, grid):
        """
        Visualization module for plotting simulation fields.
        
        Parameters:
            config (dict): Simulation configuration dictionary.
            grid (Grid): An instance of the Grid class containing meshgrid and spacing information.
        """
        self.config = config
        self.grid = grid
        
        # Determine output directory for figures.
        self.output_dir = config.get('simulation', {}).get('output', {}).get('output_dir', 'outputs')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def plot_snapshot(self, snapshot):
        """
        Plot the scalar potential and vector potential for a given snapshot.
        
        Parameters:
            snapshot (dict): A dictionary containing 'step', 'phi', 'Ax', and 'Ay' fields.
        
        Returns:
            matplotlib.figure.Figure: The created figure object.
        """
        X, Y = self.grid.get_meshgrid()
        phi = snapshot['phi']
        Ax = snapshot['Ax']
        Ay = snapshot['Ay']
        step = snapshot['step']
        
        # Create a figure with two subplots: one for phi and one for the vector field A.
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot scalar potential (ϕ) using imshow.
        im0 = axes[0].imshow(phi.T, extent=[0, self.grid.x_length, 0, self.grid.y_length],
                               origin='lower', cmap='viridis')
        axes[0].set_title(f"Scalar Potential ϕ (Step {step})")
        axes[0].set_xlabel("x (m)")
        axes[0].set_ylabel("y (m)")
        fig.colorbar(im0, ax=axes[0])
        
        # Plot vector potential using a quiver plot.
        # Downsample grid for clarity if needed.
        skip = (slice(None, None, max(1, int(self.grid.x_points / 20))),
                slice(None, None, max(1, int(self.grid.y_points / 20))))
        axes[1].quiver(X[skip], Y[skip], Ax.T[skip], Ay.T[skip])
        axes[1].set_title(f"Vector Potential A (Step {step})")
        axes[1].set_xlabel("x (m)")
        axes[1].set_ylabel("y (m)")
        axes[1].set_aspect('equal')
        
        plt.tight_layout()
        return fig

    def save_snapshot(self, snapshot):
        """
        Save the visualization for a given snapshot to a PNG file.
        
        Parameters:
            snapshot (dict): Snapshot dictionary containing 'step', 'phi', 'Ax', and 'Ay'.
        """
        fig = self.plot_snapshot(snapshot)
        step = snapshot['step']
        filename = os.path.join(self.output_dir, f"snapshot_step_{step:05d}.png")
        fig.savefig(filename)
        plt.close(fig)

    def animate_snapshots(self, snapshots, save_filename="simulation_animation.mp4"):
        X, Y = self.grid.get_meshgrid()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Initialize the scalar potential image.
        im0 = axes[0].imshow(snapshots[0]['phi'].T,
                            extent=[0, self.grid.x_length, 0, self.grid.y_length],
                            origin='lower', cmap='viridis')
        axes[0].set_title("Scalar Potential ϕ")
        axes[0].set_xlabel("x (m)")
        axes[0].set_ylabel("y (m)")
        fig.colorbar(im0, ax=axes[0])
        
        # Set up the quiver plot for the vector potential.
        skip = (slice(None, None, max(1, int(self.grid.x_points / 20))),
                slice(None, None, max(1, int(self.grid.y_points / 20))))
        q = axes[1].quiver(X[skip], Y[skip],
                        snapshots[0]['Ax'].T[skip],
                        snapshots[0]['Ay'].T[skip])
        axes[1].set_title("Vector Potential A")
        axes[1].set_xlabel("x (m)")
        axes[1].set_ylabel("y (m)")
        axes[1].set_aspect('equal')
        
        def update(frame):
            snapshot = snapshots[frame]
            im0.set_data(snapshot['phi'].T)
            axes[0].set_title(f"Scalar Potential ϕ (Step {snapshot['step']})")
            # Update the quiver data without reassigning collections.
            q.set_UVC(snapshot['Ax'].T[skip], snapshot['Ay'].T[skip])
            axes[1].set_title(f"Vector Potential A (Step {snapshot['step']})")
            return im0, q

        anim = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=200, blit=False)
        anim.save(os.path.join(self.output_dir, save_filename), writer='ffmpeg')
        plt.close(fig)
