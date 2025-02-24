# modules/visualization.py

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

class Visualization:
    def __init__(self, config, grid):
        self.config = config
        self.grid = grid
        self.output_dir = config.get('simulation', {}).get('output', {}).get('output_dir', 'outputs')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def plot_snapshot(self, snapshot):
        X, Y = self.grid.get_meshgrid()
        phi = snapshot['phi']
        Ax = snapshot['Ax']
        Ay = snapshot['Ay']
        step = snapshot['step']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        im0 = axes[0].imshow(phi.T, extent=[0, self.grid.x_length, 0, self.grid.y_length],
                               origin='lower', cmap='viridis')
        axes[0].set_title(f"Scalar Potential ϕ (Step {step})")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        fig.colorbar(im0, ax=axes[0])
        
        skip = (slice(None, None, max(1, int(self.grid.x_points / 20))),
                slice(None, None, max(1, int(self.grid.y_points / 20))))
        q = axes[1].quiver(X[skip], Y[skip], Ax.T[skip], Ay.T[skip])
        axes[1].set_title(f"Vector Potential A (Step {step})")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].set_aspect('equal')
        
        energy = snapshot.get('energy', None)
        if energy is not None:
            im2 = axes[2].imshow(energy.T, extent=[0, self.grid.x_length, 0, self.grid.y_length],
                                  origin='lower', cmap='inferno')
            axes[2].set_title(f"Energy Density (Step {step})")
            axes[2].set_xlabel("x")
            axes[2].set_ylabel("y")
            fig.colorbar(im2, ax=axes[2])
        else:
            axes[2].axis('off')
            axes[2].set_title("No Energy Data")
        
        plt.tight_layout()
        return fig

    def save_snapshot(self, snapshot):
        fig = self.plot_snapshot(snapshot)
        step = snapshot['step']
        filename = os.path.join(self.output_dir, f"snapshot_step_{step:05d}.png")
        fig.savefig(filename)
        plt.close(fig)

    def animate_snapshots(self, snapshots, save_filename="simulation_animation.mp4"):
        X, Y = self.grid.get_meshgrid()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        im0 = axes[0].imshow(snapshots[0]['phi'].T,
                               extent=[0, self.grid.x_length, 0, self.grid.y_length],
                               origin='lower', cmap='viridis')
        axes[0].set_title("Scalar Potential ϕ")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        fig.colorbar(im0, ax=axes[0])
        
        skip = (slice(None, None, max(1, int(self.grid.x_points / 20))),
                slice(None, None, max(1, int(self.grid.y_points / 20))))
        q = axes[1].quiver(X[skip], Y[skip],
                           snapshots[0]['Ax'].T[skip],
                           snapshots[0]['Ay'].T[skip])
        axes[1].set_title("Vector Potential A")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].set_aspect('equal')
        
        def update(frame):
            snapshot = snapshots[frame]
            im0.set_data(snapshot['phi'].T)
            axes[0].set_title(f"Scalar Potential ϕ (Step {snapshot['step']})")
            q.set_UVC(snapshot['Ax'].T[skip], snapshot['Ay'].T[skip])
            axes[1].set_title(f"Vector Potential A (Step {snapshot['step']})")
            return im0, q

        anim = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=200, blit=False)
        anim.save(os.path.join(self.output_dir, save_filename), writer='ffmpeg')
        plt.close(fig)

    def plot_energy_time_series(self, times, scalar_energy, vector_energy):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(times, scalar_energy, label='Scalar Energy')
        ax.plot(times, vector_energy, label='Vector Energy')
        ax.set_xlabel("Time")
        ax.set_ylabel("Energy")
        ax.set_title("Energy Time Series")
        ax.legend()
        return fig
