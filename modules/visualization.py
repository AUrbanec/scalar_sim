# modules/visualization.py

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import colors
try:
    from mpl_toolkits.mplot3d import Axes3D
    from skimage import measure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    VOLUME_RENDER_AVAILABLE = True
except ImportError:
    VOLUME_RENDER_AVAILABLE = False

class Visualization:
    def __init__(self, config, grid):
        """
        Initialize visualization tools for the scalar wave simulation
        
        Parameters
        ----------
        config : dict
            Configuration dictionary containing simulation parameters
        grid : Grid
            Grid object with mesh information
        """
        self.config = config
        self.grid = grid
        self.dimension = config.get('simulation', {}).get('dimension', 3)
        self.output_dir = config.get('simulation', {}).get('output', {}).get('output_dir', 'outputs')
        
        # Visualization parameters
        viz_config = config.get('simulation', {}).get('visualization', {})
        self.slice_axis = viz_config.get('slice_axis', 'z')
        self.slice_position = viz_config.get('slice_position', 0.5)
        self.volume_rendering = viz_config.get('volume_rendering', True) and VOLUME_RENDER_AVAILABLE
        self.isosurface_levels = viz_config.get('isosurface_levels', [0.005, 0.01, 0.02])
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def plot_snapshot_2d(self, snapshot):
        """
        Plot a 2D snapshot of the simulation
        
        Parameters
        ----------
        snapshot : dict
            Snapshot dictionary containing field data
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure object containing the plot
        """
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
        
    def plot_snapshot_3d(self, snapshot):
        """
        Plot a 3D snapshot of the simulation using slices
        
        Parameters
        ----------
        snapshot : dict
            Snapshot dictionary containing field data
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure object containing the plot
        """
        step = snapshot['step']
        phi = snapshot['phi']
        Ax = snapshot['Ax']
        Ay = snapshot['Ay']
        Az = snapshot['Az']
        
        # Get 2D slice coordinates
        (X_slice, Y_slice), slice_idx = self.grid.get_slice(
            axis=self.slice_axis,
            position=self.slice_position
        )
        
        # Extract slices of the 3D fields
        if self.slice_axis == 'x':
            phi_slice = phi[slice_idx, :, :]
            Ay_slice = Ay[slice_idx, :, :]
            Az_slice = Az[slice_idx, :, :]
            A_slice_x = Ay_slice
            A_slice_y = Az_slice
            extent = [0, self.grid.y_length, 0, self.grid.z_length]
            xlabel, ylabel = "y", "z"
        elif self.slice_axis == 'y':
            phi_slice = phi[:, slice_idx, :]
            Ax_slice = Ax[:, slice_idx, :]
            Az_slice = Az[:, slice_idx, :]
            A_slice_x = Ax_slice
            A_slice_y = Az_slice
            extent = [0, self.grid.x_length, 0, self.grid.z_length]
            xlabel, ylabel = "x", "z"
        else:  # slice_axis == 'z'
            phi_slice = phi[:, :, slice_idx]
            Ax_slice = Ax[:, :, slice_idx]
            Ay_slice = Ay[:, :, slice_idx]
            A_slice_x = Ax_slice
            A_slice_y = Ay_slice
            extent = [0, self.grid.x_length, 0, self.grid.y_length]
            xlabel, ylabel = "x", "y"
        
        # Create the figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot scalar potential slice
        im0 = axes[0].imshow(phi_slice.T, extent=extent, origin='lower', cmap='viridis')
        axes[0].set_title(f"Scalar Potential ϕ ({self.slice_axis}={self.slice_position:.2f}, Step {step})")
        axes[0].set_xlabel(xlabel)
        axes[0].set_ylabel(ylabel)
        fig.colorbar(im0, ax=axes[0])
        
        # Plot vector potential slice
        skip = (slice(None, None, max(1, int(phi_slice.shape[0] / 20))),
                slice(None, None, max(1, int(phi_slice.shape[1] / 20))))
        q = axes[1].quiver(X_slice[skip], Y_slice[skip], 
                           A_slice_x.T[skip], A_slice_y.T[skip])
        axes[1].set_title(f"Vector Potential A ({self.slice_axis}={self.slice_position:.2f}, Step {step})")
        axes[1].set_xlabel(xlabel)
        axes[1].set_ylabel(ylabel)
        axes[1].set_aspect('equal')
        
        # Plot energy slice if available
        energy = snapshot.get('energy', None)
        if energy is not None:
            if self.slice_axis == 'x':
                energy_slice = energy[slice_idx, :, :]
            elif self.slice_axis == 'y':
                energy_slice = energy[:, slice_idx, :]
            else:  # slice_axis == 'z'
                energy_slice = energy[:, :, slice_idx]
                
            im2 = axes[2].imshow(energy_slice.T, extent=extent,
                                 origin='lower', cmap='inferno')
            axes[2].set_title(f"Energy Density ({self.slice_axis}={self.slice_position:.2f}, Step {step})")
            axes[2].set_xlabel(xlabel)
            axes[2].set_ylabel(ylabel)
            fig.colorbar(im2, ax=axes[2])
        else:
            axes[2].axis('off')
            axes[2].set_title("No Energy Data")
        
        plt.tight_layout()
        return fig
        
    def plot_volume_snapshot(self, snapshot):
        """
        Create a 3D volume visualization
        
        Parameters
        ----------
        snapshot : dict
            Snapshot dictionary containing field data
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure object containing the 3D volume plot
        """
        if not VOLUME_RENDER_AVAILABLE:
            print("Warning: Volume rendering libraries not available. Install scikit-image for 3D visualization.")
            return None
            
        X, Y, Z = self.grid.get_meshgrid()
        phi = snapshot['phi']
        step = snapshot['step']
        
        # Create figure with 3D axis
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Determine threshold values
        phi_max = np.max(phi)
        levels = [level * phi_max for level in self.isosurface_levels]
        
        # Plot isosurfaces
        colors_list = ['blue', 'green', 'red']
        for i, level in enumerate(levels):
            if i < len(colors_list):
                color = colors_list[i]
            else:
                color = 'gray'
                
            try:
                verts, faces, _, _ = measure.marching_cubes(phi, level)
                
                # Scale vertices to physical coordinates
                verts[:, 0] = verts[:, 0] * self.grid.dx
                verts[:, 1] = verts[:, 1] * self.grid.dy
                verts[:, 2] = verts[:, 2] * self.grid.dz
                
                # Create polygon collection
                mesh = Poly3DCollection(verts[faces], alpha=0.3, color=color)
                ax.add_collection3d(mesh)
            except:
                print(f"Warning: Could not create isosurface for level {level}")
        
        # Set plot limits and labels
        ax.set_xlim(0, self.grid.x_length)
        ax.set_ylim(0, self.grid.y_length)
        ax.set_zlim(0, self.grid.z_length)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Scalar Potential ϕ Isosurfaces (Step {step})')
        
        plt.tight_layout()
        return fig
        
    def plot_snapshot(self, snapshot):
        """
        Plot a snapshot of the simulation (2D or 3D)
        
        Parameters
        ----------
        snapshot : dict
            Snapshot dictionary containing field data
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure object containing the plot
        """
        if self.dimension == 2:
            return self.plot_snapshot_2d(snapshot)
        else:
            # Always plot the slice representation
            slice_fig = self.plot_snapshot_3d(snapshot)
            
            # Optionally plot volume rendering if enabled
            if self.volume_rendering:
                vol_fig = self.plot_volume_snapshot(snapshot)
                
                # Save volume figure if created successfully
                if vol_fig is not None:
                    step = snapshot['step']
                    vol_filename = os.path.join(self.output_dir, f"volume_snapshot_{step:05d}.png")
                    vol_fig.savefig(vol_filename)
                    plt.close(vol_fig)
                    
            return slice_fig

    def save_snapshot(self, snapshot):
        """
        Save a snapshot visualization to a file
        
        Parameters
        ----------
        snapshot : dict
            Snapshot dictionary containing field data
        """
        fig = self.plot_snapshot(snapshot)
        step = snapshot['step']
        filename = os.path.join(self.output_dir, f"snapshot_step_{step:05d}.png")
        fig.savefig(filename)
        plt.close(fig)

    def animate_snapshots_2d(self, snapshots, save_filename="simulation_animation.mp4"):
        """
        Create an animation of 2D snapshots
        
        Parameters
        ----------
        snapshots : list
            List of snapshot dictionaries
        save_filename : str
            Filename for the animation
        """
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
        
    def animate_snapshots_3d(self, snapshots, save_filename="simulation_animation.mp4"):
        """
        Create an animation of 3D snapshots
        
        Parameters
        ----------
        snapshots : list
            List of snapshot dictionaries
        save_filename : str
            Filename for the animation
        """
        # Get 2D slice coordinates
        (X_slice, Y_slice), slice_idx = self.grid.get_slice(
            axis=self.slice_axis,
            position=self.slice_position
        )
        
        # Set up the figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Extract first snapshot data
        if self.slice_axis == 'x':
            phi_slice = snapshots[0]['phi'][slice_idx, :, :]
            Ay_slice = snapshots[0]['Ay'][slice_idx, :, :]
            Az_slice = snapshots[0]['Az'][slice_idx, :, :]
            A_slice_x = Ay_slice
            A_slice_y = Az_slice
            extent = [0, self.grid.y_length, 0, self.grid.z_length]
            xlabel, ylabel = "y", "z"
        elif self.slice_axis == 'y':
            phi_slice = snapshots[0]['phi'][:, slice_idx, :]
            Ax_slice = snapshots[0]['Ax'][:, slice_idx, :]
            Az_slice = snapshots[0]['Az'][:, slice_idx, :]
            A_slice_x = Ax_slice
            A_slice_y = Az_slice
            extent = [0, self.grid.x_length, 0, self.grid.z_length]
            xlabel, ylabel = "x", "z"
        else:  # slice_axis == 'z'
            phi_slice = snapshots[0]['phi'][:, :, slice_idx]
            Ax_slice = snapshots[0]['Ax'][:, :, slice_idx]
            Ay_slice = snapshots[0]['Ay'][:, :, slice_idx]
            A_slice_x = Ax_slice
            A_slice_y = Ay_slice
            extent = [0, self.grid.x_length, 0, self.grid.y_length]
            xlabel, ylabel = "x", "y"
        
        # Initialize plots
        im0 = axes[0].imshow(phi_slice.T, extent=extent,
                              origin='lower', cmap='viridis')
        axes[0].set_title(f"Scalar Potential ϕ ({self.slice_axis}={self.slice_position:.2f})")
        axes[0].set_xlabel(xlabel)
        axes[0].set_ylabel(ylabel)
        fig.colorbar(im0, ax=axes[0])
        
        skip = (slice(None, None, max(1, int(phi_slice.shape[0] / 20))),
                slice(None, None, max(1, int(phi_slice.shape[1] / 20))))
        q = axes[1].quiver(X_slice[skip], Y_slice[skip], 
                           A_slice_x.T[skip], A_slice_y.T[skip])
        axes[1].set_title(f"Vector Potential A ({self.slice_axis}={self.slice_position:.2f})")
        axes[1].set_xlabel(xlabel)
        axes[1].set_ylabel(ylabel)
        axes[1].set_aspect('equal')
        
        def update(frame):
            snapshot = snapshots[frame]
            
            # Extract slice data
            if self.slice_axis == 'x':
                phi_slice = snapshot['phi'][slice_idx, :, :]
                Ay_slice = snapshot['Ay'][slice_idx, :, :]
                Az_slice = snapshot['Az'][slice_idx, :, :]
                A_slice_x = Ay_slice
                A_slice_y = Az_slice
            elif self.slice_axis == 'y':
                phi_slice = snapshot['phi'][:, slice_idx, :]
                Ax_slice = snapshot['Ax'][:, slice_idx, :]
                Az_slice = snapshot['Az'][:, slice_idx, :]
                A_slice_x = Ax_slice
                A_slice_y = Az_slice
            else:  # slice_axis == 'z'
                phi_slice = snapshot['phi'][:, :, slice_idx]
                Ax_slice = snapshot['Ax'][:, :, slice_idx]
                Ay_slice = snapshot['Ay'][:, :, slice_idx]
                A_slice_x = Ax_slice
                A_slice_y = Ay_slice
            
            # Update plots
            im0.set_data(phi_slice.T)
            axes[0].set_title(f"Scalar Potential ϕ ({self.slice_axis}={self.slice_position:.2f}, Step {snapshot['step']})")
            q.set_UVC(A_slice_x.T[skip], A_slice_y.T[skip])
            axes[1].set_title(f"Vector Potential A ({self.slice_axis}={self.slice_position:.2f}, Step {snapshot['step']})")
            return im0, q

        anim = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=200, blit=False)
        anim.save(os.path.join(self.output_dir, save_filename), writer='ffmpeg')
        plt.close(fig)

    def animate_snapshots(self, snapshots, save_filename="simulation_animation.mp4"):
        """
        Create an animation of snapshots (2D or 3D)
        
        Parameters
        ----------
        snapshots : list
            List of snapshot dictionaries
        save_filename : str
            Filename for the animation
        """
        if self.dimension == 2:
            self.animate_snapshots_2d(snapshots, save_filename)
        else:
            self.animate_snapshots_3d(snapshots, save_filename)

    def plot_energy_time_series(self, times, scalar_energy, vector_energy):
        """
        Plot energy time series data
        
        Parameters
        ----------
        times : ndarray
            Array of time values
        scalar_energy : ndarray
            Array of scalar energy values
        vector_energy : ndarray
            Array of vector energy values
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure object containing the energy plot
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(times, scalar_energy, label='Scalar Energy')
        ax.plot(times, vector_energy, label='Vector Energy')
        ax.set_xlabel("Time")
        ax.set_ylabel("Energy")
        ax.set_title("Energy Time Series")
        ax.legend()
        return fig
