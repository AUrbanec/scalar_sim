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
        phi_max = np.max(np.abs(phi))
        if phi_max == 0:
            print("Warning: Maximum phi value is zero, no isosurfaces to display")
            return None
            
        levels = [level * phi_max for level in self.isosurface_levels]
        
        # Plot isosurfaces
        colors_list = ['blue', 'green', 'red']
        isosurfaces_plotted = 0
        
        for i, level in enumerate(levels):
            if i < len(colors_list):
                color = colors_list[i]
            else:
                color = 'gray'
                
            try:
                # Using scikit-image's marching_cubes algorithm
                verts, faces, _, _ = measure.marching_cubes(phi, level)
                
                # Scale vertices to physical coordinates
                verts[:, 0] = verts[:, 0] * self.grid.dx
                verts[:, 1] = verts[:, 1] * self.grid.dy
                verts[:, 2] = verts[:, 2] * self.grid.dz
                
                # Create polygon collection
                mesh = Poly3DCollection(verts[faces], alpha=0.3, color=color)
                ax.add_collection3d(mesh)
                isosurfaces_plotted += 1
            except Exception as e:
                print(f"Warning: Could not create isosurface for level {level}. Error: {str(e)}")
        
        # If we couldn't plot any isosurfaces, try with lower levels
        if isosurfaces_plotted == 0:
            lower_levels = [0.001, 0.002, 0.003]
            for i, level in enumerate(lower_levels):
                try:
                    level_value = level * phi_max
                    verts, faces, _, _ = measure.marching_cubes(phi, level_value)
                    
                    # Scale vertices to physical coordinates
                    verts[:, 0] = verts[:, 0] * self.grid.dx
                    verts[:, 1] = verts[:, 1] * self.grid.dy
                    verts[:, 2] = verts[:, 2] * self.grid.dz
                    
                    # Create polygon collection
                    mesh = Poly3DCollection(verts[faces], alpha=0.3, color='purple')
                    ax.add_collection3d(mesh)
                    isosurfaces_plotted += 1
                    print(f"Successfully plotted lower level isosurface at {level} * max_phi")
                    break  # Stop after successfully plotting one isosurface
                except Exception as e:
                    continue
        
        # If still no isosurfaces, return None
        if isosurfaces_plotted == 0:
            print("Could not create any isosurfaces. Check field values and try different isosurface levels.")
            return None
        
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

    def animate_volume_snapshots(self, snapshots, save_filename="volume_animation.mp4"):
        """
        Create an animation of 3D volume snapshots
        
        Parameters
        ----------
        snapshots : list
            List of snapshot dictionaries
        save_filename : str
            Filename for the animation
        """
        if not VOLUME_RENDER_AVAILABLE:
            print("Warning: Volume rendering libraries not available. Install scikit-image for 3D visualization.")
            return
            
        # Determine threshold values based on maximum phi across all snapshots
        all_phi_max = max([np.max(np.abs(snapshot['phi'])) for snapshot in snapshots])
        if all_phi_max == 0:
            print("Warning: Maximum phi value is zero across all snapshots, no isosurfaces to display")
            return
        
        # Use a consistent level for all frames
        level = 0.01 * all_phi_max  # Use a single level for smoother animation
        
        # Create figure
        fig = plt.figure(figsize=(10, 8))
        
        # Try to create the first frame to test the level
        test_phi = snapshots[0]['phi']
        try:
            verts, faces, _, _ = measure.marching_cubes(test_phi, level)
        except Exception as e:
            # If the level is too high, try a lower level
            print(f"Initial level setting failed. Trying a lower level. Error: {str(e)}")
            level = 0.005 * all_phi_max
            try:
                verts, faces, _, _ = measure.marching_cubes(test_phi, level)
            except Exception as e:
                # If still failing, try with absolute values
                print(f"Second level attempt failed. Trying with absolute values. Error: {str(e)}")
                level = 0.001 * all_phi_max
                try:
                    verts, faces, _, _ = measure.marching_cubes(np.abs(test_phi), level)
                    # If this works, use absolute values for all frames
                    use_abs = True
                except Exception as e:
                    print(f"All attempts to create isosurfaces failed. Error: {str(e)}")
                    return
        else:
            use_abs = False
        
        print(f"Creating 3D volume animation with isosurface level {level:.6f} (use_abs={use_abs})")
        
        # Function to create a frame
        def make_frame(i):
            snapshot = snapshots[i]
            phi = snapshot['phi']
            if use_abs:
                phi = np.abs(phi)
            step = snapshot['step']
            
            # Clear previous frame
            plt.clf()
            ax = fig.add_subplot(111, projection='3d')
            
            try:
                # Using scikit-image's marching_cubes algorithm
                verts, faces, _, _ = measure.marching_cubes(phi, level)
                
                # Scale vertices to physical coordinates
                verts[:, 0] = verts[:, 0] * self.grid.dx
                verts[:, 1] = verts[:, 1] * self.grid.dy
                verts[:, 2] = verts[:, 2] * self.grid.dz
                
                # Create polygon collection
                mesh = Poly3DCollection(verts[faces], alpha=0.5, color='blue')
                ax.add_collection3d(mesh)
                
                # If we're using absolute values, try to plot the negative isosurface as well
                if not use_abs:
                    try:
                        neg_verts, neg_faces, _, _ = measure.marching_cubes(phi, -level)
                        
                        # Scale vertices to physical coordinates
                        neg_verts[:, 0] = neg_verts[:, 0] * self.grid.dx
                        neg_verts[:, 1] = neg_verts[:, 1] * self.grid.dy
                        neg_verts[:, 2] = neg_verts[:, 2] * self.grid.dz
                        
                        # Create polygon collection for negative values
                        neg_mesh = Poly3DCollection(neg_verts[neg_faces], alpha=0.5, color='red')
                        ax.add_collection3d(neg_mesh)
                    except Exception:
                        # It's okay if this fails, we might not have negative values
                        pass
                
                # Set plot limits and labels
                ax.set_xlim(0, self.grid.x_length)
                ax.set_ylim(0, self.grid.y_length)
                ax.set_zlim(0, self.grid.z_length)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title(f'Scalar Potential ϕ Isosurface (Step {step})')
                
                # Choose view based on frame number for a rotating effect
                # Start with a fixed elevation for stability
                elev = 30
                # Rotate the azimuth for a spinning effect, but not too fast
                azim = (i * 2) % 360
                
                ax.view_init(elev=elev, azim=azim)
                
            except Exception as e:
                print(f"Warning: Could not create isosurface for frame {i}. Error: {str(e)}")
                ax.text(0.5, 0.5, 0.5, "No isosurface available", 
                       horizontalalignment='center', verticalalignment='center')
                
            return [ax]
        
        # Create animation with a fixed frame rate
        anim = animation.FuncAnimation(fig, make_frame, frames=len(snapshots), interval=200, blit=False)
        
        # Save animation
        writer = animation.FFMpegWriter(fps=10)
        volume_filename = os.path.join(self.output_dir, save_filename)
        anim.save(volume_filename, writer=writer)
        plt.close(fig)
        print(f"3D volume animation saved to {volume_filename}")
    
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
            # Create the standard 2D slice animation
            self.animate_snapshots_3d(snapshots, save_filename)
            
            # Also create a 3D volume animation if volume rendering is enabled
            if self.volume_rendering:
                self.animate_volume_snapshots(snapshots, "volume_animation.mp4")

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
