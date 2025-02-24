# modules/solver.py

import numpy as np
import os

class Solver:
    def __init__(self, config, grid, phi, A):
        """
        Initialize the solver with configuration, grid and initial fields.

        Parameters:
            config (dict): Simulation configuration dictionary.
            grid (Grid): An instance of the Grid class containing meshgrid and spacing information.
            phi (np.ndarray): Initial scalar potential field.
            A (tuple of np.ndarray): Initial vector potential (A_x, A_y) as a tuple.
        """
        self.config = config
        self.grid = grid

        # Current fields
        self.phi_current = phi.copy()
        self.Ax_current = A[0].copy()
        self.Ay_current = A[1].copy()

        # Assume initial time derivative is zero; use the same field for the previous time step.
        self.phi_old = phi.copy()
        self.Ax_old = A[0].copy()
        self.Ay_old = A[1].copy()

        # Retrieve simulation time parameters.
        time_config = config.get('simulation', {}).get('time', {})
        self.dt = time_config.get('dt', 0.001)
        self.total_time = time_config.get('total_time', 1.0)
        self.num_steps = int(self.total_time / self.dt)

        # Retrieve material properties and calculate the wave speed.
        material_config = config.get('simulation', {}).get('material', {})
        epsilon = material_config.get('epsilon', 8.854e-12)
        mu = material_config.get('mu', 1.25663706e-6)
        self.c = 1.0 / np.sqrt(epsilon * mu)  # Speed of light in the medium

        # Retrieve grid spacing.
        self.dx, self.dy = grid.get_spacing()

        # Retrieve boundary condition type.
        self.bc_type = config.get('simulation', {}).get('boundary_conditions', {}).get('type', 'dirichlet')

        # Snapshot configuration.
        self.snapshot_interval = config.get('simulation', {}).get('output', {}).get('snapshot_interval', 50)
        self.snapshots = []  # Container to store snapshots during simulation

    def laplacian(self, field):
        """
        Compute the Laplacian of a 2D field using central differences.

        Parameters:
            field (np.ndarray): 2D array representing the field.

        Returns:
            np.ndarray: 2D array representing the Laplacian of the field.
        """
        lap = np.zeros_like(field)
        # Compute second-order differences for interior points.
        lap[1:-1, 1:-1] = (
            (field[2:, 1:-1] - 2 * field[1:-1, 1:-1] + field[:-2, 1:-1]) / self.dx**2 +
            (field[1:-1, 2:] - 2 * field[1:-1, 1:-1] + field[1:-1, :-2]) / self.dy**2
        )
        return lap

    def apply_boundary_conditions(self, field, field_name="phi"):
        """
        Apply boundary conditions to a field array.
        
        For the scalar potential (phi), a Neumann (zero-gradient) condition is applied,
        ensuring the scalar mode is preserved. For the vector potential components (Ax, Ay),
        if the boundary condition type is 'dirichlet', they are forced to zero at the edges.

        Parameters:
            field (np.ndarray): 2D field array.
            field_name (str): Name of the field ('phi', 'Ax', or 'Ay').

        Returns:
            np.ndarray: Field array after applying boundary conditions.
        """
        new_field = field.copy()
        if self.bc_type.lower() == 'dirichlet':
            if field_name == "phi":
                # For phi, apply a zero-gradient (Neumann) boundary condition.
                new_field[0, :] = new_field[1, :]
                new_field[-1, :] = new_field[-2, :]
                new_field[:, 0] = new_field[:, 1]
                new_field[:, -1] = new_field[:, -2]
            else:
                # For vector potential components, enforce Dirichlet (zero) boundaries.
                new_field[0, :] = 0.0
                new_field[-1, :] = 0.0
                new_field[:, 0] = 0.0
                new_field[:, -1] = 0.0
        elif self.bc_type.lower() == 'neumann':
            # Apply Neumann (zero-gradient) boundaries for all fields.
            new_field[0, :] = new_field[1, :]
            new_field[-1, :] = new_field[-2, :]
            new_field[:, 0] = new_field[:, 1]
            new_field[:, -1] = new_field[:, -2]
        else:
            # Default to Neumann if the boundary condition type is unknown.
            new_field[0, :] = new_field[1, :]
            new_field[-1, :] = new_field[-2, :]
            new_field[:, 0] = new_field[:, 1]
            new_field[:, -1] = new_field[:, -2]
        return new_field

    def step(self):
        """
        Perform one time step update for the scalar and vector potentials using a leapfrog scheme.
        """
        # Compute Laplacians for current fields.
        lap_phi = self.laplacian(self.phi_current)
        lap_Ax = self.laplacian(self.Ax_current)
        lap_Ay = self.laplacian(self.Ay_current)

        # Update equations for the wave equation:
        # new_field = 2 * current_field - old_field + c^2 * dt^2 * Laplacian(current_field)
        phi_new = 2 * self.phi_current - self.phi_old + (self.c**2 * self.dt**2) * lap_phi
        Ax_new = 2 * self.Ax_current - self.Ax_old + (self.c**2 * self.dt**2) * lap_Ax
        Ay_new = 2 * self.Ay_current - self.Ay_old + (self.c**2 * self.dt**2) * lap_Ay

        # Apply boundary conditions.
        phi_new = self.apply_boundary_conditions(phi_new, "phi")
        Ax_new = self.apply_boundary_conditions(Ax_new, "Ax")
        Ay_new = self.apply_boundary_conditions(Ay_new, "Ay")

        # Update previous fields for next time step.
        self.phi_old = self.phi_current.copy()
        self.Ax_old = self.Ax_current.copy()
        self.Ay_old = self.Ay_current.copy()

        # Set the new fields as current.
        self.phi_current = phi_new.copy()
        self.Ax_current = Ax_new.copy()
        self.Ay_current = Ay_new.copy()

    def run(self):
        """
        Execute the time-stepping loop for the simulation.
        Snapshots of the fields are stored at intervals defined by snapshot_interval.

        Returns:
            list: A list of snapshots. Each snapshot is a dictionary containing the step index
                  and copies of the current phi, Ax, and Ay fields.
        """
        for step in range(self.num_steps):
            self.step()

            # Store snapshot if current step is at the defined interval.
            if step % self.snapshot_interval == 0:
                snapshot = {
                    'step': step,
                    'phi': self.phi_current.copy(),
                    'Ax': self.Ax_current.copy(),
                    'Ay': self.Ay_current.copy()
                }
                self.snapshots.append(snapshot)

        return self.snapshots
