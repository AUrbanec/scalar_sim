# modules/solver.py

import numpy as np

class Solver:
    def __init__(self, config, grid, phi, A):
        """
        Initialize the solver for scalar wave simulation
        
        Parameters
        ----------
        config : dict
            Configuration dictionary containing simulation parameters
        grid : Grid
            Grid object with mesh information
        phi : ndarray
            Initial scalar potential field
        A : tuple of ndarrays
            Initial vector potential field components
        """
        self.config = config
        self.grid = grid
        self.dimension = config.get('simulation', {}).get('dimension', 3)

        # Field initialization
        self.phi_current = phi.copy()
        self.phi_old = phi.copy()
        
        # Initialize vector potential components
        if self.dimension == 2:
            self.Ax_current = A[0].copy()
            self.Ay_current = A[1].copy()
            self.Ax_old = A[0].copy()
            self.Ay_old = A[1].copy()
            self.Az_current = None
            self.Az_old = None
        else:  # 3D case
            self.Ax_current = A[0].copy()
            self.Ay_current = A[1].copy()
            self.Az_current = A[2].copy()
            self.Ax_old = A[0].copy()
            self.Ay_old = A[1].copy()
            self.Az_old = A[2].copy()

        # Time configuration
        time_config = config.get('simulation', {}).get('time', {})
        self.dt = time_config.get('dt', 0.001)
        self.total_time = time_config.get('total_time', 1.0)
        self.num_steps = int(self.total_time / self.dt)

        # Material properties
        material_config = config.get('simulation', {}).get('material', {})
        epsilon = material_config.get('epsilon', 1.0)
        mu = material_config.get('mu', 1.0)
        self.c = 1.0 / np.sqrt(epsilon * mu)

        # Grid spacing
        if self.dimension == 2:
            self.dx, self.dy = grid.get_spacing()
            self.dz = None
        else:
            self.dx, self.dy, self.dz = grid.get_spacing()

        # Boundary conditions
        bc_config = config.get('simulation', {}).get('boundary_conditions', {})
        self.bc_type = bc_config.get('type', 'dirichlet').lower()
        self.impedance = bc_config.get('impedance', 0.1)

        # Source term configuration
        source_config = config.get('simulation', {}).get('source', {})
        self.source_enabled = source_config.get('enabled', False)
        self.source_config = source_config

        # Nonlinearity configuration
        nonlin_config = config.get('simulation', {}).get('nonlinearity', {})
        self.nonlinearity_enabled = nonlin_config.get('enabled', False)
        self.nonlinearity_strength = nonlin_config.get('strength', 0.0)

        # Snapshot configuration
        self.snapshot_interval = config.get('simulation', {}).get('output', {}).get('snapshot_interval', 50)
        self.snapshots = []
        self.current_time = 0.0

    def laplacian_2d(self, field):
        """
        Compute the 2D Laplacian of the field
        
        Parameters
        ----------
        field : ndarray
            Input field for which to compute the Laplacian
            
        Returns
        -------
        ndarray
            Laplacian of the field
        """
        lap = np.zeros_like(field)
        lap[1:-1, 1:-1] = (
            (field[2:, 1:-1] - 2 * field[1:-1, 1:-1] + field[:-2, 1:-1]) / self.dx**2 +
            (field[1:-1, 2:] - 2 * field[1:-1, 1:-1] + field[1:-1, :-2]) / self.dy**2
        )
        return lap
        
    def laplacian_3d(self, field):
        """
        Compute the 3D Laplacian of the field
        
        Parameters
        ----------
        field : ndarray
            Input field for which to compute the Laplacian
            
        Returns
        -------
        ndarray
            Laplacian of the field
        """
        lap = np.zeros_like(field)
        lap[1:-1, 1:-1, 1:-1] = (
            (field[2:, 1:-1, 1:-1] - 2 * field[1:-1, 1:-1, 1:-1] + field[:-2, 1:-1, 1:-1]) / self.dx**2 +
            (field[1:-1, 2:, 1:-1] - 2 * field[1:-1, 1:-1, 1:-1] + field[1:-1, :-2, 1:-1]) / self.dy**2 +
            (field[1:-1, 1:-1, 2:] - 2 * field[1:-1, 1:-1, 1:-1] + field[1:-1, 1:-1, :-2]) / self.dz**2
        )
        return lap
        
    def laplacian(self, field):
        """
        Compute the Laplacian of the field (2D or 3D)
        
        Parameters
        ----------
        field : ndarray
            Input field for which to compute the Laplacian
            
        Returns
        -------
        ndarray
            Laplacian of the field
        """
        if self.dimension == 2:
            return self.laplacian_2d(field)
        else:
            return self.laplacian_3d(field)

    def compute_source_term(self, t):
        """
        Compute the source term for the scalar wave equation
        
        Parameters
        ----------
        t : float
            Current time
            
        Returns
        -------
        ndarray
            Source term field
        """
        if not self.source_enabled:
            return 0.0
            
        s_config = self.source_config
        amplitude = s_config.get('amplitude', 1.0)
        frequency = s_config.get('frequency', 50.0)
        modulation = s_config.get('modulation', 'amplitude')
        
        # Calculate time component of source
        if modulation == 'amplitude':
            source_time = np.sin(2 * np.pi * frequency * t)
        elif modulation == 'frequency':
            source_time = np.sin(2 * np.pi * frequency * t * (1 + 0.1 * np.sin(2 * np.pi * frequency * t)))
        else:
            source_time = 1.0
        
        # Calculate spatial component of source
        if self.dimension == 2:
            center = s_config.get('center', [0.5, 0.5])
            sigma = s_config.get('sigma', 0.05)
            X, Y = self.grid.get_meshgrid()
            source_spatial = np.exp(-(((X - center[0])**2 + (Y - center[1])**2) / (2 * sigma**2)))
        else:  # 3D case
            center = s_config.get('center', [0.5, 0.5, 0.5])
            sigma = s_config.get('sigma', 0.05)
            X, Y, Z = self.grid.get_meshgrid()
            source_spatial = np.exp(-(
                ((X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2) / (2 * sigma**2)
            ))
            
        return amplitude * source_time * source_spatial

    def apply_dirichlet_bc_2d(self, field, field_name="phi"):
        """
        Apply Dirichlet boundary conditions in 2D
        
        Parameters
        ----------
        field : ndarray
            Field to apply boundary conditions to
        field_name : str
            Name of the field ("phi", "Ax", "Ay", "Az")
            
        Returns
        -------
        ndarray
            Field with boundary conditions applied
        """
        if field_name == "phi":
            field[0, :] = field[1, :]
            field[-1, :] = field[-2, :]
            field[:, 0] = field[:, 1]
            field[:, -1] = field[:, -2]
        else:  # Vector potential components
            field[0, :] = 0.0
            field[-1, :] = 0.0
            field[:, 0] = 0.0
            field[:, -1] = 0.0
        return field
        
    def apply_dirichlet_bc_3d(self, field, field_name="phi"):
        """
        Apply Dirichlet boundary conditions in 3D
        
        Parameters
        ----------
        field : ndarray
            Field to apply boundary conditions to
        field_name : str
            Name of the field ("phi", "Ax", "Ay", "Az")
            
        Returns
        -------
        ndarray
            Field with boundary conditions applied
        """
        if field_name == "phi":
            field[0, :, :] = field[1, :, :]
            field[-1, :, :] = field[-2, :, :]
            field[:, 0, :] = field[:, 1, :]
            field[:, -1, :] = field[:, -2, :]
            field[:, :, 0] = field[:, :, 1]
            field[:, :, -1] = field[:, :, -2]
        else:  # Vector potential components
            field[0, :, :] = 0.0
            field[-1, :, :] = 0.0
            field[:, 0, :] = 0.0
            field[:, -1, :] = 0.0
            field[:, :, 0] = 0.0
            field[:, :, -1] = 0.0
        return field

    def apply_neumann_bc_2d(self, field):
        """
        Apply Neumann boundary conditions in 2D
        
        Parameters
        ----------
        field : ndarray
            Field to apply boundary conditions to
            
        Returns
        -------
        ndarray
            Field with boundary conditions applied
        """
        field[0, :] = field[1, :]
        field[-1, :] = field[-2, :]
        field[:, 0] = field[:, 1]
        field[:, -1] = field[:, -2]
        return field
        
    def apply_neumann_bc_3d(self, field):
        """
        Apply Neumann boundary conditions in 3D
        
        Parameters
        ----------
        field : ndarray
            Field to apply boundary conditions to
            
        Returns
        -------
        ndarray
            Field with boundary conditions applied
        """
        field[0, :, :] = field[1, :, :]
        field[-1, :, :] = field[-2, :, :]
        field[:, 0, :] = field[:, 1, :]
        field[:, -1, :] = field[:, -2, :]
        field[:, :, 0] = field[:, :, 1]
        field[:, :, -1] = field[:, :, -2]
        return field

    def apply_impedance_bc_2d(self, field):
        """
        Apply impedance boundary conditions in 2D
        
        Parameters
        ----------
        field : ndarray
            Field to apply boundary conditions to
            
        Returns
        -------
        ndarray
            Field with boundary conditions applied
        """
        field[-1, :] = field[-2, :] - self.impedance * (field[-2, :] - field[-3, :])
        field[0, :] = field[1, :] + self.impedance * (field[2, :] - field[1, :])
        field[:, 0] = field[:, 1] + self.impedance * (field[:, 2] - field[:, 1])
        field[:, -1] = field[:, -2] - self.impedance * (field[:, -2] - field[:, -3])
        return field
        
    def apply_impedance_bc_3d(self, field):
        """
        Apply impedance boundary conditions in 3D
        
        Parameters
        ----------
        field : ndarray
            Field to apply boundary conditions to
            
        Returns
        -------
        ndarray
            Field with boundary conditions applied
        """
        field[-1, :, :] = field[-2, :, :] - self.impedance * (field[-2, :, :] - field[-3, :, :])
        field[0, :, :] = field[1, :, :] + self.impedance * (field[2, :, :] - field[1, :, :])
        field[:, 0, :] = field[:, 1, :] + self.impedance * (field[:, 2, :] - field[:, 1, :])
        field[:, -1, :] = field[:, -2, :] - self.impedance * (field[:, -2, :] - field[:, -3, :])
        field[:, :, 0] = field[:, :, 1] + self.impedance * (field[:, :, 2] - field[:, :, 1])
        field[:, :, -1] = field[:, :, -2] - self.impedance * (field[:, :, -2] - field[:, :, -3])
        return field

    def apply_boundary_conditions(self, field, field_name="phi"):
        """
        Apply boundary conditions to a field
        
        Parameters
        ----------
        field : ndarray
            Field to apply boundary conditions to
        field_name : str
            Name of the field ("phi", "Ax", "Ay", "Az")
            
        Returns
        -------
        ndarray
            Field with boundary conditions applied
        """
        if self.dimension == 2:
            if self.bc_type == 'dirichlet':
                return self.apply_dirichlet_bc_2d(field, field_name)
            elif self.bc_type == 'neumann':
                return self.apply_neumann_bc_2d(field)
            elif self.bc_type == 'advanced':
                return self.apply_impedance_bc_2d(field)
            else:
                return self.apply_neumann_bc_2d(field)
        else:  # 3D case
            if self.bc_type == 'dirichlet':
                return self.apply_dirichlet_bc_3d(field, field_name)
            elif self.bc_type == 'neumann':
                return self.apply_neumann_bc_3d(field)
            elif self.bc_type == 'advanced':
                return self.apply_impedance_bc_3d(field)
            else:
                return self.apply_neumann_bc_3d(field)

    def step(self):
        """
        Perform a single time step in the simulation
        """
        # Compute Laplacians
        lap_phi = self.laplacian(self.phi_current)
        lap_Ax = self.laplacian(self.Ax_current)
        lap_Ay = self.laplacian(self.Ay_current)
        if self.dimension == 3:
            lap_Az = self.laplacian(self.Az_current)

        # Source term and nonlinearity
        source_term = self.compute_source_term(self.current_time) if self.source_enabled else 0.0
        
        if self.nonlinearity_enabled:
            non_linear_term = self.nonlinearity_strength * (self.phi_current ** 3)
        else:
            non_linear_term = 0.0

        # Update scalar potential
        phi_new = (2 * self.phi_current - self.phi_old +
                   (self.c**2 * self.dt**2) * lap_phi +
                   self.dt**2 * source_term +
                   self.dt**2 * non_linear_term)
        
        # Update vector potential
        Ax_new = 2 * self.Ax_current - self.Ax_old + (self.c**2 * self.dt**2) * lap_Ax
        Ay_new = 2 * self.Ay_current - self.Ay_old + (self.c**2 * self.dt**2) * lap_Ay
        if self.dimension == 3:
            Az_new = 2 * self.Az_current - self.Az_old + (self.c**2 * self.dt**2) * lap_Az

        # Apply boundary conditions
        phi_new = self.apply_boundary_conditions(phi_new, "phi")
        Ax_new = self.apply_boundary_conditions(Ax_new, "Ax")
        Ay_new = self.apply_boundary_conditions(Ay_new, "Ay")
        if self.dimension == 3:
            Az_new = self.apply_boundary_conditions(Az_new, "Az")

        # Update fields
        self.phi_old = self.phi_current.copy()
        self.Ax_old = self.Ax_current.copy()
        self.Ay_old = self.Ay_current.copy()
        if self.dimension == 3:
            self.Az_old = self.Az_current.copy()

        self.phi_current = phi_new.copy()
        self.Ax_current = Ax_new.copy()
        self.Ay_current = Ay_new.copy()
        if self.dimension == 3:
            self.Az_current = Az_new.copy()

        self.current_time += self.dt

    def run(self):
        """
        Run the full simulation
        
        Returns
        -------
        list
            List of snapshot dictionaries
        """
        for step in range(self.num_steps):
            self.step()
            if step % self.snapshot_interval == 0:
                if self.dimension == 2:
                    snapshot = {
                        'step': step,
                        'time': self.current_time,
                        'phi': self.phi_current.copy(),
                        'Ax': self.Ax_current.copy(),
                        'Ay': self.Ay_current.copy()
                    }
                else:  # 3D case
                    snapshot = {
                        'step': step,
                        'time': self.current_time,
                        'phi': self.phi_current.copy(),
                        'Ax': self.Ax_current.copy(),
                        'Ay': self.Ay_current.copy(),
                        'Az': self.Az_current.copy()
                    }
                self.snapshots.append(snapshot)
        return self.snapshots
