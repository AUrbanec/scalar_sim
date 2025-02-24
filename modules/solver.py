# modules/solver.py

import numpy as np

class Solver:
    def __init__(self, config, grid, phi, A):
        self.config = config
        self.grid = grid

        # Field initialization
        self.phi_current = phi.copy()
        self.Ax_current = A[0].copy()
        self.Ay_current = A[1].copy()

        self.phi_old = phi.copy()
        self.Ax_old = A[0].copy()
        self.Ay_old = A[1].copy()

        time_config = config.get('simulation', {}).get('time', {})
        self.dt = time_config.get('dt', 0.001)
        self.total_time = time_config.get('total_time', 1.0)
        self.num_steps = int(self.total_time / self.dt)

        material_config = config.get('simulation', {}).get('material', {})
        epsilon = material_config.get('epsilon', 1.0)
        mu = material_config.get('mu', 1.0)
        self.c = 1.0 / np.sqrt(epsilon * mu)

        self.dx, self.dy = grid.get_spacing()

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

    def laplacian(self, field):
        lap = np.zeros_like(field)
        lap[1:-1, 1:-1] = (
            (field[2:, 1:-1] - 2 * field[1:-1, 1:-1] + field[:-2, 1:-1]) / self.dx**2 +
            (field[1:-1, 2:] - 2 * field[1:-1, 1:-1] + field[1:-1, :-2]) / self.dy**2
        )
        return lap

    def compute_source_term(self, t):
        if not self.source_enabled:
            return 0.0
        s_config = self.source_config
        amplitude = s_config.get('amplitude', 1.0)
        frequency = s_config.get('frequency', 50.0)
        center = s_config.get('center', [0.5, 0.5])
        sigma = s_config.get('sigma', 0.05)
        modulation = s_config.get('modulation', 'amplitude')
        X, Y = self.grid.get_meshgrid()
        source_spatial = np.exp(-(((X - center[0])**2 + (Y - center[1])**2) / (2 * sigma**2)))
        if modulation == 'amplitude':
            source_time = np.sin(2 * np.pi * frequency * t)
        elif modulation == 'frequency':
            source_time = np.sin(2 * np.pi * frequency * t * (1 + 0.1 * np.sin(2 * np.pi * frequency * t)))
        else:
            source_time = 1.0
        return amplitude * source_time * source_spatial

    def apply_dirichlet_bc(self, field, field_name="phi"):
        if field_name == "phi":
            field[0, :] = field[1, :]
            field[-1, :] = field[-2, :]
            field[:, 0] = field[:, 1]
            field[:, -1] = field[:, -2]
        else:
            field[0, :] = 0.0
            field[-1, :] = 0.0
            field[:, 0] = 0.0
            field[:, -1] = 0.0
        return field

    def apply_neumann_bc(self, field):
        field[0, :] = field[1, :]
        field[-1, :] = field[-2, :]
        field[:, 0] = field[:, 1]
        field[:, -1] = field[:, -2]
        return field

    def apply_impedance_bc(self, field):
        field[-1, :] = field[-2, :] - self.impedance * (field[-2, :] - field[-3, :])
        field[0, :] = field[1, :] + self.impedance * (field[2, :] - field[1, :])
        field[:, 0] = field[:, 1] + self.impedance * (field[:, 2] - field[:, 1])
        field[:, -1] = field[:, -2] - self.impedance * (field[:, -2] - field[:, -3])
        return field

    def apply_boundary_conditions(self, field, field_name="phi"):
        if self.bc_type == 'dirichlet':
            return self.apply_dirichlet_bc(field, field_name)
        elif self.bc_type == 'neumann':
            return self.apply_neumann_bc(field)
        elif self.bc_type == 'advanced':
            return self.apply_impedance_bc(field)
        else:
            return self.apply_neumann_bc(field)

    def step(self):
        lap_phi = self.laplacian(self.phi_current)
        lap_Ax = self.laplacian(self.Ax_current)
        lap_Ay = self.laplacian(self.Ay_current)

        source_term = self.compute_source_term(self.current_time) if self.source_enabled else 0.0

        if self.nonlinearity_enabled:
            non_linear_term = self.nonlinearity_strength * (self.phi_current ** 3)
        else:
            non_linear_term = 0.0

        phi_new = (2 * self.phi_current - self.phi_old +
                   (self.c**2 * self.dt**2) * lap_phi +
                   self.dt**2 * source_term +
                   self.dt**2 * non_linear_term)
        Ax_new = 2 * self.Ax_current - self.Ax_old + (self.c**2 * self.dt**2) * lap_Ax
        Ay_new = 2 * self.Ay_current - self.Ay_old + (self.c**2 * self.dt**2) * lap_Ay

        phi_new = self.apply_boundary_conditions(phi_new, "phi")
        Ax_new = self.apply_boundary_conditions(Ax_new, "Ax")
        Ay_new = self.apply_boundary_conditions(Ay_new, "Ay")

        self.phi_old = self.phi_current.copy()
        self.Ax_old = self.Ax_current.copy()
        self.Ay_old = self.Ay_current.copy()

        self.phi_current = phi_new.copy()
        self.Ax_current = Ax_new.copy()
        self.Ay_current = Ay_new.copy()

        self.current_time += self.dt

    def run(self):
        for step in range(self.num_steps):
            self.step()
            if step % self.snapshot_interval == 0:
                snapshot = {
                    'step': step,
                    'time': self.current_time,
                    'phi': self.phi_current.copy(),
                    'Ax': self.Ax_current.copy(),
                    'Ay': self.Ay_current.copy()
                }
                self.snapshots.append(snapshot)
        return self.snapshots
