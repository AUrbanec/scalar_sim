# modules/solver.py

from abc import ABC, abstractmethod
import numpy as np
import numba as nb
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import numpy.typing as npt

@nb.njit(parallel=True)
def laplacian_2d_optimized(field: npt.NDArray[np.float64], 
                          dx: float, 
                          dy: float) -> npt.NDArray[np.float64]:
    """
    Numba-optimized 2D Laplacian calculation
    """
    lap = np.zeros_like(field)
    nx, ny = field.shape
    
    for i in nb.prange(1, nx-1):
        for j in range(1, ny-1):
            lap[i, j] = (
                (field[i+1, j] - 2 * field[i, j] + field[i-1, j]) / (dx*dx) +
                (field[i, j+1] - 2 * field[i, j] + field[i, j-1]) / (dy*dy)
            )
    return lap

@nb.njit(parallel=True)
def laplacian_3d_optimized(field: npt.NDArray[np.float64], 
                          dx: float, 
                          dy: float,
                          dz: float) -> npt.NDArray[np.float64]:
    """
    Numba-optimized 3D Laplacian calculation
    """
    lap = np.zeros_like(field)
    nx, ny, nz = field.shape
    
    for i in nb.prange(1, nx-1):
        for j in range(1, ny-1):
            for k in range(1, nz-1):
                lap[i, j, k] = (
                    (field[i+1, j, k] - 2 * field[i, j, k] + field[i-1, j, k]) / (dx*dx) +
                    (field[i, j+1, k] - 2 * field[i, j, k] + field[i, j-1, k]) / (dy*dy) +
                    (field[i, j, k+1] - 2 * field[i, j, k] + field[i, j, k-1]) / (dz*dz)
                )
    return lap

class BaseSolver(ABC):
    """
    Base abstract solver class for scalar wave simulations.
    Provides a common interface for both 2D and 3D solvers.
    """
    
    def __init__(self, config: Dict[str, Any], 
                grid: Any, 
                phi: npt.NDArray[np.float64], 
                A: Tuple[npt.NDArray[np.float64], ...]) -> None:
        """
        Initialize common solver parameters.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary containing simulation parameters
        grid : Any
            Grid object with mesh information
        phi : npt.NDArray[np.float64]
            Initial scalar potential field
        A : Tuple[npt.NDArray[np.float64], ...]
            Initial vector potential field components
        """
        self.config = config
        self.grid = grid
        
        # Field initialization
        self.phi_current = phi.copy()
        self.phi_old = phi.copy()
        
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
        self.snapshots: List[Dict[str, Any]] = []
        self.current_time = 0.0
    
    @abstractmethod
    def laplacian(self, field: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Abstract method to compute Laplacian.
        Must be implemented by subclasses.
        
        Parameters
        ----------
        field : npt.NDArray[np.float64]
            Input field for which to compute the Laplacian
            
        Returns
        -------
        npt.NDArray[np.float64]
            Laplacian of the field
        """
        pass
    
    @abstractmethod
    def apply_boundary_conditions(self, field: npt.NDArray[np.float64], 
                               field_name: str = "phi") -> npt.NDArray[np.float64]:
        """
        Abstract method to apply boundary conditions.
        Must be implemented by subclasses.
        
        Parameters
        ----------
        field : npt.NDArray[np.float64]
            Field to apply boundary conditions to
        field_name : str
            Name of the field ("phi", "Ax", "Ay", "Az")
            
        Returns
        -------
        npt.NDArray[np.float64]
            Field with boundary conditions applied
        """
        pass
    
    @abstractmethod
    def compute_source_term(self, t: float) -> Union[float, npt.NDArray[np.float64]]:
        """
        Abstract method to compute the source term.
        Must be implemented by subclasses.
        
        Parameters
        ----------
        t : float
            Current time
            
        Returns
        -------
        Union[float, npt.NDArray[np.float64]]
            Source term field or scalar
        """
        pass
    
    @abstractmethod
    def step(self) -> None:
        """
        Abstract method to perform a single time step.
        Must be implemented by subclasses.
        """
        pass
    
    def run(self) -> List[Dict[str, Any]]:
        """
        Run the full simulation.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of snapshot dictionaries
        """
        for step in range(self.num_steps):
            self.step()
            if step % self.snapshot_interval == 0:
                snapshot = self._create_snapshot(step)
                self.snapshots.append(snapshot)
        return self.snapshots
    
    @abstractmethod
    def _create_snapshot(self, step: int) -> Dict[str, Any]:
        """
        Abstract method to create a snapshot dictionary.
        Must be implemented by subclasses.
        
        Parameters
        ----------
        step : int
            Current time step
            
        Returns
        -------
        Dict[str, Any]
            Snapshot dictionary
        """
        pass


class Solver2D(BaseSolver):
    """
    2D implementation of the wave solver.
    """
    
    def __init__(self, config: Dict[str, Any], 
                grid: Any, 
                phi: npt.NDArray[np.float64], 
                A: Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]) -> None:
        """
        Initialize 2D solver.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary containing simulation parameters
        grid : Any
            Grid object with mesh information
        phi : npt.NDArray[np.float64]
            Initial scalar potential field
        A : Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
            Initial vector potential field components (Ax, Ay)
        """
        super().__init__(config, grid, phi, A)
        
        # Initialize vector potential components for 2D
        self.Ax_current = A[0].copy()
        self.Ay_current = A[1].copy()
        self.Ax_old = A[0].copy()
        self.Ay_old = A[1].copy()
        
        # Grid spacing for 2D
        self.dx, self.dy = grid.get_spacing()
    
    def laplacian(self, field: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Compute the 2D Laplacian of the field using Numba optimization.
        
        Parameters
        ----------
        field : npt.NDArray[np.float64]
            Input field for which to compute the Laplacian
            
        Returns
        -------
        npt.NDArray[np.float64]
            Laplacian of the field
        """
        return laplacian_2d_optimized(field, self.dx, self.dy)
    
    def apply_dirichlet_bc(self, field: npt.NDArray[np.float64], 
                         field_name: str = "phi") -> npt.NDArray[np.float64]:
        """
        Apply Dirichlet boundary conditions in 2D.
        
        Parameters
        ----------
        field : npt.NDArray[np.float64]
            Field to apply boundary conditions to
        field_name : str
            Name of the field ("phi", "Ax", "Ay")
            
        Returns
        -------
        npt.NDArray[np.float64]
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
    
    def apply_neumann_bc(self, field: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Apply Neumann boundary conditions in 2D.
        
        Parameters
        ----------
        field : npt.NDArray[np.float64]
            Field to apply boundary conditions to
            
        Returns
        -------
        npt.NDArray[np.float64]
            Field with boundary conditions applied
        """
        field[0, :] = field[1, :]
        field[-1, :] = field[-2, :]
        field[:, 0] = field[:, 1]
        field[:, -1] = field[:, -2]
        return field
    
    def apply_impedance_bc(self, field: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Apply impedance boundary conditions in 2D.
        
        Parameters
        ----------
        field : npt.NDArray[np.float64]
            Field to apply boundary conditions to
            
        Returns
        -------
        npt.NDArray[np.float64]
            Field with boundary conditions applied
        """
        field[-1, :] = field[-2, :] - self.impedance * (field[-2, :] - field[-3, :])
        field[0, :] = field[1, :] + self.impedance * (field[2, :] - field[1, :])
        field[:, 0] = field[:, 1] + self.impedance * (field[:, 2] - field[:, 1])
        field[:, -1] = field[:, -2] - self.impedance * (field[:, -2] - field[:, -3])
        return field
    
    def apply_boundary_conditions(self, field: npt.NDArray[np.float64], 
                               field_name: str = "phi") -> npt.NDArray[np.float64]:
        """
        Apply boundary conditions to a field in 2D.
        
        Parameters
        ----------
        field : npt.NDArray[np.float64]
            Field to apply boundary conditions to
        field_name : str
            Name of the field ("phi", "Ax", "Ay")
            
        Returns
        -------
        npt.NDArray[np.float64]
            Field with boundary conditions applied
        """
        if self.bc_type == 'dirichlet':
            return self.apply_dirichlet_bc(field, field_name)
        elif self.bc_type == 'neumann':
            return self.apply_neumann_bc(field)
        elif self.bc_type == 'advanced':
            return self.apply_impedance_bc(field)
        else:
            return self.apply_neumann_bc(field)
    
    def compute_source_term(self, t: float) -> Union[float, npt.NDArray[np.float64]]:
        """
        Compute the source term for the scalar wave equation in 2D.
        
        Parameters
        ----------
        t : float
            Current time
            
        Returns
        -------
        Union[float, npt.NDArray[np.float64]]
            Source term field or scalar
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
        center = s_config.get('center', [0.5, 0.5])
        sigma = s_config.get('sigma', 0.05)
        X, Y = self.grid.get_meshgrid()
        source_spatial = np.exp(-(((X - center[0])**2 + (Y - center[1])**2) / (2 * sigma**2)))
            
        return amplitude * source_time * source_spatial
    
    def step(self) -> None:
        """
        Perform a single time step in the 2D simulation.
        """
        # Compute Laplacians
        lap_phi = self.laplacian(self.phi_current)
        lap_Ax = self.laplacian(self.Ax_current)
        lap_Ay = self.laplacian(self.Ay_current)

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

        # Apply boundary conditions
        phi_new = self.apply_boundary_conditions(phi_new, "phi")
        Ax_new = self.apply_boundary_conditions(Ax_new, "Ax")
        Ay_new = self.apply_boundary_conditions(Ay_new, "Ay")

        # Update fields
        self.phi_old = self.phi_current.copy()
        self.Ax_old = self.Ax_current.copy()
        self.Ay_old = self.Ay_current.copy()

        self.phi_current = phi_new.copy()
        self.Ax_current = Ax_new.copy()
        self.Ay_current = Ay_new.copy()

        self.current_time += self.dt
    
    def _create_snapshot(self, step: int) -> Dict[str, Any]:
        """
        Create a snapshot dictionary for the current state in 2D.
        
        Parameters
        ----------
        step : int
            Current time step
            
        Returns
        -------
        Dict[str, Any]
            Snapshot dictionary
        """
        return {
            'step': step,
            'time': self.current_time,
            'phi': self.phi_current.copy(),
            'Ax': self.Ax_current.copy(),
            'Ay': self.Ay_current.copy()
        }


class Solver3D(BaseSolver):
    """
    3D implementation of the wave solver.
    """
    
    def __init__(self, config: Dict[str, Any], 
                grid: Any, 
                phi: npt.NDArray[np.float64], 
                A: Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]) -> None:
        """
        Initialize 3D solver.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary containing simulation parameters
        grid : Any
            Grid object with mesh information
        phi : npt.NDArray[np.float64]
            Initial scalar potential field
        A : Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]
            Initial vector potential field components (Ax, Ay, Az)
        """
        super().__init__(config, grid, phi, A)
        
        # Initialize vector potential components for 3D
        self.Ax_current = A[0].copy()
        self.Ay_current = A[1].copy()
        self.Az_current = A[2].copy()
        self.Ax_old = A[0].copy()
        self.Ay_old = A[1].copy()
        self.Az_old = A[2].copy()
        
        # Grid spacing for 3D
        self.dx, self.dy, self.dz = grid.get_spacing()
    
    def laplacian(self, field: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Compute the 3D Laplacian of the field using Numba optimization.
        
        Parameters
        ----------
        field : npt.NDArray[np.float64]
            Input field for which to compute the Laplacian
            
        Returns
        -------
        npt.NDArray[np.float64]
            Laplacian of the field
        """
        return laplacian_3d_optimized(field, self.dx, self.dy, self.dz)
    
    def apply_dirichlet_bc(self, field: npt.NDArray[np.float64], 
                         field_name: str = "phi") -> npt.NDArray[np.float64]:
        """
        Apply Dirichlet boundary conditions in 3D.
        
        Parameters
        ----------
        field : npt.NDArray[np.float64]
            Field to apply boundary conditions to
        field_name : str
            Name of the field ("phi", "Ax", "Ay", "Az")
            
        Returns
        -------
        npt.NDArray[np.float64]
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
    
    def apply_neumann_bc(self, field: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Apply Neumann boundary conditions in 3D.
        
        Parameters
        ----------
        field : npt.NDArray[np.float64]
            Field to apply boundary conditions to
            
        Returns
        -------
        npt.NDArray[np.float64]
            Field with boundary conditions applied
        """
        field[0, :, :] = field[1, :, :]
        field[-1, :, :] = field[-2, :, :]
        field[:, 0, :] = field[:, 1, :]
        field[:, -1, :] = field[:, -2, :]
        field[:, :, 0] = field[:, :, 1]
        field[:, :, -1] = field[:, :, -2]
        return field
    
    def apply_impedance_bc(self, field: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Apply impedance boundary conditions in 3D.
        
        Parameters
        ----------
        field : npt.NDArray[np.float64]
            Field to apply boundary conditions to
            
        Returns
        -------
        npt.NDArray[np.float64]
            Field with boundary conditions applied
        """
        field[-1, :, :] = field[-2, :, :] - self.impedance * (field[-2, :, :] - field[-3, :, :])
        field[0, :, :] = field[1, :, :] + self.impedance * (field[2, :, :] - field[1, :, :])
        field[:, 0, :] = field[:, 1, :] + self.impedance * (field[:, 2, :] - field[:, 1, :])
        field[:, -1, :] = field[:, -2, :] - self.impedance * (field[:, -2, :] - field[:, -3, :])
        field[:, :, 0] = field[:, :, 1] + self.impedance * (field[:, :, 2] - field[:, :, 1])
        field[:, :, -1] = field[:, :, -2] - self.impedance * (field[:, :, -2] - field[:, :, -3])
        return field
    
    def apply_boundary_conditions(self, field: npt.NDArray[np.float64], 
                               field_name: str = "phi") -> npt.NDArray[np.float64]:
        """
        Apply boundary conditions to a field in 3D.
        
        Parameters
        ----------
        field : npt.NDArray[np.float64]
            Field to apply boundary conditions to
        field_name : str
            Name of the field ("phi", "Ax", "Ay", "Az")
            
        Returns
        -------
        npt.NDArray[np.float64]
            Field with boundary conditions applied
        """
        if self.bc_type == 'dirichlet':
            return self.apply_dirichlet_bc(field, field_name)
        elif self.bc_type == 'neumann':
            return self.apply_neumann_bc(field)
        elif self.bc_type == 'advanced':
            return self.apply_impedance_bc(field)
        else:
            return self.apply_neumann_bc(field)
    
    def compute_source_term(self, t: float) -> Union[float, npt.NDArray[np.float64]]:
        """
        Compute the source term for the scalar wave equation in 3D.
        
        Parameters
        ----------
        t : float
            Current time
            
        Returns
        -------
        Union[float, npt.NDArray[np.float64]]
            Source term field or scalar
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
        center = s_config.get('center', [0.5, 0.5, 0.5])
        sigma = s_config.get('sigma', 0.05)
        X, Y, Z = self.grid.get_meshgrid()
        source_spatial = np.exp(-(
            ((X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2) / (2 * sigma**2)
        ))
            
        return amplitude * source_time * source_spatial
    
    def step(self) -> None:
        """
        Perform a single time step in the 3D simulation.
        """
        # Compute Laplacians
        lap_phi = self.laplacian(self.phi_current)
        lap_Ax = self.laplacian(self.Ax_current)
        lap_Ay = self.laplacian(self.Ay_current)
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
        Az_new = 2 * self.Az_current - self.Az_old + (self.c**2 * self.dt**2) * lap_Az

        # Apply boundary conditions
        phi_new = self.apply_boundary_conditions(phi_new, "phi")
        Ax_new = self.apply_boundary_conditions(Ax_new, "Ax")
        Ay_new = self.apply_boundary_conditions(Ay_new, "Ay")
        Az_new = self.apply_boundary_conditions(Az_new, "Az")

        # Update fields
        self.phi_old = self.phi_current.copy()
        self.Ax_old = self.Ax_current.copy()
        self.Ay_old = self.Ay_current.copy()
        self.Az_old = self.Az_current.copy()

        self.phi_current = phi_new.copy()
        self.Ax_current = Ax_new.copy()
        self.Ay_current = Ay_new.copy()
        self.Az_current = Az_new.copy()

        self.current_time += self.dt
    
    def _create_snapshot(self, step: int) -> Dict[str, Any]:
        """
        Create a snapshot dictionary for the current state in 3D.
        
        Parameters
        ----------
        step : int
            Current time step
            
        Returns
        -------
        Dict[str, Any]
            Snapshot dictionary
        """
        return {
            'step': step,
            'time': self.current_time,
            'phi': self.phi_current.copy(),
            'Ax': self.Ax_current.copy(),
            'Ay': self.Ay_current.copy(),
            'Az': self.Az_current.copy()
        }


def create_solver(config: Dict[str, Any], 
                 grid: Any, 
                 phi: npt.NDArray[np.float64], 
                 A: Tuple[npt.NDArray[np.float64], ...]) -> BaseSolver:
    """
    Factory function to create the appropriate solver based on dimension.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary containing simulation parameters
    grid : Any
        Grid object with mesh information
    phi : npt.NDArray[np.float64]
        Initial scalar potential field
    A : Tuple[npt.NDArray[np.float64], ...]
        Initial vector potential field components
        
    Returns
    -------
    BaseSolver
        Instance of Solver2D or Solver3D
    """
    dimension = config.get('simulation', {}).get('dimension', 3)
    if dimension == 2:
        return Solver2D(config, grid, phi, A)
    else:
        return Solver3D(config, grid, phi, A)


# For backward compatibility
class Solver:
    """
    Legacy Solver class for backward compatibility.
    Delegates to the appropriate solver implementation.
    """
    
    def __init__(self, config: Dict[str, Any], 
                grid: Any, 
                phi: npt.NDArray[np.float64], 
                A: Tuple[npt.NDArray[np.float64], ...]) -> None:
        """
        Initialize the legacy solver.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary containing simulation parameters
        grid : Any
            Grid object with mesh information
        phi : npt.NDArray[np.float64]
            Initial scalar potential field
        A : Tuple[npt.NDArray[np.float64], ...]
            Initial vector potential field components
        """
        self._solver = create_solver(config, grid, phi, A)
        self.dimension = config.get('simulation', {}).get('dimension', 3)
        
        # Forward common attributes for backward compatibility
        self.phi_current = self._solver.phi_current
        self.phi_old = self._solver.phi_old
        self.dt = self._solver.dt
        self.c = self._solver.c
        self.num_steps = self._solver.num_steps
        self.current_time = self._solver.current_time
        self.snapshots = self._solver.snapshots
        
        # Forward dimension-specific attributes
        if self.dimension == 2:
            self.Ax_current = self._solver.Ax_current
            self.Ay_current = self._solver.Ay_current
            self.Ax_old = self._solver.Ax_old
            self.Ay_old = self._solver.Ay_old
            self.Az_current = None
            self.Az_old = None
            self.dx, self.dy = grid.get_spacing()
            self.dz = None
        else:
            self.Ax_current = self._solver.Ax_current
            self.Ay_current = self._solver.Ay_current
            self.Az_current = self._solver.Az_current
            self.Ax_old = self._solver.Ax_old
            self.Ay_old = self._solver.Ay_old
            self.Az_old = self._solver.Az_old
            self.dx, self.dy, self.dz = grid.get_spacing()
    
    def laplacian(self, field: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Forward to appropriate implementation"""
        return self._solver.laplacian(field)
    
    def apply_boundary_conditions(self, field: npt.NDArray[np.float64], 
                               field_name: str = "phi") -> npt.NDArray[np.float64]:
        """Forward to appropriate implementation"""
        return self._solver.apply_boundary_conditions(field, field_name)
    
    def compute_source_term(self, t: float) -> Union[float, npt.NDArray[np.float64]]:
        """Forward to appropriate implementation"""
        return self._solver.compute_source_term(t)
    
    def step(self) -> None:
        """Forward to appropriate implementation"""
        self._solver.step()
        
        # Update properties for backward compatibility
        self.phi_current = self._solver.phi_current
        self.phi_old = self._solver.phi_old
        self.current_time = self._solver.current_time
        
        if self.dimension == 2:
            self.Ax_current = self._solver.Ax_current
            self.Ay_current = self._solver.Ay_current
            self.Ax_old = self._solver.Ax_old
            self.Ay_old = self._solver.Ay_old
        else:
            self.Ax_current = self._solver.Ax_current
            self.Ay_current = self._solver.Ay_current
            self.Az_current = self._solver.Az_current
            self.Ax_old = self._solver.Ax_old
            self.Ay_old = self._solver.Ay_old
            self.Az_old = self._solver.Az_old
    
    def run(self) -> List[Dict[str, Any]]:
        """Forward to appropriate implementation"""
        snapshots = self._solver.run()
        
        # Update attributes after run
        self.phi_current = self._solver.phi_current
        self.phi_old = self._solver.phi_old
        self.current_time = self._solver.current_time
        self.snapshots = self._solver.snapshots
        
        if self.dimension == 2:
            self.Ax_current = self._solver.Ax_current
            self.Ay_current = self._solver.Ay_current
            self.Ax_old = self._solver.Ax_old
            self.Ay_old = self._solver.Ay_old
        else:
            self.Ax_current = self._solver.Ax_current
            self.Ay_current = self._solver.Ay_current
            self.Az_current = self._solver.Az_current
            self.Ax_old = self._solver.Ax_old
            self.Ay_old = self._solver.Ay_old
            self.Az_old = self._solver.Az_old
            
        return snapshots
