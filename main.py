# main.py

import os
import yaml
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy.typing as npt
from dataclasses import dataclass
from modules.grid import Grid
from modules.initialization import FieldInitialization
from modules.solver import Solver, create_solver
from modules.visualization import Visualization
from modules.data_output import DataOutput
from analysis import compute_total_energies

@dataclass
class GridConfig:
    x_points: int
    y_points: int
    z_points: Optional[int]
    x_length: float
    y_length: float
    z_length: Optional[float]

@dataclass
class TimeConfig:
    dt: float
    total_time: float

@dataclass
class MaterialConfig:
    epsilon: float
    mu: float

@dataclass
class BoundaryConfig:
    type: str
    impedance: float

@dataclass
class SourceConfig:
    enabled: bool
    amplitude: float
    frequency: float
    modulation: str
    center: List[float]
    sigma: float

@dataclass
class NonlinearityConfig:
    enabled: bool
    strength: float

@dataclass
class OutputConfig:
    output_dir: str
    snapshot_interval: int

@dataclass
class SimulationConfig:
    dimension: int
    grid: GridConfig
    time: TimeConfig
    material: MaterialConfig
    boundary_conditions: BoundaryConfig
    source: SourceConfig
    nonlinearity: NonlinearityConfig
    output: OutputConfig

class ConfigValidator:
    """Validates configuration and checks for CFL condition"""
    
    @staticmethod
    def validate_grid(config: Dict[str, Any]) -> GridConfig:
        """
        Validate grid configuration
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary
            
        Returns
        -------
        GridConfig
            Validated grid configuration
        """
        grid_config = config.get('simulation', {}).get('grid', {})
        dimension = config.get('simulation', {}).get('dimension', 3)
        
        # Basic validations
        x_points = grid_config.get('x_points', 100)
        if x_points < 3:
            raise ValueError("x_points must be at least 3 for finite difference calculations")
            
        y_points = grid_config.get('y_points', 100)
        if y_points < 3:
            raise ValueError("y_points must be at least 3 for finite difference calculations")
            
        if dimension == 3:
            z_points = grid_config.get('z_points', 100)
            if z_points < 3:
                raise ValueError("z_points must be at least 3 for finite difference calculations")
        else:
            z_points = None
            
        x_length = grid_config.get('x_length', 1.0)
        if x_length <= 0:
            raise ValueError("x_length must be positive")
            
        y_length = grid_config.get('y_length', 1.0)
        if y_length <= 0:
            raise ValueError("y_length must be positive")
            
        if dimension == 3:
            z_length = grid_config.get('z_length', 1.0)
            if z_length <= 0:
                raise ValueError("z_length must be positive")
        else:
            z_length = None
            
        return GridConfig(
            x_points=x_points,
            y_points=y_points,
            z_points=z_points,
            x_length=x_length,
            y_length=y_length,
            z_length=z_length
        )
    
    @staticmethod
    def validate_time(config: Dict[str, Any]) -> TimeConfig:
        """
        Validate time configuration
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary
            
        Returns
        -------
        TimeConfig
            Validated time configuration
        """
        time_config = config.get('simulation', {}).get('time', {})
        
        dt = time_config.get('dt', 0.001)
        if dt <= 0:
            raise ValueError("dt must be positive")
            
        total_time = time_config.get('total_time', 1.0)
        if total_time <= 0:
            raise ValueError("total_time must be positive")
            
        return TimeConfig(dt=dt, total_time=total_time)
    
    @staticmethod
    def validate_material(config: Dict[str, Any]) -> MaterialConfig:
        """
        Validate material configuration
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary
            
        Returns
        -------
        MaterialConfig
            Validated material configuration
        """
        material_config = config.get('simulation', {}).get('material', {})
        
        epsilon = material_config.get('epsilon', 1.0)
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
            
        mu = material_config.get('mu', 1.0)
        if mu <= 0:
            raise ValueError("mu must be positive")
            
        return MaterialConfig(epsilon=epsilon, mu=mu)
    
    @staticmethod
    def validate_boundary(config: Dict[str, Any]) -> BoundaryConfig:
        """
        Validate boundary configuration
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary
            
        Returns
        -------
        BoundaryConfig
            Validated boundary configuration
        """
        bc_config = config.get('simulation', {}).get('boundary_conditions', {})
        
        bc_type = bc_config.get('type', 'dirichlet').lower()
        if bc_type not in ['dirichlet', 'neumann', 'advanced']:
            raise ValueError("Boundary condition type must be one of: dirichlet, neumann, advanced")
            
        impedance = bc_config.get('impedance', 0.1)
        if impedance < 0:
            raise ValueError("impedance must be non-negative")
            
        return BoundaryConfig(type=bc_type, impedance=impedance)
    
    @staticmethod
    def validate_source(config: Dict[str, Any]) -> SourceConfig:
        """
        Validate source configuration
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary
            
        Returns
        -------
        SourceConfig
            Validated source configuration
        """
        source_config = config.get('simulation', {}).get('source', {})
        dimension = config.get('simulation', {}).get('dimension', 3)
        
        enabled = source_config.get('enabled', False)
        amplitude = source_config.get('amplitude', 1.0)
        frequency = source_config.get('frequency', 50.0)
        if frequency <= 0:
            raise ValueError("frequency must be positive")
            
        modulation = source_config.get('modulation', 'amplitude')
        if modulation not in ['amplitude', 'frequency', 'none']:
            raise ValueError("modulation must be one of: amplitude, frequency, none")
            
        center = source_config.get('center', [0.5, 0.5, 0.5] if dimension == 3 else [0.5, 0.5])
        if len(center) != dimension:
            raise ValueError(f"center should have {dimension} components for a {dimension}D simulation")
            
        sigma = source_config.get('sigma', 0.05)
        if sigma <= 0:
            raise ValueError("sigma must be positive")
            
        return SourceConfig(
            enabled=enabled,
            amplitude=amplitude,
            frequency=frequency,
            modulation=modulation,
            center=center,
            sigma=sigma
        )
    
    @staticmethod
    def validate_nonlinearity(config: Dict[str, Any]) -> NonlinearityConfig:
        """
        Validate nonlinearity configuration
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary
            
        Returns
        -------
        NonlinearityConfig
            Validated nonlinearity configuration
        """
        nonlin_config = config.get('simulation', {}).get('nonlinearity', {})
        
        enabled = nonlin_config.get('enabled', False)
        strength = nonlin_config.get('strength', 0.0)
        
        return NonlinearityConfig(enabled=enabled, strength=strength)
    
    @staticmethod
    def validate_output(config: Dict[str, Any]) -> OutputConfig:
        """
        Validate output configuration
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary
            
        Returns
        -------
        OutputConfig
            Validated output configuration
        """
        output_config = config.get('simulation', {}).get('output', {})
        
        output_dir = output_config.get('output_dir', 'outputs')
        snapshot_interval = output_config.get('snapshot_interval', 50)
        if snapshot_interval <= 0:
            raise ValueError("snapshot_interval must be positive")
            
        return OutputConfig(output_dir=output_dir, snapshot_interval=snapshot_interval)
    
    @staticmethod
    def check_cfl_condition(config: Dict[str, Any]) -> Tuple[bool, float, float]:
        """
        Check if CFL condition is satisfied for numerical stability
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary
            
        Returns
        -------
        Tuple[bool, float, float]
            Tuple containing (is_stable, actual_dt, max_stable_dt)
        """
        grid_config = config.get('simulation', {}).get('grid', {})
        time_config = config.get('simulation', {}).get('time', {})
        dimension = config.get('simulation', {}).get('dimension', 3)
        
        # Get material properties for wave speed
        material_config = config.get('simulation', {}).get('material', {})
        epsilon = material_config.get('epsilon', 1.0)
        mu = material_config.get('mu', 1.0)
        c = 1.0 / np.sqrt(epsilon * mu)
        
        # Calculate grid spacing
        dx = grid_config.get('x_length', 1.0) / (grid_config.get('x_points', 100) - 1)
        dy = grid_config.get('y_length', 1.0) / (grid_config.get('y_points', 100) - 1)
        
        if dimension == 3:
            dz = grid_config.get('z_length', 1.0) / (grid_config.get('z_points', 100) - 1)
            min_spacing = min(dx, dy, dz)
        else:
            min_spacing = min(dx, dy)
        
        # Calculate maximum stable time step (CFL condition)
        # For explicit FDTD, dt ≤ dx/(c*sqrt(D)) where D is dimension
        max_stable_dt = min_spacing / (c * np.sqrt(dimension))
        actual_dt = time_config.get('dt', 0.001)
        
        is_stable = actual_dt <= max_stable_dt
        
        return is_stable, actual_dt, max_stable_dt
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Validate the full configuration
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary
            
        Returns
        -------
        Tuple[Dict[str, Any], List[str]]
            Tuple containing (validated_config, warnings_list)
        """
        warnings = []
        
        # Validate core components
        try:
            dimension = config.get('simulation', {}).get('dimension', 3)
            if dimension not in [2, 3]:
                raise ValueError("dimension must be 2 or 3")
                
            grid_config = cls.validate_grid(config)
            time_config = cls.validate_time(config)
            material_config = cls.validate_material(config)
            boundary_config = cls.validate_boundary(config)
            source_config = cls.validate_source(config)
            nonlinearity_config = cls.validate_nonlinearity(config)
            output_config = cls.validate_output(config)
            
            # Check CFL condition
            is_stable, actual_dt, max_stable_dt = cls.check_cfl_condition(config)
            if not is_stable:
                warnings.append(
                    f"CFL condition not satisfied. Current dt={actual_dt} exceeds "
                    f"maximum stable dt={max_stable_dt:.6f}. Simulation may be unstable."
                )
                
            # Create a validated simulation config
            sim_config = SimulationConfig(
                dimension=dimension,
                grid=grid_config,
                time=time_config,
                material=material_config,
                boundary_conditions=boundary_config,
                source=source_config,
                nonlinearity=nonlinearity_config,
                output=output_config
            )
            
            # For now we just return the original config and warnings
            # In the future, we could convert the validated config back to a dict
            
        except ValueError as e:
            # Convert validation errors to warnings for now
            warnings.append(f"Configuration error: {str(e)}")
            
        return config, warnings

def load_config(config_file: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load and validate the configuration from a YAML file
    
    Parameters
    ----------
    config_file : str
        Path to the configuration file
        
    Returns
    -------
    Dict[str, Any]
        Validated configuration dictionary
    """
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            
        # Validate the loaded configuration
        config, warnings = ConfigValidator.validate_config(config)
        
        # Print any warnings
        for warning in warnings:
            print(f"WARNING: {warning}")
            
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Error loading configuration: {str(e)}")

def main() -> None:
    """Main function to run the simulation"""
    # Load configuration
    config = load_config()
    
    # Check simulation dimension
    dimension = config.get('simulation', {}).get('dimension', 3)
    print(f"Running {dimension}D simulation...")
    
    # Check CFL condition explicitly
    is_stable, actual_dt, max_stable_dt = ConfigValidator.check_cfl_condition(config)
    if not is_stable:
        print(f"WARNING: CFL condition not satisfied. Current dt={actual_dt} exceeds "
              f"maximum stable dt={max_stable_dt:.6f}.")
        user_continue = input("Simulation may be unstable. Continue anyway? (y/n): ")
        if user_continue.lower() != 'y':
            print("Simulation aborted.")
            return
    
    # Initialize grid, fields, and solver
    grid = Grid(config)
    initializer = FieldInitialization(config, grid)
    phi, A = initializer.initialize_fields()
    
    # Use the new solver factory function
    solver = create_solver(config, grid, phi, A)
    
    # Run simulation
    print("Starting simulation...")
    snapshots = solver.run()
    print(f"Simulation complete. Collected {len(snapshots)} snapshots.")

    # Compute energy metrics
    if dimension == 2:
        dx, dy = grid.get_spacing()
        times, scalar_energy, vector_energy = compute_total_energies(
            snapshots, solver.dt, solver.c, dx, dy
        )
    else:  # 3D case
        dx, dy, dz = grid.get_spacing()
        times, scalar_energy, vector_energy = compute_total_energies(
            snapshots, solver.dt, solver.c, dx, dy, dz
        )

    # Save data outputs
    data_out = DataOutput(config)
    data_out.save_all_snapshots(snapshots)
    
    # Create and save checkpoint
    if dimension == 2:
        state = {
            'phi': solver.phi_current,
            'Ax': solver.Ax_current,
            'Ay': solver.Ay_current,
            'step': solver.num_steps
        }
    else:  # 3D case
        state = {
            'phi': solver.phi_current,
            'Ax': solver.Ax_current,
            'Ay': solver.Ay_current,
            'Az': solver.Az_current,
            'step': solver.num_steps
        }
    data_out.save_checkpoint(state, checkpoint_name="final_checkpoint.npz")
    data_out.save_energy_data(times, scalar_energy, vector_energy)

    # Create visualizations
    viz = Visualization(config, grid)
    for snapshot in snapshots:
        viz.save_snapshot(snapshot)
    viz.animate_snapshots(snapshots, save_filename="simulation_animation.mp4")
    
    # Create and save energy time series plot
    fig_energy = viz.plot_energy_time_series(times, scalar_energy, vector_energy)
    energy_fig_path = os.path.join(config['simulation']['output']['output_dir'], "energy_time_series.png")
    fig_energy.savefig(energy_fig_path)
    print("Data output and visualization complete.")

if __name__ == "__main__":
    main()
