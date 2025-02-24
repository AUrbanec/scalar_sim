# modules/initialization.py

import numpy as np

class FieldInitialization:
    def __init__(self, config, grid):
        """
        Initialize field configurations for scalar wave simulation
        
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
        
        initial_config = config.get('simulation', {}).get('initial_conditions', {})
        self.phi_amplitude = initial_config.get('phi_amplitude', 0.01)
        self.A_amplitude = initial_config.get('A_amplitude', 0.0)
        self.profile = initial_config.get('profile', 'gaussian')
        
        if self.dimension == 2:
            self.X, self.Y = grid.get_meshgrid()
            self.center_x = grid.x_length / 2.0
            self.center_y = grid.y_length / 2.0
        else:  # 3D case
            self.X, self.Y, self.Z = grid.get_meshgrid()
            self.center_x = grid.x_length / 2.0
            self.center_y = grid.y_length / 2.0
            self.center_z = grid.z_length / 2.0

    def initialize_fields(self):
        """
        Initialize the scalar and vector potential fields
        
        Returns
        -------
        tuple
            Scalar potential (phi) and vector potential (A)
        """
        if self.dimension == 2:
            return self._initialize_fields_2d()
        else:
            return self._initialize_fields_3d()
            
    def _initialize_fields_2d(self):
        """
        Initialize fields for 2D simulation
        
        Returns
        -------
        tuple
            Scalar potential (phi) and vector potential (A)
        """
        sigma_x = self.grid.x_length * 0.1
        sigma_y = self.grid.y_length * 0.1
        
        # Initialize scalar potential
        phi = self.phi_amplitude * np.exp(-(((self.X - self.center_x)**2) / (2 * sigma_x**2) +
                                            ((self.Y - self.center_y)**2) / (2 * sigma_y**2)))
        
        # Initialize vector potential
        if self.A_amplitude != 0.0:
            A_x = self.A_amplitude * np.exp(-(((self.X - self.center_x)**2) / (2 * sigma_x**2) +
                                              ((self.Y - self.center_y)**2) / (2 * sigma_y**2)))
            A_y = self.A_amplitude * np.exp(-(((self.X - self.center_x)**2) / (2 * sigma_x**2) +
                                              ((self.Y - self.center_y)**2) / (2 * sigma_y**2)))
            A = (A_x, A_y)
        else:
            A_x = np.zeros_like(self.X)
            A_y = np.zeros_like(self.Y)
            A = (A_x, A_y)
            
        return phi, A
        
    def _initialize_fields_3d(self):
        """
        Initialize fields for 3D simulation
        
        Returns
        -------
        tuple
            Scalar potential (phi) and vector potential (A)
        """
        sigma_x = self.grid.x_length * 0.1
        sigma_y = self.grid.y_length * 0.1
        sigma_z = self.grid.z_length * 0.1
        
        # Initialize scalar potential based on profile
        if self.profile == 'gaussian':
            phi = self.phi_amplitude * np.exp(-(
                ((self.X - self.center_x)**2) / (2 * sigma_x**2) +
                ((self.Y - self.center_y)**2) / (2 * sigma_y**2) +
                ((self.Z - self.center_z)**2) / (2 * sigma_z**2)
            ))
        elif self.profile == 'spherical':
            r = np.sqrt(
                (self.X - self.center_x)**2 +
                (self.Y - self.center_y)**2 +
                (self.Z - self.center_z)**2
            )
            phi = self.phi_amplitude * np.exp(-(r**2) / (2 * sigma_x**2))
        else:  # custom or default
            phi = self.phi_amplitude * np.exp(-(
                ((self.X - self.center_x)**2) / (2 * sigma_x**2) +
                ((self.Y - self.center_y)**2) / (2 * sigma_y**2) +
                ((self.Z - self.center_z)**2) / (2 * sigma_z**2)
            ))
            
        # Initialize vector potential
        if self.A_amplitude != 0.0:
            if self.profile == 'gaussian' or self.profile == 'spherical':
                A_x = self.A_amplitude * np.exp(-(
                    ((self.X - self.center_x)**2) / (2 * sigma_x**2) +
                    ((self.Y - self.center_y)**2) / (2 * sigma_y**2) +
                    ((self.Z - self.center_z)**2) / (2 * sigma_z**2)
                ))
                A_y = self.A_amplitude * np.exp(-(
                    ((self.X - self.center_x)**2) / (2 * sigma_x**2) +
                    ((self.Y - self.center_y)**2) / (2 * sigma_y**2) +
                    ((self.Z - self.center_z)**2) / (2 * sigma_z**2)
                ))
                A_z = self.A_amplitude * np.exp(-(
                    ((self.X - self.center_x)**2) / (2 * sigma_x**2) +
                    ((self.Y - self.center_y)**2) / (2 * sigma_y**2) +
                    ((self.Z - self.center_z)**2) / (2 * sigma_z**2)
                ))
            else:
                A_x = self.A_amplitude * np.zeros_like(self.X)
                A_y = self.A_amplitude * np.zeros_like(self.Y)
                A_z = self.A_amplitude * np.zeros_like(self.Z)
        else:
            A_x = np.zeros_like(self.X)
            A_y = np.zeros_like(self.Y)
            A_z = np.zeros_like(self.Z)
            
        A = (A_x, A_y, A_z)
        return phi, A
