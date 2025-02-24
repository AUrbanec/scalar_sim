# modules/initialization.py

import numpy as np

class FieldInitialization:
    def __init__(self, config, grid):
        """
        Initialize the field potentials on the simulation grid.

        Parameters:
            config (dict): Configuration dictionary containing simulation settings.
            grid (Grid): An instance of the Grid class containing meshgrid and spacing information.
        """
        self.config = config
        self.grid = grid
        
        # Retrieve initial condition amplitudes from configuration.
        initial_config = config.get('simulation', {}).get('initial_conditions', {})
        self.phi_amplitude = initial_config.get('phi_amplitude', 0.01)
        self.A_amplitude = initial_config.get('A_amplitude', 0.0)
        
        # Retrieve the meshgrid arrays.
        self.X, self.Y = grid.get_meshgrid()
        # Determine the center of the domain for the perturbation.
        self.center_x = grid.x_length / 2.0
        self.center_y = grid.y_length / 2.0

    def initialize_fields(self):
        """
        Initializes the scalar and vector potentials.

        Returns:
            tuple: (phi, A) where phi is a 2D numpy array representing the scalar potential,
                   and A is a tuple of two 2D numpy arrays (A_x, A_y) representing the vector potential.
        """
        # Define Gaussian parameters for the localized perturbation.
        sigma_x = self.grid.x_length * 0.1
        sigma_y = self.grid.y_length * 0.1
        
        # Initialize the scalar potential phi with a Gaussian bump.
        phi = self.phi_amplitude * np.exp(-(((self.X - self.center_x)**2) / (2 * sigma_x**2) +
                                            ((self.Y - self.center_y)**2) / (2 * sigma_y**2)))
        
        # Initialize the vector potential A components.
        if self.A_amplitude != 0.0:
            A_x = self.A_amplitude * np.exp(-(((self.X - self.center_x)**2) / (2 * sigma_x**2) +
                                              ((self.Y - self.center_y)**2) / (2 * sigma_y**2)))
            A_y = self.A_amplitude * np.exp(-(((self.X - self.center_x)**2) / (2 * sigma_x**2) +
                                              ((self.Y - self.center_y)**2) / (2 * sigma_y**2)))
        else:
            A_x = np.zeros_like(self.X)
            A_y = np.zeros_like(self.Y)
        
        A = (A_x, A_y)
        return phi, A
