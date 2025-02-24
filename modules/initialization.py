# modules/initialization.py

import numpy as np

class FieldInitialization:
    def __init__(self, config, grid):
        self.config = config
        self.grid = grid
        
        initial_config = config.get('simulation', {}).get('initial_conditions', {})
        self.phi_amplitude = initial_config.get('phi_amplitude', 0.01)
        self.A_amplitude = initial_config.get('A_amplitude', 0.0)
        
        self.X, self.Y = grid.get_meshgrid()
        self.center_x = grid.x_length / 2.0
        self.center_y = grid.y_length / 2.0

    def initialize_fields(self):
        sigma_x = self.grid.x_length * 0.1
        sigma_y = self.grid.y_length * 0.1
        phi = self.phi_amplitude * np.exp(-(((self.X - self.center_x)**2) / (2 * sigma_x**2) +
                                            ((self.Y - self.center_y)**2) / (2 * sigma_y**2)))
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
