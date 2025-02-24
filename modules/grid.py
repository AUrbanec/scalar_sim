# modules/grid.py

import numpy as np
from .geometry import Geometry

class Grid:
    def __init__(self, config):
        """
        Initialize the simulation grid using parameters from the configuration.
        This includes creating the meshgrid using the Geometry module and computing
        the uniform grid spacing in the x and y directions.

        Parameters:
            config (dict): Configuration dictionary containing simulation settings.
        """
        # Instantiate the Geometry class to create the spatial domain.
        self.geometry = Geometry(config)
        self.X, self.Y = self.geometry.get_meshgrid()
        
        # Retrieve grid dimensions and physical domain lengths.
        self.x_points = self.geometry.x_points
        self.y_points = self.geometry.y_points
        self.x_length = self.geometry.x_length
        self.y_length = self.geometry.y_length

        # Calculate grid spacings assuming uniform distribution.
        self.dx = self.x_length / (self.x_points - 1)
        self.dy = self.y_length / (self.y_points - 1)

    def get_spacing(self):
        """
        Returns the grid spacing in the x and y directions.

        Returns:
            tuple: (dx, dy) representing the grid spacing.
        """
        return self.dx, self.dy

    def get_meshgrid(self):
        """
        Returns the meshgrid arrays for the simulation domain.

        Returns:
            tuple: (X, Y) meshgrid arrays.
        """
        return self.X, self.Y
