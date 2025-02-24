# modules/grid.py

import numpy as np
from .geometry import Geometry

class Grid:
    def __init__(self, config):
        """
        Initialize the grid for the simulation
        
        Parameters
        ----------
        config : dict
            Configuration dictionary containing simulation parameters
        """
        self.dimension = config.get('simulation', {}).get('dimension', 3)
        self.geometry = Geometry(config)
        
        if self.dimension == 2:
            self.X, self.Y = self.geometry.get_meshgrid()
            self.x_points = self.geometry.x_points
            self.y_points = self.geometry.y_points
            self.x_length = self.geometry.x_length
            self.y_length = self.geometry.y_length
            self.dx = self.x_length / (self.x_points - 1)
            self.dy = self.y_length / (self.y_points - 1)
            self.dz = None
            self.z_points = None
            self.z_length = None
            self.Z = None
        else:  # 3D case
            self.X, self.Y, self.Z = self.geometry.get_meshgrid()
            self.x_points = self.geometry.x_points
            self.y_points = self.geometry.y_points
            self.z_points = self.geometry.z_points
            self.x_length = self.geometry.x_length
            self.y_length = self.geometry.y_length
            self.z_length = self.geometry.z_length
            self.dx = self.x_length / (self.x_points - 1)
            self.dy = self.y_length / (self.y_points - 1)
            self.dz = self.z_length / (self.z_points - 1)

    def get_spacing(self):
        """
        Get the grid spacing in each dimension
        
        Returns
        -------
        tuple
            Grid spacing (dx, dy) for 2D or (dx, dy, dz) for 3D
        """
        if self.dimension == 2:
            return self.dx, self.dy
        else:
            return self.dx, self.dy, self.dz

    def get_meshgrid(self):
        """
        Get the meshgrid coordinates
        
        Returns
        -------
        tuple
            Meshgrid arrays (X, Y) for 2D or (X, Y, Z) for 3D
        """
        if self.dimension == 2:
            return self.X, self.Y
        else:
            return self.X, self.Y, self.Z
            
    def get_slice(self, axis='z', position=0.5):
        """
        Extract a 2D slice from the 3D domain
        
        Parameters
        ----------
        axis : str
            Axis perpendicular to the slice ('x', 'y', or 'z')
        position : float
            Normalized position (0 to 1) along the axis
            
        Returns
        -------
        tuple
            2D slice coordinates (X, Y) or (X, Z) or (Y, Z)
        int
            Index corresponding to the position
        """
        if self.dimension == 2:
            return (self.X, self.Y), None
            
        if axis == 'x':
            idx = int(position * (self.x_points - 1))
            return (self.Y[:, idx, :], self.Z[:, idx, :]), idx
        elif axis == 'y':
            idx = int(position * (self.y_points - 1))
            return (self.X[idx, :, :], self.Z[idx, :, :]), idx
        else:  # axis == 'z'
            idx = int(position * (self.z_points - 1))
            return (self.X[:, :, idx], self.Y[:, :, idx]), idx
