# modules/geometry.py

import numpy as np

class Geometry:
    def __init__(self, config):
        """
        Initialize the geometry using grid parameters from the configuration.
        
        Parameters:
            config (dict): Configuration dictionary containing simulation settings.
                           Expected keys:
                           - 'simulation' -> 'grid': dict with 'x_points', 'y_points',
                             'x_length', and 'y_length'.
        """
        grid_config = config.get('simulation', {}).get('grid', {})
        self.x_points = grid_config.get('x_points', 100)
        self.y_points = grid_config.get('y_points', 100)
        self.x_length = grid_config.get('x_length', 1.0)
        self.y_length = grid_config.get('y_length', 1.0)
        
        # Create coordinate arrays for the domain.
        self.x = np.linspace(0, self.x_length, self.x_points)
        self.y = np.linspace(0, self.y_length, self.y_points)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
    
    def get_meshgrid(self):
        """
        Returns:
            tuple: Meshgrid arrays (X, Y) corresponding to the simulation domain.
        """
        return self.X, self.Y

    def create_parallel_plate_mask(self, plate_thickness=None):
        """
        Generates a boolean mask for a resonant structure based on parallel plates.
        In this default configuration, two horizontal plates are defined:
          - Lower plate: from y = 0 to y = plate_thickness
          - Upper plate: from y = y_length - plate_thickness to y = y_length
          
        Parameters:
            plate_thickness (float, optional): Thickness of the plates. If not provided,
                                                 defaults to 5% of y_length.
        
        Returns:
            numpy.ndarray: A 2D boolean array of shape (x_points, y_points) where True
                           indicates the location of a plate.
        """
        if plate_thickness is None:
            plate_thickness = 0.05 * self.y_length
        
        mask = np.zeros((self.x_points, self.y_points), dtype=bool)
        # Mark the lower plate region.
        mask[:, self.y <= plate_thickness] = True
        # Mark the upper plate region.
        mask[:, self.y >= self.y_length - plate_thickness] = True
        
        return mask

    def create_cavity_mask(self, wall_thickness=None):
        """
        Generates a boolean mask for a rectangular cavity. The cavity is defined by having
        walls along the boundaries of the simulation domain.
        
        Parameters:
            wall_thickness (float, optional): Thickness of the cavity walls. If not provided,
                                                defaults to 5% of the smaller domain dimension.
        
        Returns:
            numpy.ndarray: A 2D boolean array of shape (x_points, y_points) where True
                           indicates the location of the cavity walls.
        """
        if wall_thickness is None:
            wall_thickness = 0.05 * min(self.x_length, self.y_length)
        
        mask = np.zeros((self.x_points, self.y_points), dtype=bool)
        # Left wall.
        mask[self.x <= wall_thickness, :] = True
        # Right wall.
        mask[self.x >= self.x_length - wall_thickness, :] = True
        # Bottom wall.
        mask[:, self.y <= wall_thickness] = True
        # Top wall.
        mask[:, self.y >= self.y_length - wall_thickness] = True
        
        return mask
