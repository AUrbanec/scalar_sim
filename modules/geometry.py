# modules/geometry.py

import numpy as np

class Geometry:
    def __init__(self, config):
        grid_config = config.get('simulation', {}).get('grid', {})
        self.x_points = grid_config.get('x_points', 100)
        self.y_points = grid_config.get('y_points', 100)
        self.x_length = grid_config.get('x_length', 1.0)
        self.y_length = grid_config.get('y_length', 1.0)
        
        self.x = np.linspace(0, self.x_length, self.x_points)
        self.y = np.linspace(0, self.y_length, self.y_points)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
    
    def get_meshgrid(self):
        return self.X, self.Y

    def create_parallel_plate_mask(self, plate_thickness=None):
        if plate_thickness is None:
            plate_thickness = 0.05 * self.y_length
        mask = np.zeros((self.x_points, self.y_points), dtype=bool)
        mask[:, self.y <= plate_thickness] = True
        mask[:, self.y >= self.y_length - plate_thickness] = True
        return mask

    def create_cavity_mask(self, wall_thickness=None):
        if wall_thickness is None:
            wall_thickness = 0.05 * min(self.x_length, self.y_length)
        mask = np.zeros((self.x_points, self.y_points), dtype=bool)
        mask[self.x <= wall_thickness, :] = True
        mask[self.x >= self.x_length - wall_thickness, :] = True
        mask[:, self.y <= wall_thickness] = True
        mask[:, self.y >= self.y_length - wall_thickness] = True
        return mask
