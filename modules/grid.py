# modules/grid.py

import numpy as np
from .geometry import Geometry

class Grid:
    def __init__(self, config):
        self.geometry = Geometry(config)
        self.X, self.Y = self.geometry.get_meshgrid()
        self.x_points = self.geometry.x_points
        self.y_points = self.geometry.y_points
        self.x_length = self.geometry.x_length
        self.y_length = self.geometry.y_length
        self.dx = self.x_length / (self.x_points - 1)
        self.dy = self.y_length / (self.y_points - 1)

    def get_spacing(self):
        return self.dx, self.dy

    def get_meshgrid(self):
        return self.X, self.Y
