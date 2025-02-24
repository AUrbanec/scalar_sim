# modules/geometry.py

import numpy as np

class Geometry:
    def __init__(self, config):
        """
        Initialize the geometry for the scalar wave simulation
        
        Parameters
        ----------
        config : dict
            Configuration dictionary containing simulation parameters
        """
        self.dimension = config.get('simulation', {}).get('dimension', 3)
        grid_config = config.get('simulation', {}).get('grid', {})
        self.x_points = grid_config.get('x_points', 100)
        self.y_points = grid_config.get('y_points', 100)
        self.x_length = grid_config.get('x_length', 1.0)
        self.y_length = grid_config.get('y_length', 1.0)
        
        self.x = np.linspace(0, self.x_length, self.x_points)
        self.y = np.linspace(0, self.y_length, self.y_points)
        
        if self.dimension == 3:
            self.z_points = grid_config.get('z_points', 100)
            self.z_length = grid_config.get('z_length', 1.0)
            self.z = np.linspace(0, self.z_length, self.z_points)
            self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        else:
            self.z_points = None
            self.z_length = None
            self.z = None
            self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
            self.Z = None
            
        # Read cavity configuration
        self.cavity_config = config.get('simulation', {}).get('cavity', {})
        self.cavity_type = self.cavity_config.get('type', 'box')
        self.cavity_params = self.cavity_config.get('parameters', {})
        self.wall_thickness = self.cavity_config.get('wall_thickness', 0.05)
    
    def get_meshgrid(self):
        """
        Return the meshgrid coordinates
        
        Returns
        -------
        tuple
            Meshgrid arrays (X, Y) for 2D or (X, Y, Z) for 3D
        """
        if self.dimension == 2:
            return self.X, self.Y
        else:
            return self.X, self.Y, self.Z

    def create_parallel_plate_mask(self, plate_thickness=None):
        """
        Create a boolean mask for parallel plate geometry
        
        Parameters
        ----------
        plate_thickness : float, optional
            Thickness of the plates
            
        Returns
        -------
        ndarray
            Boolean mask with True at plate locations
        """
        if plate_thickness is None:
            plate_thickness = 0.05 * self.y_length
            
        if self.dimension == 2:
            mask = np.zeros((self.x_points, self.y_points), dtype=bool)
            mask[:, self.y <= plate_thickness] = True
            mask[:, self.y >= self.y_length - plate_thickness] = True
        else:
            mask = np.zeros((self.x_points, self.y_points, self.z_points), dtype=bool)
            mask[:, self.y <= plate_thickness, :] = True
            mask[:, self.y >= self.y_length - plate_thickness, :] = True
            
        return mask

    def create_cavity_mask(self, wall_thickness=None):
        """
        Create a boolean mask for box cavity geometry
        
        Parameters
        ----------
        wall_thickness : float, optional
            Thickness of the cavity walls
            
        Returns
        -------
        ndarray
            Boolean mask with True at wall locations
        """
        if wall_thickness is None:
            wall_thickness = self.wall_thickness
            
        if self.dimension == 2:
            mask = np.zeros((self.x_points, self.y_points), dtype=bool)
            mask[self.x <= wall_thickness, :] = True
            mask[self.x >= self.x_length - wall_thickness, :] = True
            mask[:, self.y <= wall_thickness] = True
            mask[:, self.y >= self.y_length - wall_thickness] = True
        else:
            mask = np.zeros((self.x_points, self.y_points, self.z_points), dtype=bool)
            mask[self.x <= wall_thickness, :, :] = True
            mask[self.x >= self.x_length - wall_thickness, :, :] = True
            mask[:, self.y <= wall_thickness, :] = True
            mask[:, self.y >= self.y_length - wall_thickness, :] = True
            mask[:, :, self.z <= wall_thickness] = True
            mask[:, :, self.z >= self.z_length - wall_thickness] = True
            
        return mask
        
    def create_sphere_mask(self, center=None, radius=None, wall_thickness=None):
        """
        Create a boolean mask for a spherical cavity
        
        Parameters
        ----------
        center : list, optional
            Center coordinates [x, y, z]
        radius : float, optional
            Radius of the sphere
        wall_thickness : float, optional
            Thickness of the cavity walls
            
        Returns
        -------
        ndarray
            Boolean mask with True at wall locations
        """
        if self.dimension == 2:
            raise ValueError("Sphere mask is only available in 3D simulations")
            
        if center is None:
            center = [self.x_length/2, self.y_length/2, self.z_length/2]
            
        if radius is None:
            radius = self.cavity_params.get('radius', min(self.x_length, self.y_length, self.z_length) * 0.4)
            
        if wall_thickness is None:
            wall_thickness = self.wall_thickness
            
        r_inner = radius - wall_thickness
        r_outer = radius
        
        mask = np.zeros((self.x_points, self.y_points, self.z_points), dtype=bool)
        
        # Calculate distance from center for all points
        r_squared = (
            (self.X - center[0])**2 + 
            (self.Y - center[1])**2 + 
            (self.Z - center[2])**2
        )
        
        # Points with r_inner < distance <= r_outer are part of the wall
        mask[(r_squared > r_inner**2) & (r_squared <= r_outer**2)] = True
                        
        return mask
        
    def create_cylinder_mask(self, center=None, radius=None, height=None, axis='z', wall_thickness=None):
        """
        Create a boolean mask for a cylindrical cavity
        
        Parameters
        ----------
        center : list, optional
            Center coordinates [x, y, z]
        radius : float, optional
            Radius of the cylinder
        height : float, optional
            Height of the cylinder
        axis : str, optional
            Cylinder axis orientation ('x', 'y', or 'z')
        wall_thickness : float, optional
            Thickness of the cavity walls
            
        Returns
        -------
        ndarray
            Boolean mask with True at wall locations
        """
        if self.dimension == 2:
            raise ValueError("Cylinder mask is only available in 3D simulations")
            
        if center is None:
            center = [self.x_length/2, self.y_length/2, self.z_length/2]
            
        if radius is None:
            radius = self.cavity_params.get('radius', min(self.x_length, self.y_length, self.z_length) * 0.3)
            
        if height is None:
            height = self.cavity_params.get('height', min(self.x_length, self.y_length, self.z_length) * 0.6)
            
        if wall_thickness is None:
            wall_thickness = self.wall_thickness
            
        r_inner = radius - wall_thickness
        r_outer = radius
        h_inner = height - wall_thickness
        h_outer = height
        
        mask = np.zeros((self.x_points, self.y_points, self.z_points), dtype=bool)
        
        if axis == 'x':
            # Distance from x-axis in yz-plane
            r_squared = (self.Y - center[1])**2 + (self.Z - center[2])**2
            # Distance along x-axis
            x_dist = np.abs(self.X - center[0])
            
            # Cylindrical wall
            cylindrical_wall = (r_squared > r_inner**2) & (r_squared <= r_outer**2) & (x_dist <= h_outer/2)
            
            # End caps
            end_caps = (r_squared <= r_outer**2) & (x_dist > h_inner/2) & (x_dist <= h_outer/2)
            
        elif axis == 'y':
            # Distance from y-axis in xz-plane
            r_squared = (self.X - center[0])**2 + (self.Z - center[2])**2
            # Distance along y-axis
            y_dist = np.abs(self.Y - center[1])
            
            # Cylindrical wall
            cylindrical_wall = (r_squared > r_inner**2) & (r_squared <= r_outer**2) & (y_dist <= h_outer/2)
            
            # End caps
            end_caps = (r_squared <= r_outer**2) & (y_dist > h_inner/2) & (y_dist <= h_outer/2)
            
        else:  # axis == 'z'
            # Distance from z-axis in xy-plane
            r_squared = (self.X - center[0])**2 + (self.Y - center[1])**2
            # Distance along z-axis
            z_dist = np.abs(self.Z - center[2])
            
            # Cylindrical wall
            cylindrical_wall = (r_squared > r_inner**2) & (r_squared <= r_outer**2) & (z_dist <= h_outer/2)
            
            # End caps
            end_caps = (r_squared <= r_outer**2) & (z_dist > h_inner/2) & (z_dist <= h_outer/2)
        
        # Combine walls and caps to form the complete cylinder
        mask[cylindrical_wall | end_caps] = True
        
        return mask
        
    def create_toroid_mask(self, center=None, major_radius=None, minor_radius=None, wall_thickness=None):
        """
        Create a boolean mask for a toroidal cavity
        
        Parameters
        ----------
        center : list, optional
            Center coordinates [x, y, z]
        major_radius : float, optional
            Major radius of the torus (distance from center to tube center)
        minor_radius : float, optional
            Minor radius of the torus (radius of the tube)
        wall_thickness : float, optional
            Thickness of the cavity walls
            
        Returns
        -------
        ndarray
            Boolean mask with True at wall locations
        """
        if self.dimension == 2:
            raise ValueError("Toroid mask is only available in 3D simulations")
            
        if center is None:
            center = [self.x_length/2, self.y_length/2, self.z_length/2]
            
        if major_radius is None:
            major_radius = self.cavity_params.get('major_radius', min(self.x_length, self.y_length) * 0.3)
            
        if minor_radius is None:
            minor_radius = self.cavity_params.get('minor_radius', min(self.x_length, self.y_length, self.z_length) * 0.1)
            
        if wall_thickness is None:
            wall_thickness = self.wall_thickness
            
        r_inner = minor_radius - wall_thickness
        r_outer = minor_radius
        
        mask = np.zeros((self.x_points, self.y_points, self.z_points), dtype=bool)
        
        # Calculate distance from xy-plane center
        dist_from_center_xy = np.sqrt((self.X - center[0])**2 + (self.Y - center[1])**2)
        
        # Calculate distance from the toroid's circular tube center
        dist_from_torus_center = np.sqrt((dist_from_center_xy - major_radius)**2 + (self.Z - center[2])**2)
        
        # Points where r_inner < distance <= r_outer are part of the wall
        mask[(dist_from_torus_center > r_inner) & (dist_from_torus_center <= r_outer)] = True
        
        return mask
        
    def create_resonant_cavity(self):
        """
        Create a boolean mask for the selected resonant cavity type
        
        Returns
        -------
        ndarray
            Boolean mask with True at cavity wall locations
        """
        if self.cavity_type == 'box':
            return self.create_cavity_mask(self.wall_thickness)
        elif self.cavity_type == 'sphere':
            return self.create_sphere_mask(wall_thickness=self.wall_thickness)
        elif self.cavity_type == 'cylinder':
            axis = self.cavity_params.get('axis', 'z')
            return self.create_cylinder_mask(axis=axis, wall_thickness=self.wall_thickness)
        elif self.cavity_type == 'toroid':
            return self.create_toroid_mask(wall_thickness=self.wall_thickness)
        else:
            # Default to box cavity if unknown type
            print(f"Unknown cavity type '{self.cavity_type}', using box cavity")
            return self.create_cavity_mask(self.wall_thickness)
