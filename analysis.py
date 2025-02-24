# analysis.py

import numpy as np
import numba as nb
from typing import Tuple, List, Dict, Any, Optional, Union
import numpy.typing as npt

@nb.njit
def compute_gradient_2d_optimized(field: npt.NDArray[np.float64], 
                               dx: float, 
                               dy: float) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Optimized computation of the gradient of a 2D field
    
    Parameters
    ----------
    field : npt.NDArray[np.float64]
        2D field array
    dx : float
        Grid spacing in x direction
    dy : float
        Grid spacing in y direction
        
    Returns
    -------
    Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        (gradient_x, gradient_y)
    """
    nx, ny = field.shape
    grad_x = np.zeros_like(field)
    grad_y = np.zeros_like(field)
    
    # Inside region
    for i in range(1, nx-1):
        for j in range(ny):
            grad_x[i, j] = (field[i+1, j] - field[i-1, j]) / (2*dx)
    
    for i in range(nx):
        for j in range(1, ny-1):
            grad_y[i, j] = (field[i, j+1] - field[i, j-1]) / (2*dy)
    
    return grad_x, grad_y

@nb.njit
def compute_gradient_3d_optimized(field: npt.NDArray[np.float64], 
                               dx: float, 
                               dy: float, 
                               dz: float) -> Tuple[npt.NDArray[np.float64], 
                                                npt.NDArray[np.float64], 
                                                npt.NDArray[np.float64]]:
    """
    Optimized computation of the gradient of a 3D field
    
    Parameters
    ----------
    field : npt.NDArray[np.float64]
        3D field array
    dx : float
        Grid spacing in x direction
    dy : float
        Grid spacing in y direction
    dz : float
        Grid spacing in z direction
        
    Returns
    -------
    Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]
        (gradient_x, gradient_y, gradient_z)
    """
    nx, ny, nz = field.shape
    grad_x = np.zeros_like(field)
    grad_y = np.zeros_like(field)
    grad_z = np.zeros_like(field)
    
    # X gradient (with loop fusion for better locality)
    for i in range(1, nx-1):
        for j in range(ny):
            for k in range(nz):
                grad_x[i, j, k] = (field[i+1, j, k] - field[i-1, j, k]) / (2*dx)
    
    # Y gradient
    for i in range(nx):
        for j in range(1, ny-1):
            for k in range(nz):
                grad_y[i, j, k] = (field[i, j+1, k] - field[i, j-1, k]) / (2*dy)
    
    # Z gradient
    for i in range(nx):
        for j in range(ny):
            for k in range(1, nz-1):
                grad_z[i, j, k] = (field[i, j, k+1] - field[i, j, k-1]) / (2*dz)
    
    return grad_x, grad_y, grad_z

def compute_gradient_2d(field: npt.NDArray[np.float64], 
                     dx: float, 
                     dy: float) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute the gradient of a 2D field
    
    Parameters
    ----------
    field : npt.NDArray[np.float64]
        2D field array
    dx : float
        Grid spacing in x direction
    dy : float
        Grid spacing in y direction
        
    Returns
    -------
    Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        (gradient_x, gradient_y)
    """
    return compute_gradient_2d_optimized(field, dx, dy)

def compute_gradient_3d(field: npt.NDArray[np.float64], 
                     dx: float, 
                     dy: float, 
                     dz: float) -> Tuple[npt.NDArray[np.float64], 
                                       npt.NDArray[np.float64], 
                                       npt.NDArray[np.float64]]:
    """
    Compute the gradient of a 3D field
    
    Parameters
    ----------
    field : npt.NDArray[np.float64]
        3D field array
    dx : float
        Grid spacing in x direction
    dy : float
        Grid spacing in y direction
    dz : float
        Grid spacing in z direction
        
    Returns
    -------
    Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]
        (gradient_x, gradient_y, gradient_z)
    """
    return compute_gradient_3d_optimized(field, dx, dy, dz)

def compute_gradient(field: npt.NDArray[np.float64], 
                  dx: float, 
                  dy: float, 
                  dz: Optional[float] = None) -> Union[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
                                                    Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    """
    Compute the gradient of a field (2D or 3D)
    
    Parameters
    ----------
    field : npt.NDArray[np.float64]
        Field array
    dx : float
        Grid spacing in x direction
    dy : float
        Grid spacing in y direction
    dz : Optional[float], default=None
        Grid spacing in z direction (for 3D)
        
    Returns
    -------
    Union[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]], 
          Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]]
        (gradient_x, gradient_y) for 2D or (gradient_x, gradient_y, gradient_z) for 3D
    """
    if dz is None or len(field.shape) == 2:
        return compute_gradient_2d(field, dx, dy)
    else:
        return compute_gradient_3d(field, dx, dy, dz)

@nb.njit
def compute_scalar_energy_2d_optimized(phi: npt.NDArray[np.float64], 
                                    phi_old: npt.NDArray[np.float64], 
                                    dt: float, 
                                    c: float, 
                                    dx: float, 
                                    dy: float) -> npt.NDArray[np.float64]:
    """
    Optimized computation of the energy density of the scalar field in 2D
    
    Parameters
    ----------
    phi : npt.NDArray[np.float64]
        Current scalar potential field
    phi_old : npt.NDArray[np.float64]
        Previous scalar potential field
    dt : float
        Time step
    c : float
        Wave speed
    dx : float
        Grid spacing in x direction
    dy : float
        Grid spacing in y direction
        
    Returns
    -------
    npt.NDArray[np.float64]
        Scalar energy density field
    """
    nx, ny = phi.shape
    dphi_dt = np.zeros_like(phi)
    grad_x = np.zeros_like(phi)
    grad_y = np.zeros_like(phi)
    energy = np.zeros_like(phi)
    
    # Time derivative
    for i in range(nx):
        for j in range(ny):
            dphi_dt[i, j] = (phi[i, j] - phi_old[i, j]) / dt
    
    # Spatial gradients
    for i in range(1, nx-1):
        for j in range(ny):
            grad_x[i, j] = (phi[i+1, j] - phi[i-1, j]) / (2*dx)
    
    for i in range(nx):
        for j in range(1, ny-1):
            grad_y[i, j] = (phi[i, j+1] - phi[i, j-1]) / (2*dy)
    
    # Energy calculation
    for i in range(nx):
        for j in range(ny):
            grad_phi_sq = grad_x[i, j]**2 + grad_y[i, j]**2
            energy[i, j] = 0.5 * (dphi_dt[i, j]**2) + 0.5 * (c**2) * grad_phi_sq
    
    return energy

@nb.njit
def compute_scalar_energy_3d_optimized(phi: npt.NDArray[np.float64], 
                                    phi_old: npt.NDArray[np.float64], 
                                    dt: float, 
                                    c: float, 
                                    dx: float, 
                                    dy: float, 
                                    dz: float) -> npt.NDArray[np.float64]:
    """
    Optimized computation of the energy density of the scalar field in 3D
    
    Parameters
    ----------
    phi : npt.NDArray[np.float64]
        Current scalar potential field
    phi_old : npt.NDArray[np.float64]
        Previous scalar potential field
    dt : float
        Time step
    c : float
        Wave speed
    dx : float
        Grid spacing in x direction
    dy : float
        Grid spacing in y direction
    dz : float
        Grid spacing in z direction
        
    Returns
    -------
    npt.NDArray[np.float64]
        Scalar energy density field
    """
    nx, ny, nz = phi.shape
    dphi_dt = np.zeros_like(phi)
    grad_x = np.zeros_like(phi)
    grad_y = np.zeros_like(phi)
    grad_z = np.zeros_like(phi)
    energy = np.zeros_like(phi)
    
    # Time derivative
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                dphi_dt[i, j, k] = (phi[i, j, k] - phi_old[i, j, k]) / dt
    
    # Spatial gradients
    for i in range(1, nx-1):
        for j in range(ny):
            for k in range(nz):
                grad_x[i, j, k] = (phi[i+1, j, k] - phi[i-1, j, k]) / (2*dx)
    
    for i in range(nx):
        for j in range(1, ny-1):
            for k in range(nz):
                grad_y[i, j, k] = (phi[i, j+1, k] - phi[i, j-1, k]) / (2*dy)
    
    for i in range(nx):
        for j in range(ny):
            for k in range(1, nz-1):
                grad_z[i, j, k] = (phi[i, j, k+1] - phi[i, j, k-1]) / (2*dz)
    
    # Energy calculation
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                grad_phi_sq = grad_x[i, j, k]**2 + grad_y[i, j, k]**2 + grad_z[i, j, k]**2
                energy[i, j, k] = 0.5 * (dphi_dt[i, j, k]**2) + 0.5 * (c**2) * grad_phi_sq
    
    return energy

def compute_scalar_energy_2d(phi: npt.NDArray[np.float64], 
                          phi_old: npt.NDArray[np.float64], 
                          dt: float, 
                          c: float, 
                          dx: float, 
                          dy: float) -> npt.NDArray[np.float64]:
    """
    Compute the energy density of the scalar field in 2D
    
    Parameters
    ----------
    phi : npt.NDArray[np.float64]
        Current scalar potential field
    phi_old : npt.NDArray[np.float64]
        Previous scalar potential field
    dt : float
        Time step
    c : float
        Wave speed
    dx : float
        Grid spacing in x direction
    dy : float
        Grid spacing in y direction
        
    Returns
    -------
    npt.NDArray[np.float64]
        Scalar energy density field
    """
    return compute_scalar_energy_2d_optimized(phi, phi_old, dt, c, dx, dy)

def compute_scalar_energy_3d(phi: npt.NDArray[np.float64], 
                          phi_old: npt.NDArray[np.float64], 
                          dt: float, 
                          c: float, 
                          dx: float, 
                          dy: float, 
                          dz: float) -> npt.NDArray[np.float64]:
    """
    Compute the energy density of the scalar field in 3D
    
    Parameters
    ----------
    phi : npt.NDArray[np.float64]
        Current scalar potential field
    phi_old : npt.NDArray[np.float64]
        Previous scalar potential field
    dt : float
        Time step
    c : float
        Wave speed
    dx : float
        Grid spacing in x direction
    dy : float
        Grid spacing in y direction
    dz : float
        Grid spacing in z direction
        
    Returns
    -------
    npt.NDArray[np.float64]
        Scalar energy density field
    """
    return compute_scalar_energy_3d_optimized(phi, phi_old, dt, c, dx, dy, dz)

def compute_scalar_energy(phi: npt.NDArray[np.float64], 
                       phi_old: npt.NDArray[np.float64], 
                       dt: float, 
                       c: float, 
                       dx: float, 
                       dy: float, 
                       dz: Optional[float] = None) -> npt.NDArray[np.float64]:
    """
    Compute the energy density of the scalar field (2D or 3D)
    
    Parameters
    ----------
    phi : npt.NDArray[np.float64]
        Current scalar potential field
    phi_old : npt.NDArray[np.float64]
        Previous scalar potential field
    dt : float
        Time step
    c : float
        Wave speed
    dx : float
        Grid spacing in x direction
    dy : float
        Grid spacing in y direction
    dz : Optional[float], default=None
        Grid spacing in z direction (for 3D)
        
    Returns
    -------
    npt.NDArray[np.float64]
        Scalar energy density field
    """
    if dz is None or len(phi.shape) == 2:
        return compute_scalar_energy_2d(phi, phi_old, dt, c, dx, dy)
    else:
        return compute_scalar_energy_3d(phi, phi_old, dt, c, dx, dy, dz)

@nb.njit
def compute_vector_energy_2d_optimized(Ax: npt.NDArray[np.float64], 
                                    Ay: npt.NDArray[np.float64], 
                                    dt: float, 
                                    c: float, 
                                    dx: float, 
                                    dy: float) -> npt.NDArray[np.float64]:
    """
    Optimized computation of the energy density of the vector field in 2D
    
    Parameters
    ----------
    Ax : npt.NDArray[np.float64]
        X component of vector potential
    Ay : npt.NDArray[np.float64]
        Y component of vector potential
    dt : float
        Time step
    c : float
        Wave speed
    dx : float
        Grid spacing in x direction
    dy : float
        Grid spacing in y direction
        
    Returns
    -------
    npt.NDArray[np.float64]
        Vector energy density field
    """
    nx, ny = Ax.shape
    grad_Ax_x = np.zeros_like(Ax)
    grad_Ax_y = np.zeros_like(Ax)
    grad_Ay_x = np.zeros_like(Ay)
    grad_Ay_y = np.zeros_like(Ay)
    energy = np.zeros_like(Ax)
    
    # Compute gradients of Ax
    for i in range(1, nx-1):
        for j in range(ny):
            grad_Ax_x[i, j] = (Ax[i+1, j] - Ax[i-1, j]) / (2*dx)
    
    for i in range(nx):
        for j in range(1, ny-1):
            grad_Ax_y[i, j] = (Ax[i, j+1] - Ax[i, j-1]) / (2*dy)
    
    # Compute gradients of Ay
    for i in range(1, nx-1):
        for j in range(ny):
            grad_Ay_x[i, j] = (Ay[i+1, j] - Ay[i-1, j]) / (2*dx)
    
    for i in range(nx):
        for j in range(1, ny-1):
            grad_Ay_y[i, j] = (Ay[i, j+1] - Ay[i, j-1]) / (2*dy)
    
    # Compute energy
    for i in range(nx):
        for j in range(ny):
            grad_A_sq = (grad_Ax_x[i, j]**2 + grad_Ax_y[i, j]**2 + 
                        grad_Ay_x[i, j]**2 + grad_Ay_y[i, j]**2)
            energy[i, j] = 0.5 * (c**2) * grad_A_sq
    
    return energy

@nb.njit
def compute_vector_energy_3d_optimized(Ax: npt.NDArray[np.float64], 
                                    Ay: npt.NDArray[np.float64], 
                                    Az: npt.NDArray[np.float64], 
                                    dt: float, 
                                    c: float, 
                                    dx: float, 
                                    dy: float, 
                                    dz: float) -> npt.NDArray[np.float64]:
    """
    Optimized computation of the energy density of the vector field in 3D
    
    Parameters
    ----------
    Ax : npt.NDArray[np.float64]
        X component of vector potential
    Ay : npt.NDArray[np.float64]
        Y component of vector potential
    Az : npt.NDArray[np.float64]
        Z component of vector potential
    dt : float
        Time step
    c : float
        Wave speed
    dx : float
        Grid spacing in x direction
    dy : float
        Grid spacing in y direction
    dz : float
        Grid spacing in z direction
        
    Returns
    -------
    npt.NDArray[np.float64]
        Vector energy density field
    """
    nx, ny, nz = Ax.shape
    grad_Ax_x = np.zeros_like(Ax)
    grad_Ax_y = np.zeros_like(Ax)
    grad_Ax_z = np.zeros_like(Ax)
    grad_Ay_x = np.zeros_like(Ay)
    grad_Ay_y = np.zeros_like(Ay)
    grad_Ay_z = np.zeros_like(Ay)
    grad_Az_x = np.zeros_like(Az)
    grad_Az_y = np.zeros_like(Az)
    grad_Az_z = np.zeros_like(Az)
    energy = np.zeros_like(Ax)
    
    # Compute gradients of Ax
    for i in range(1, nx-1):
        for j in range(ny):
            for k in range(nz):
                grad_Ax_x[i, j, k] = (Ax[i+1, j, k] - Ax[i-1, j, k]) / (2*dx)
    
    for i in range(nx):
        for j in range(1, ny-1):
            for k in range(nz):
                grad_Ax_y[i, j, k] = (Ax[i, j+1, k] - Ax[i, j-1, k]) / (2*dy)
    
    for i in range(nx):
        for j in range(ny):
            for k in range(1, nz-1):
                grad_Ax_z[i, j, k] = (Ax[i, j, k+1] - Ax[i, j, k-1]) / (2*dz)
    
    # Compute gradients of Ay
    for i in range(1, nx-1):
        for j in range(ny):
            for k in range(nz):
                grad_Ay_x[i, j, k] = (Ay[i+1, j, k] - Ay[i-1, j, k]) / (2*dx)
    
    for i in range(nx):
        for j in range(1, ny-1):
            for k in range(nz):
                grad_Ay_y[i, j, k] = (Ay[i, j+1, k] - Ay[i, j-1, k]) / (2*dy)
    
    for i in range(nx):
        for j in range(ny):
            for k in range(1, nz-1):
                grad_Ay_z[i, j, k] = (Ay[i, j, k+1] - Ay[i, j, k-1]) / (2*dz)
    
    # Compute gradients of Az
    for i in range(1, nx-1):
        for j in range(ny):
            for k in range(nz):
                grad_Az_x[i, j, k] = (Az[i+1, j, k] - Az[i-1, j, k]) / (2*dx)
    
    for i in range(nx):
        for j in range(1, ny-1):
            for k in range(nz):
                grad_Az_y[i, j, k] = (Az[i, j+1, k] - Az[i, j-1, k]) / (2*dy)
    
    for i in range(nx):
        for j in range(ny):
            for k in range(1, nz-1):
                grad_Az_z[i, j, k] = (Az[i, j, k+1] - Az[i, j, k-1]) / (2*dz)
    
    # Compute energy
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                grad_A_sq = (grad_Ax_x[i, j, k]**2 + grad_Ax_y[i, j, k]**2 + grad_Ax_z[i, j, k]**2 +
                            grad_Ay_x[i, j, k]**2 + grad_Ay_y[i, j, k]**2 + grad_Ay_z[i, j, k]**2 +
                            grad_Az_x[i, j, k]**2 + grad_Az_y[i, j, k]**2 + grad_Az_z[i, j, k]**2)
                energy[i, j, k] = 0.5 * (c**2) * grad_A_sq
    
    return energy

def compute_vector_energy_2d(Ax: npt.NDArray[np.float64], 
                          Ay: npt.NDArray[np.float64], 
                          dt: float, 
                          c: float, 
                          dx: float, 
                          dy: float) -> npt.NDArray[np.float64]:
    """
    Compute the energy density of the vector field in 2D
    
    Parameters
    ----------
    Ax : npt.NDArray[np.float64]
        X component of vector potential
    Ay : npt.NDArray[np.float64]
        Y component of vector potential
    dt : float
        Time step
    c : float
        Wave speed
    dx : float
        Grid spacing in x direction
    dy : float
        Grid spacing in y direction
        
    Returns
    -------
    npt.NDArray[np.float64]
        Vector energy density field
    """
    return compute_vector_energy_2d_optimized(Ax, Ay, dt, c, dx, dy)

def compute_vector_energy_3d(Ax: npt.NDArray[np.float64], 
                          Ay: npt.NDArray[np.float64], 
                          Az: npt.NDArray[np.float64], 
                          dt: float, 
                          c: float, 
                          dx: float, 
                          dy: float, 
                          dz: float) -> npt.NDArray[np.float64]:
    """
    Compute the energy density of the vector field in 3D
    
    Parameters
    ----------
    Ax : npt.NDArray[np.float64]
        X component of vector potential
    Ay : npt.NDArray[np.float64]
        Y component of vector potential
    Az : npt.NDArray[np.float64]
        Z component of vector potential
    dt : float
        Time step
    c : float
        Wave speed
    dx : float
        Grid spacing in x direction
    dy : float
        Grid spacing in y direction
    dz : float
        Grid spacing in z direction
        
    Returns
    -------
    npt.NDArray[np.float64]
        Vector energy density field
    """
    return compute_vector_energy_3d_optimized(Ax, Ay, Az, dt, c, dx, dy, dz)

def compute_vector_energy(A_components: Union[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]], 
                                           Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]], 
                       dt: float, 
                       c: float, 
                       dx: float, 
                       dy: float, 
                       dz: Optional[float] = None) -> npt.NDArray[np.float64]:
    """
    Compute the energy density of the vector field (2D or 3D)
    
    Parameters
    ----------
    A_components : Union[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]], 
                       Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]]
        Vector potential components (Ax, Ay) for 2D or (Ax, Ay, Az) for 3D
    dt : float
        Time step
    c : float
        Wave speed
    dx : float
        Grid spacing in x direction
    dy : float
        Grid spacing in y direction
    dz : Optional[float], default=None
        Grid spacing in z direction (for 3D)
        
    Returns
    -------
    npt.NDArray[np.float64]
        Vector energy density field
    """
    if dz is None or len(A_components) == 2:
        Ax, Ay = A_components
        return compute_vector_energy_2d(Ax, Ay, dt, c, dx, dy)
    else:
        Ax, Ay, Az = A_components
        return compute_vector_energy_3d(Ax, Ay, Az, dt, c, dx, dy, dz)

@nb.njit
def compute_curl_3d_optimized(Ax: npt.NDArray[np.float64], 
                           Ay: npt.NDArray[np.float64], 
                           Az: npt.NDArray[np.float64], 
                           dx: float, 
                           dy: float, 
                           dz: float) -> Tuple[npt.NDArray[np.float64], 
                                            npt.NDArray[np.float64], 
                                            npt.NDArray[np.float64]]:
    """
    Optimized computation of the curl of a 3D vector field
    
    Parameters
    ----------
    Ax : npt.NDArray[np.float64]
        X component of vector field
    Ay : npt.NDArray[np.float64]
        Y component of vector field
    Az : npt.NDArray[np.float64]
        Z component of vector field
    dx : float
        Grid spacing in x direction
    dy : float
        Grid spacing in y direction
    dz : float
        Grid spacing in z direction
        
    Returns
    -------
    Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]
        (curl_x, curl_y, curl_z)
    """
    nx, ny, nz = Ax.shape
    curl_x = np.zeros_like(Ax)
    curl_y = np.zeros_like(Ay)
    curl_z = np.zeros_like(Az)
    
    # dAz/dy - dAy/dz
    for i in range(nx):
        for j in range(1, ny-1):
            for k in range(1, nz-1):
                curl_x[i, j, k] = ((Az[i, j+1, k] - Az[i, j-1, k]) / (2*dy) - 
                                  (Ay[i, j, k+1] - Ay[i, j, k-1]) / (2*dz))
    
    # dAx/dz - dAz/dx
    for i in range(1, nx-1):
        for j in range(ny):
            for k in range(1, nz-1):
                curl_y[i, j, k] = ((Ax[i, j, k+1] - Ax[i, j, k-1]) / (2*dz) - 
                                  (Az[i+1, j, k] - Az[i-1, j, k]) / (2*dx))
    
    # dAy/dx - dAx/dy
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            for k in range(nz):
                curl_z[i, j, k] = ((Ay[i+1, j, k] - Ay[i-1, j, k]) / (2*dx) - 
                                  (Ax[i, j+1, k] - Ax[i, j-1, k]) / (2*dy))
    
    return curl_x, curl_y, curl_z

def compute_curl_3d(Ax: npt.NDArray[np.float64], 
                 Ay: npt.NDArray[np.float64], 
                 Az: npt.NDArray[np.float64], 
                 dx: float, 
                 dy: float, 
                 dz: float) -> Tuple[npt.NDArray[np.float64], 
                                   npt.NDArray[np.float64], 
                                   npt.NDArray[np.float64]]:
    """
    Compute the curl of a 3D vector field
    
    Parameters
    ----------
    Ax : npt.NDArray[np.float64]
        X component of vector field
    Ay : npt.NDArray[np.float64]
        Y component of vector field
    Az : npt.NDArray[np.float64]
        Z component of vector field
    dx : float
        Grid spacing in x direction
    dy : float
        Grid spacing in y direction
    dz : float
        Grid spacing in z direction
        
    Returns
    -------
    Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]
        (curl_x, curl_y, curl_z)
    """
    return compute_curl_3d_optimized(Ax, Ay, Az, dx, dy, dz)

def compute_total_energies(snapshots: List[Dict[str, Any]], 
                        dt: float, 
                        c: float, 
                        dx: float, 
                        dy: float, 
                        dz: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute total scalar and vector energies for all snapshots
    
    Parameters
    ----------
    snapshots : List[Dict[str, Any]]
        List of snapshot dictionaries
    dt : float
        Time step
    c : float
        Wave speed
    dx : float
        Grid spacing in x direction
    dy : float
        Grid spacing in y direction
    dz : Optional[float], default=None
        Grid spacing in z direction (for 3D)
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (times, scalar_energy_list, vector_energy_list)
    """
    times = []
    scalar_energy_list = []
    vector_energy_list = []
    
    # Check if we're in 2D or 3D mode
    is_3d = dz is not None
    
    for snapshot in snapshots:
        times.append(snapshot.get('time', 0.0))
        
        # Here, we approximate dphi/dt as zero because phi_old is not stored per snapshot.
        # In a refined analysis, consecutive snapshots should be used.
        
        # Compute scalar energy
        scalar_energy = compute_scalar_energy(
            snapshot['phi'], snapshot['phi'], dt, c, dx, dy, dz
        )
        
        # Compute vector energy
        if is_3d:
            vector_energy = compute_vector_energy(
                (snapshot['Ax'], snapshot['Ay'], snapshot['Az']), dt, c, dx, dy, dz
            )
        else:
            vector_energy = compute_vector_energy(
                (snapshot['Ax'], snapshot['Ay']), dt, c, dx, dy
            )
        
        # Add to lists and update snapshot
        scalar_energy_list.append(np.sum(scalar_energy))
        vector_energy_list.append(np.sum(vector_energy))
        snapshot['energy'] = scalar_energy + vector_energy
        
    return np.array(times), np.array(scalar_energy_list), np.array(vector_energy_list)
