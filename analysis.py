# analysis.py

import numpy as np

def compute_gradient_2d(field, dx, dy):
    """
    Compute the gradient of a 2D field
    
    Parameters
    ----------
    field : ndarray
        2D field array
    dx : float
        Grid spacing in x direction
    dy : float
        Grid spacing in y direction
        
    Returns
    -------
    tuple
        (gradient_x, gradient_y)
    """
    grad_x = np.zeros_like(field)
    grad_y = np.zeros_like(field)
    grad_x[1:-1, :] = (field[2:, :] - field[:-2, :]) / (2*dx)
    grad_y[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2*dy)
    return grad_x, grad_y

def compute_gradient_3d(field, dx, dy, dz):
    """
    Compute the gradient of a 3D field
    
    Parameters
    ----------
    field : ndarray
        3D field array
    dx : float
        Grid spacing in x direction
    dy : float
        Grid spacing in y direction
    dz : float
        Grid spacing in z direction
        
    Returns
    -------
    tuple
        (gradient_x, gradient_y, gradient_z)
    """
    grad_x = np.zeros_like(field)
    grad_y = np.zeros_like(field)
    grad_z = np.zeros_like(field)
    
    grad_x[1:-1, :, :] = (field[2:, :, :] - field[:-2, :, :]) / (2*dx)
    grad_y[:, 1:-1, :] = (field[:, 2:, :] - field[:, :-2, :]) / (2*dy)
    grad_z[:, :, 1:-1] = (field[:, :, 2:] - field[:, :, :-2]) / (2*dz)
    
    return grad_x, grad_y, grad_z

def compute_gradient(field, dx, dy, dz=None):
    """
    Compute the gradient of a field (2D or 3D)
    
    Parameters
    ----------
    field : ndarray
        Field array
    dx : float
        Grid spacing in x direction
    dy : float
        Grid spacing in y direction
    dz : float, optional
        Grid spacing in z direction (for 3D)
        
    Returns
    -------
    tuple
        (gradient_x, gradient_y) for 2D or (gradient_x, gradient_y, gradient_z) for 3D
    """
    if dz is None or len(field.shape) == 2:
        return compute_gradient_2d(field, dx, dy)
    else:
        return compute_gradient_3d(field, dx, dy, dz)

def compute_scalar_energy_2d(phi, phi_old, dt, c, dx, dy):
    """
    Compute the energy density of the scalar field in 2D
    
    Parameters
    ----------
    phi : ndarray
        Current scalar potential field
    phi_old : ndarray
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
    ndarray
        Scalar energy density field
    """
    dphi_dt = (phi - phi_old) / dt
    grad_x, grad_y = compute_gradient_2d(phi, dx, dy)
    grad_phi_sq = grad_x**2 + grad_y**2
    energy = 0.5 * (dphi_dt**2) + 0.5 * (c**2) * grad_phi_sq
    return energy

def compute_scalar_energy_3d(phi, phi_old, dt, c, dx, dy, dz):
    """
    Compute the energy density of the scalar field in 3D
    
    Parameters
    ----------
    phi : ndarray
        Current scalar potential field
    phi_old : ndarray
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
    ndarray
        Scalar energy density field
    """
    dphi_dt = (phi - phi_old) / dt
    grad_x, grad_y, grad_z = compute_gradient_3d(phi, dx, dy, dz)
    grad_phi_sq = grad_x**2 + grad_y**2 + grad_z**2
    energy = 0.5 * (dphi_dt**2) + 0.5 * (c**2) * grad_phi_sq
    return energy

def compute_scalar_energy(phi, phi_old, dt, c, dx, dy, dz=None):
    """
    Compute the energy density of the scalar field (2D or 3D)
    
    Parameters
    ----------
    phi : ndarray
        Current scalar potential field
    phi_old : ndarray
        Previous scalar potential field
    dt : float
        Time step
    c : float
        Wave speed
    dx : float
        Grid spacing in x direction
    dy : float
        Grid spacing in y direction
    dz : float, optional
        Grid spacing in z direction (for 3D)
        
    Returns
    -------
    ndarray
        Scalar energy density field
    """
    if dz is None or len(phi.shape) == 2:
        return compute_scalar_energy_2d(phi, phi_old, dt, c, dx, dy)
    else:
        return compute_scalar_energy_3d(phi, phi_old, dt, c, dx, dy, dz)

def compute_vector_energy_2d(Ax, Ay, dt, c, dx, dy):
    """
    Compute the energy density of the vector field in 2D
    
    Parameters
    ----------
    Ax : ndarray
        X component of vector potential
    Ay : ndarray
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
    ndarray
        Vector energy density field
    """
    grad_Ax_x, grad_Ax_y = compute_gradient_2d(Ax, dx, dy)
    grad_Ay_x, grad_Ay_y = compute_gradient_2d(Ay, dx, dy)
    grad_A_sq = grad_Ax_x**2 + grad_Ax_y**2 + grad_Ay_x**2 + grad_Ay_y**2
    energy = 0.5 * (c**2) * grad_A_sq
    return energy

def compute_vector_energy_3d(Ax, Ay, Az, dt, c, dx, dy, dz):
    """
    Compute the energy density of the vector field in 3D
    
    Parameters
    ----------
    Ax : ndarray
        X component of vector potential
    Ay : ndarray
        Y component of vector potential
    Az : ndarray
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
    ndarray
        Vector energy density field
    """
    grad_Ax_x, grad_Ax_y, grad_Ax_z = compute_gradient_3d(Ax, dx, dy, dz)
    grad_Ay_x, grad_Ay_y, grad_Ay_z = compute_gradient_3d(Ay, dx, dy, dz)
    grad_Az_x, grad_Az_y, grad_Az_z = compute_gradient_3d(Az, dx, dy, dz)
    
    grad_A_sq = (grad_Ax_x**2 + grad_Ax_y**2 + grad_Ax_z**2 +
                 grad_Ay_x**2 + grad_Ay_y**2 + grad_Ay_z**2 +
                 grad_Az_x**2 + grad_Az_y**2 + grad_Az_z**2)
                 
    energy = 0.5 * (c**2) * grad_A_sq
    return energy

def compute_vector_energy(A_components, dt, c, dx, dy, dz=None):
    """
    Compute the energy density of the vector field (2D or 3D)
    
    Parameters
    ----------
    A_components : tuple
        Vector potential components (Ax, Ay) for 2D or (Ax, Ay, Az) for 3D
    dt : float
        Time step
    c : float
        Wave speed
    dx : float
        Grid spacing in x direction
    dy : float
        Grid spacing in y direction
    dz : float, optional
        Grid spacing in z direction (for 3D)
        
    Returns
    -------
    ndarray
        Vector energy density field
    """
    if dz is None or len(A_components) == 2:
        Ax, Ay = A_components
        return compute_vector_energy_2d(Ax, Ay, dt, c, dx, dy)
    else:
        Ax, Ay, Az = A_components
        return compute_vector_energy_3d(Ax, Ay, Az, dt, c, dx, dy, dz)

def compute_curl_3d(Ax, Ay, Az, dx, dy, dz):
    """
    Compute the curl of a 3D vector field
    
    Parameters
    ----------
    Ax : ndarray
        X component of vector field
    Ay : ndarray
        Y component of vector field
    Az : ndarray
        Z component of vector field
    dx : float
        Grid spacing in x direction
    dy : float
        Grid spacing in y direction
    dz : float
        Grid spacing in z direction
        
    Returns
    -------
    tuple
        (curl_x, curl_y, curl_z)
    """
    curl_x = np.zeros_like(Ax)
    curl_y = np.zeros_like(Ay)
    curl_z = np.zeros_like(Az)
    
    # dAz/dy - dAy/dz
    curl_x[:, 1:-1, 1:-1] = ((Az[:, 2:, 1:-1] - Az[:, :-2, 1:-1]) / (2*dy) - 
                              (Ay[:, 1:-1, 2:] - Ay[:, 1:-1, :-2]) / (2*dz))
    
    # dAx/dz - dAz/dx
    curl_y[1:-1, :, 1:-1] = ((Ax[1:-1, :, 2:] - Ax[1:-1, :, :-2]) / (2*dz) - 
                              (Az[2:, :, 1:-1] - Az[:-2, :, 1:-1]) / (2*dx))
    
    # dAy/dx - dAx/dy
    curl_z[1:-1, 1:-1, :] = ((Ay[2:, 1:-1, :] - Ay[:-2, 1:-1, :]) / (2*dx) - 
                              (Ax[1:-1, 2:, :] - Ax[1:-1, :-2, :]) / (2*dy))
    
    return curl_x, curl_y, curl_z

def compute_total_energies(snapshots, dt, c, dx, dy, dz=None):
    """
    Compute total scalar and vector energies for all snapshots
    
    Parameters
    ----------
    snapshots : list
        List of snapshot dictionaries
    dt : float
        Time step
    c : float
        Wave speed
    dx : float
        Grid spacing in x direction
    dy : float
        Grid spacing in y direction
    dz : float, optional
        Grid spacing in z direction (for 3D)
        
    Returns
    -------
    tuple
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
