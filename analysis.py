# analysis.py

import numpy as np

def compute_gradient(field, dx, dy):
    grad_x = np.zeros_like(field)
    grad_y = np.zeros_like(field)
    grad_x[1:-1, :] = (field[2:, :] - field[:-2, :]) / (2*dx)
    grad_y[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2*dy)
    return grad_x, grad_y

def compute_scalar_energy(phi, phi_old, dt, c, dx, dy):
    dphi_dt = (phi - phi_old) / dt
    grad_x, grad_y = compute_gradient(phi, dx, dy)
    grad_phi_sq = grad_x**2 + grad_y**2
    energy = 0.5 * (dphi_dt**2) + 0.5 * (c**2) * grad_phi_sq
    return energy

def compute_vector_energy(Ax, Ay, dt, c, dx, dy):
    grad_Ax_x, grad_Ax_y = compute_gradient(Ax, dx, dy)
    grad_Ay_x, grad_Ay_y = compute_gradient(Ay, dx, dy)
    grad_A_sq = grad_Ax_x**2 + grad_Ax_y**2 + grad_Ay_x**2 + grad_Ay_y**2
    energy = 0.5 * (c**2) * grad_A_sq
    return energy

def compute_total_energies(snapshots, dt, c, dx, dy):
    times = []
    scalar_energy_list = []
    vector_energy_list = []
    for snapshot in snapshots:
        times.append(snapshot.get('time', 0.0))
        # Here, we approximate dphi/dt as zero because phi_old is not stored per snapshot.
        # In a refined analysis, consecutive snapshots should be used.
        scalar_energy = compute_scalar_energy(snapshot['phi'], snapshot['phi'], dt, c, dx, dy)
        vector_energy = compute_vector_energy(snapshot['Ax'], snapshot['Ay'], dt, c, dx, dy)
        scalar_energy_list.append(np.sum(scalar_energy))
        vector_energy_list.append(np.sum(vector_energy))
        snapshot['energy'] = scalar_energy + vector_energy
    return np.array(times), np.array(scalar_energy_list), np.array(vector_energy_list)
