
# Scalar Wave Simulation Framework

This repository contains an advanced simulation framework designed to model the excitation and propagation of scalar (longitudinal) electromagnetic waves within a resonant structure. The framework has been specifically enhanced to facilitate research into transmitter designs that predominantly excite the scalar mode, as well as to help optimize resonant cavity geometries and boundary conditions.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [File Structure](#file-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Running the Simulation](#running-the-simulation)
  - [Parameter Sweep](#parameter-sweep)
- [Visualization and Data Output](#visualization-and-data-output)
- [Analysis](#analysis)
- [Future Enhancements](#future-enhancements)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Overview

The Scalar Wave Simulation Framework is a Python-based tool that implements a finite-difference time-domain (FDTD) scheme (with optional nonlinearity) to solve modified Maxwell equations in terms of the scalar potential (ϕ) and vector potential (A). The primary goal is to study the conditions under which a predominantly scalar (longitudinal) mode can be excited in a resonant cavity or parallel plate configuration.

The simulation is particularly useful for experimental designs where:
- A modulated high-voltage source is used to excite the resonant structure.
- Advanced boundary conditions (e.g., impedance boundaries) are needed to support scalar modes.
- A detailed energy analysis is required to isolate the scalar component from conventional transverse modes.

---

## Features

- **Scalar and Vector Field Simulation:**  
  Models both the scalar potential ϕ and the vector potential A using an FDTD leapfrog integration scheme.

- **Time-Dependent Source:**  
  Includes a configurable, modulated source term (e.g., Gaussian with amplitude modulation) that simulates the high-frequency modulated voltage source used in transmitter experiments.

- **Advanced Boundary Conditions:**  
  Supports advanced impedance-based boundary conditions to better simulate physical resonant structures, beyond standard Dirichlet and Neumann options.

- **Nonlinear Effects (Optional):**  
  Provides an option to include cubic nonlinear effects, which may become significant at high voltages.

- **Energy Analysis:**  
  Computes energy densities for both scalar and vector fields, and produces time-series plots to aid in the isolation of the scalar mode.

- **Parameter Sweep:**  
  Contains a dedicated script to perform parameter sweeps (e.g., varying impedance values) to explore the design space for optimal transmitter performance.

- **Visualization:**  
  Offers extensive visualization capabilities:
  - Static snapshots of ϕ, A, and energy density.
  - Animated visualizations (saved as MP4) showing the evolution of the fields.
  - Energy time-series plots.

---

## File Structure

```
scalar_simulation/
├── config/
│   └── config.yaml          # Main configuration file for simulation parameters
├── docs/                    # Documentation (if needed)
├── modules/
│   ├── geometry.py          # Generates domain mesh and defines resonant structure geometries
│   ├── grid.py              # Sets up the spatial grid and computes grid spacing
│   ├── initialization.py    # Initializes scalar and vector potential fields
│   ├── solver.py            # Implements the FDTD scheme with source and nonlinearity
│   ├── visualization.py     # Contains functions for static and animated visualizations
│   └── data_output.py       # Routines to save snapshots, checkpoints, and energy data
├── analysis.py              # Functions to compute gradients and energy densities
├── parameter_sweep.py       # Script to run parameter sweeps (e.g., impedance values)
├── main.py                  # Main driver script that integrates all modules and runs the simulation
├── requirements.txt         # List of required Python packages
└── README.md                # This file
```

---

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/scalar_simulation.git
   cd scalar_simulation
   ```

2. **Set Up a Virtual Environment (Recommended):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Python Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Install FFmpeg:**  
   Matplotlib uses FFmpeg for saving animations. On Ubuntu, install FFmpeg via:

   ```bash
   sudo apt update
   sudo apt install ffmpeg
   ```

5. **Verify Installation:**  
   Check that FFmpeg is installed by running:

   ```bash
   ffmpeg -version
   ```

---

## Configuration

The main configuration file is located at `config/config.yaml`. This file includes settings for:

- **Grid Parameters:**  
  Set the number of grid points and the physical domain dimensions (normalized units).

- **Time Parameters:**  
  Configure the time step (`dt`) and total simulation time.

- **Material Parameters:**  
  Use normalized permittivity and permeability (default: 1.0) to set \( c = 1 \).

- **Initial Conditions:**  
  Specify the amplitude for the initial scalar potential perturbation and vector potential.

- **Boundary Conditions:**  
  Choose the type (`dirichlet`, `neumann`, or `advanced`) and set the impedance value for advanced boundary conditions.

- **Source Settings:**  
  Enable and configure the modulated source (type, amplitude, frequency, center, and modulation scheme).

- **Nonlinearity:**  
  Optionally enable nonlinear effects and set the nonlinearity strength.

- **Output Settings:**  
  Set the snapshot interval and output directory for generated files.

Review and adjust these parameters as needed to match your experimental design.

---

## Usage

### Running the Simulation

To run the simulation, execute the main driver script:

```bash
python main.py
```

This script performs the following steps:
- Loads the configuration.
- Initializes the simulation grid and fields.
- Runs the time-stepping loop (FDTD) with the specified source, boundary conditions, and optional nonlinearity.
- Computes energy densities for analysis.
- Saves snapshots, checkpoints, and energy data.
- Generates static plots and an animation (MP4) of the evolving fields.
- Produces an energy time-series plot.

All outputs are saved in the directory specified by `output_dir` in `config/config.yaml`.

### Parameter Sweep

To explore different design parameters (e.g., varying impedance values), run:

```bash
python parameter_sweep.py
```

This script:
- Iterates over a predefined range of impedance values.
- Runs a short simulation for each parameter set.
- Computes metrics (e.g., maximum scalar amplitude).
- Saves the results to a CSV file (`parameter_sweep_results.csv`).

Review the CSV file to identify optimal parameter ranges for transmitter design.

---

## Visualization and Data Output

The simulation framework provides several visualization tools:

- **Static Snapshots:**  
  Generated for each snapshot step (PNG files) showing:
  - **Left Panel:** Scalar potential (ϕ) as a heatmap.
  - **Middle Panel:** Vector potential (A) via quiver plots.
  - **Right Panel:** Energy density maps (if energy data is available).

- **Animation:**  
  An MP4 animation (`simulation_animation.mp4`) is created to visualize the time evolution of ϕ and A.

- **Energy Time-Series:**  
  A plot of scalar and vector energy over time is generated and saved (e.g., `energy_time_series.png`).

- **Data Files:**  
  Snapshots and checkpoints are saved as compressed NPZ files for further analysis.

---

## Analysis

The `analysis.py` module provides functions to:
- Compute spatial gradients.
- Calculate energy densities for the scalar and vector fields.
- Aggregate energy metrics over time.

These tools are integrated into the main simulation workflow to facilitate mode isolation analysis. They help quantify the energy associated with the scalar mode versus the transverse modes, guiding design improvements.

---

## Future Enhancements

Potential future improvements include:
- **Refined Source Modeling:**  
  Implement additional source types and more sophisticated modulation schemes.
- **Extended Boundary Conditions:**  
  Incorporate custom boundary functions and additional mixed-type conditions.
- **Higher-Dimensional Simulations:**  
  Extend the simulation from 2D to 3D for more realistic experimental setups.
- **Interactive Visualization:**  
  Use interactive libraries (e.g., Plotly) to allow real-time exploration of simulation data.
- **Advanced Nonlinearity:**  
  Implement more complex nonlinear models relevant for high-voltage regimes.

---

## Troubleshooting

- **Numerical Instabilities / Overflow Errors:**  
  Ensure that your time step (`dt`) satisfies the CFL condition relative to your grid spacing. In normalized units with \( c = 1 \), dt should be sufficiently small (e.g., 0.001 for a grid with 100 points over a unit length).

- **FFmpeg Issues:**  
  If animations fail to save, verify that FFmpeg is correctly installed and available in your system's PATH.

- **Visualization Warnings:**  
  Warnings related to quiver plots may occur if the vector field is zero or nearly zero. These warnings are generally non-critical.

- **Parameter Sweep Failures:**  
  Check that the configuration file is correctly formatted and that the parameter sweep script is updating configuration values as expected.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

