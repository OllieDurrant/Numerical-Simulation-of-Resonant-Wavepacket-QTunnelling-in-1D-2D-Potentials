# Quantum Tunnelling Simulation (TDSE Solver)

## Overview

This project implements a numerical solver for the **time-dependent Schrödinger equation (TDSE)** to simulate quantum tunnelling phenomena.

A Gaussian wavepacket is evolved through various potentials, demonstrating key quantum effects such as:

* tunnelling through classically forbidden barriers
* resonant tunnelling in single and double barrier systems
* wavepacket dispersion and interference
* extension to two-dimensional quantum systems

The simulation is performed in **natural units** (( \hbar = m = 1 )) using sparse matrix methods and time integration.

---

## Methods

### Numerical Approach

* Finite-difference discretisation of the Hamiltonian
* Sparse tridiagonal matrix representation of the Laplacian
* Time evolution solved using `scipy.integrate.solve_ivp`
* Complex absorbing potentials (CAP) used to prevent boundary reflections

### Features

* Multiple potential types:

  * single / double / triple barriers
  * wells and harmonic potentials
  * 2D circular and square barriers
* Configurable Gaussian wavepackets with directional momentum
* Analytical comparison for transmission coefficients
* Extraction of transmission and reflection probabilities

---

## Key Results

### 1. Free Wavepacket Evolution

* Numerical solution matches analytical Gaussian spreading
* Confirms correctness of TDSE implementation

### 2. Quantum Tunnelling

* Wavepacket partially transmits through barriers with ( E < V_0 )
* Transmission probabilities extracted numerically
* Strong agreement with analytical solutions for plane waves

### 3. Resonant Tunnelling

* Clear transmission peaks observed for specific energies
* Double barrier systems show quasi-bound states and interference effects

### 4. 2D Extension

* Simulation extended to two spatial dimensions
* Demonstrates:

  * coherent states in harmonic potentials
  * scattering from circular and square barriers
  * interference patterns and tunnelling in 2D geometries

---

## Example Outputs

See included report (`Quantum_Tunnelling.pdf`) for:

* wavepacket evolution snapshots
* transmission vs barrier parameters
* resonant tunnelling spectra
* 2D visualisations and interference patterns

---

## Technologies Used

* Python
* NumPy
* SciPy (sparse matrices, ODE solvers)
* Matplotlib

---

## Key Takeaways

* Numerical TDSE methods accurately reproduce known analytical results
* Wavepacket-based simulations naturally capture energy spread effects absent in plane wave theory
* Theory of resonant tunnelling matches predictions, shown in report
* The framework scales to higher dimensions and more complex potentials
* Provides a foundation for studying advanced quantum systems (e.g. quantum transport, resonant devices)

---


## Author

Oliver Durrant
BSc Theoretical Physics, University of Nottingham

________________________________________
Author
Oliver Durrant
BSc Theoretical Physics, University of Nottingham

