# SovereignDefault-GPU-Solver

The project solves a standard sovereign default setup where a government chooses whether to repay or default on its debt given stochastic income realizations.  
The model follows Arellano (2008, AER) and is implemented to run fully on the GPU for computational efficiency.

The same GPU-based dynamic programming approach is also used in our research paper “Domestic vs. Foreign Law: Portfolio Dynamics of Sovereign Debt” (with Angelo Mendes),
where it enables solving portfolio models with more than one million state points.


Key steps:
1. Construct income and bond grids on the host (CPU).
2. Transfer data to device memory.
3. Execute CUDA kernels for:
   - Initialization of value and price functions
   - Value function iteration under repayment and default
   - Price updating via lenders’ zero-profit condition
   - Policy function and convergence computation
4. Return equilibrium objects to MATLAB as structured output.

## Features

- Dynamic programming loop entirely on GPU
- MATLAB MEX interface for visualization and analysis
- Device constant memory for fast parameter access
- Error tracking via Thrust for clean convergence evaluation

## Project Structure
```
SovereignDefault-GPU-Solver/
│
├── host/
│   ├── helpers.cpp        # Builds grids, income processes, and utilities
│   ├── helpers.h
│
├── device/
│   ├── solver.cu          # CUDA kernels and solver loop
│   ├── solver.h
│   ├── device_constants.h # Parameters stored in __constant__ memory
│
├── mex/
│   ├── mex_entrypoint.cu  # MATLAB interface (MEX gateway)
│
├── README.md
└── LICENSE
```
