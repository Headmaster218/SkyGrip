# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is SBMPC - a generic sampling-based Model Predictive Control library built on JAX, implementing the Feedback-MPPI method for high-frequency state feedback corrections in robotic control applications.

## Development Commands

### Environment Setup
```bash
# Create conda environment
mamba env create -f environment.yml

# Activate environment
conda activate sbmpc

# Install package (choose based on your setup):
pip install -e .                    # CPU-only
pip install -e ".[cuda12]"          # GPU with pip-installed CUDA
pip install -e ".[cuda12_local]"    # GPU with locally installed CUDA
```

### Building and Versioning
```bash
# Build package (test build without tagging)
python create_tags_and_build.py hotfix

# Create new release with git tag
python create_tags_and_build.py hotfix --tag_and_push  # Increment patch version
python create_tags_and_build.py minor --tag_and_push   # Increment minor version
python create_tags_and_build.py major --tag_and_push   # Increment major version

# Build wheel using hatch
hatch build

# Force reinstall wheel for testing (same version)
pip install ./dist/sbmpc-X.X.X-py3-none-any.whl --force-reinstall --no-deps
```

### Running Examples
```bash
# Run quadrotor control example
cd examples && python quadrotor.py

# Run other examples (from examples directory)
python franka_kinematic_control.py
python quadrotor_obstacles.py
python unicycle.py
```

## Architecture Overview

### Core Module Structure

The library is organized around a modular architecture for sampling-based MPC:

1. **Model Layer** (`sbmpc/model.py`):
   - `BaseModel`: Abstract base class defining the interface for dynamics models
   - `ModelParametric`: Implements parametric dynamics with multiple integration schemes (Euler, RK4, semi-implicit Euler)
   - `ModelMjx`: MuJoCo integration for physics simulation
   - Models handle state propagation, sensitivity computation, and maintain dimensionality (nq, nv, nu, np)

2. **Solver Layer** (`sbmpc/solvers.py`):
   - `RolloutGenerator`: Core MPC solver that generates and evaluates trajectory rollouts
   - `BaseObjective`: Abstract class for defining cost functions and constraints
   - Implements parallel rollout evaluation using JAX's vmap
   - Supports sensitivity computation for feedback gains
   - Handles control interpolation (spline smoothing) and input clipping

3. **Supporting Components**:
   - `sampler.py`: Generates control samples for the MPPI algorithm
   - `gains.py`: Computes feedback gains from rollout sensitivities
   - `filter.py`: Implements cubic spline interpolation for control smoothing
   - `settings.py`: Configuration management for MPC parameters
   - `simulation.py`: High-level simulation utilities
   - `geometry.py`: Geometric utilities for robotics
   - `obstacle_loader.py`: Tools for loading and handling obstacles in simulations

### Key Design Patterns

- **JAX-based Computation**: All core computations use JAX for automatic differentiation and JIT compilation
- **Vectorized Operations**: Extensive use of `vmap` for parallel trajectory evaluation
- **Configurable Integration**: Support for multiple integration schemes selected at runtime
- **Modular Objectives**: Separation of dynamics (Model) from task specification (Objective)

### Examples Directory

Contains demonstration scripts for various robotic systems:
- Quadrotor control (`quadrotor.py`, `quadrotor_obstacles.py`)
- Manipulator control (`franka_kinematic_control.py`)
- Hybrid systems (`quadrotor_arm_test.py`, `quadrotor_skygrip.py`)
- Test configurations (`task_configs.py`, `mppi_gains_test.py`)

Each example includes corresponding MuJoCo XML models and asset files in subdirectories.

## Important Notes

- The library uses JAX's JIT compilation extensively - be aware of compilation overhead on first run
- GPU acceleration requires appropriate CUDA setup as specified in installation
- Version management is tied to git tags via hatch-vcs
- The Feedback-MPPI method is the primary algorithm implementation
- Examples require MuJoCo XML scene files located in subdirectories (e.g., `bitcraze_crazyflie_2/`, `franka_emika_panda/`)
- Integration schemes available: `"si_euler"`, `"euler"`, `"rk4"`, `"custom_discrete"` (defined in `MODEL_PARAMETRIC_INTEGRATOR_TYPES`)
- Core dependencies: JAX, NumPy, MuJoCo-MJX, SciPy, Control, Matplotlib, Interpax