# AeroVortexSimulator

Real-time 3D Computational Fluid Dynamics (CFD) simulator with GPU-accelerated Lattice Boltzmann Method (LBM) and interactive visualization. Load any 3D model, voxelize it, and watch the wind flow around it in real time.

> **Proyecto de aprendizaje e ingenieria.** El objetivo es implementar desde cero los fundamentos de CFD, computacion GPU con CUDA, y visualizacion cientifica 3D. No reemplaza herramientas profesionales como ANSYS u OpenFOAM, pero implementa los mismos metodos numericos subyacentes.

---

## Features

### GPU-Accelerated LBM Solver
- **D3Q19 lattice** with 19 discrete velocity directions in 3D
- **Collision models**: BGK (single-relaxation-time) and MRT (multiple-relaxation-time via TRT approximation)
- **Smagorinsky LES** subgrid-scale turbulence model for realistic vortex dynamics
- **Boundary conditions**: bounce-back on solids, prescribed-velocity inlet, zero-gradient (Neumann) outlet
- **6-direction wind**: configurable wind from any axis face (+X, -X, +Y, -Y, +Z, -Z)
- **Velocity ramp-up**: gradual 60% to 100% inlet velocity over 150 steps to prevent numerical shock
- **Stability guards**: automatic reset of divergent cells to freestream equilibrium
- **GPU memory validation**: checks available VRAM before allocation with real-time estimation in GUI
- Runs entirely on CUDA — handles grids up to 400x200x200 in real time on modern GPUs

### Aerodynamic Force Computation
- **Momentum exchange method** (Ladd MEA) computed on GPU via atomic reduction — more accurate than pressure integration on staircase boundaries
- Drag (Cd), lift (Cl), and side force (Cs) coefficients
- Exponential moving average smoothing for stable display
- Real-time force history plots with up to 500 data points
- Pressure-based calculation available as fallback

### Convergence Detection
- **RMS velocity change** monitoring between timesteps (GPU-accelerated)
- Automatic **"[Converged]"** indicator in GUI when RMS < 1e-5
- Helps determine when the flow has reached steady state

### Visualization Modes

| Mode | Method | Description |
|------|--------|-------------|
| **Streamlines** | RK4 integration | Direction-based coloring (cool=aligned, warm=vortex). Surface-hugging seeds that graze the model with up to 60 surface bounces |
| **Wind Particles** | RK2 advection | Lagrangian tracer jets with collision deflection, connected as line strips |
| **Slice Plane** | Texture mapping | Cross-sectional views of velocity, pressure, or vorticity on any axis |
| **Surface Pressure** | Vertex interpolation | Pressure field mapped onto the 3D model surface with Blinn-Phong lighting |
| **Volume Rendering** | GPU ray marching | 3D field visualization with adjustable density and opacity |

### Flow Coloring
CFD-style deviation colormap — color represents the angle between local velocity and freestream direction:

| Color | Meaning |
|-------|---------|
| Ice white / Blue | Aligned with freestream |
| Cyan / Green | Mild deviation |
| Yellow / Orange | Significant separation |
| Red | Vortex core |
| Dark purple | Reverse flow |

### Model Support
- **Import**: STL, OBJ, FBX, PLY, DAE, 3DS, glTF/GLB (via Assimp)
- **Drag & drop** loading
- **Built-in test models**: Sphere, Cylinder, NACA 0012 airfoil — each with reference Cd/Cl values for validation
- **Robust voxelization** using Separating Axis Theorem (SAT) triangle-AABB intersection with flood-fill for interior detection

### Data Export
- **VTK StructuredGrid** (.vts) — open in ParaView for advanced post-processing
- **Coefficient CSV** — Cd/Cl history over time with simulation parameters
- **Flow field CSV** — per-slice statistics (min/max/mean velocity, pressure, vorticity)
- **HTML report** — full simulation dashboard with dark theme, SVG sparkline plots, Reynolds number analysis, convergence assessment, cell classification, and validation comparison
- **Screenshots** (BMP)

### GUI
ImGui-based control panel with:
- Model library browser + file dialog
- Grid resolution presets (Coarse 100^3, Medium 200^3, Fine 300^3) with VRAM estimation
- Physics controls: tau, inlet velocity, collision model, Smagorinsky constant
- Visualization toggles for all rendering modes
- Wind direction selector (6 directions)
- Aero coefficient display with live sparkline plots
- Convergence monitor (RMS change)
- Export buttons for all formats

---

## Architecture

```
src/
+-- core/
|   +-- lbm3d.cuh / .cu       D3Q19 CUDA solver (BGK, MRT, Smagorinsky, momentum exchange)
|   +-- lbm2d.cuh / .cu       D2Q9 lid-driven cavity demo
|   +-- aero_forces.h / .cpp   Pressure-based force calculation (fallback)
|   +-- voxelizer.h / .cpp     SAT-based triangle-AABB voxelization
|   +-- log.h                  Logging macros (DEBUG/INFO/WARN/ERROR)
+-- geometry/
|   +-- mesh.h                 GPU mesh data structures
|   +-- model_loader.h / .cpp  Assimp model loading + normalization
|   +-- primitives.h / .cpp    Procedural sphere, cylinder, NACA 0012
+-- visualization/
|   +-- renderer.h / .cpp      Model rendering with Phong lighting
|   +-- camera.h               Orbit camera with pan/zoom
|   +-- streamlines.h / .cpp   RK4 streamline generation + surface-hugging
|   +-- particles.h / .cpp     Lagrangian particle tracer jets
|   +-- slice_plane.h / .cpp   Cross-sectional field visualization
|   +-- surface_pressure.h/.cpp  Pressure coloring on model surface
|   +-- volume_renderer.h/.cpp   Ray-marched volume rendering
|   +-- field2d.h / .cpp       2D field renderer (for LBM2D demo)
+-- export/
|   +-- data_export.h / .cpp   VTK, CSV, HTML report, screenshot export
+-- ui/
|   +-- gui.h / .cpp           ImGui control panel
+-- app.h / .cpp               Application orchestrator
+-- main.cpp                   Entry point (GLFW window setup)
```

### Key Design Decisions
- **Structure of Arrays (SoA)** memory layout for coalesced GPU reads: `f[q][idx]` where `idx = x + y*nx + z*nx*ny`
- **Ping-pong buffers** for distribution functions — pointer swap instead of data copy
- **Cached field access** — GPU-to-CPU transfers only when data is stale (step-based cache tracking)
- **Fused collide+stream kernel** — single kernel pass handles collision, streaming, and all boundary conditions
- **Adaptive seeding** — streamline/particle seeds auto-sized to model AABB with surface-proximal seeds for close interaction

---

## Build

### Requirements
- **CMake** 3.24+
- **CUDA Toolkit** 11.0+ (tested with 12.x)
- **C++17** compiler (MSVC 2019+, GCC 9+, Clang 10+)
- **GPU**: NVIDIA with compute capability 6.0+ (Pascal or newer)
- ~2 GB VRAM for default 200x100x100 grid

All other dependencies are fetched automatically via CMake FetchContent:
- [GLFW](https://www.glfw.org/) 3.4 — Window/input management
- [Dear ImGui](https://github.com/ocornut/imgui) (docking branch) — Immediate mode GUI
- [GLM](https://github.com/g-truc/glm) — OpenGL Mathematics
- [Assimp](https://github.com/assimp/assimp) — Open Asset Import Library
- GLAD (included) — OpenGL loader

### Build Steps

```bash
# Clone
git clone https://github.com/ZabaHD4K/AeroVortexSimulator.git
cd AeroVortexSimulator

# Configure and build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# Run
./build/Release/aero3d_simulator.exe
```

Or on Windows with the included batch file:
```cmd
build.bat
build\aero3d_simulator.exe
```

> Las dependencias se descargan automaticamente en la primera compilacion (~5 min la primera vez).

---

## Usage

### Quick Start
1. Launch the application
2. Load a 3D model: drag & drop a file, use the file browser, or select from the model library
3. Click **"Start Wind Tunnel"** — the model is automatically voxelized and the simulation begins
4. Orbit (left click), pan (right click), zoom (scroll) to explore the flow
5. Toggle visualization modes in the GUI panel
6. Wait for the "[Converged]" indicator for meaningful aero coefficients

### Physics Controls

| Parameter | Default | Description |
|-----------|---------|-------------|
| Tau | 0.6 | Relaxation time. Lower = higher Reynolds number. Keep > 0.505 for stability |
| Inlet Velocity | 0.05 | Freestream speed in lattice units. Keep < 0.1 for incompressible assumption |
| Collision Model | MRT | MRT is more stable than BGK at same Reynolds number |
| Smagorinsky Cs | 0.12 | LES turbulence constant. 0.1-0.2 typical range |
| Steps/Frame | 10 | LBM iterations per render frame. More = faster convergence |

### Reynolds Number

```
Re = U * L / v,    v = (tau - 0.5) / 3
```

With default settings (tau=0.6, U=0.05, L~40 cells): **Re ~ 60**

For higher Re: decrease tau (toward 0.51) or increase grid resolution.

### Validation
Use the built-in test models (Sphere, Cylinder, NACA 0012) to compare computed Cd/Cl against published reference values. The GUI shows expected vs. actual coefficients and percent error.

### Controls

| Key | Action |
|-----|--------|
| Left click + drag | Orbit camera |
| Right click + drag | Pan |
| Scroll | Zoom |
| R | Reset camera |
| Esc | Exit |

---

## Theory

### Lattice Boltzmann Method

Instead of solving the Navier-Stokes equations directly, LBM simulates fluid as populations of particles moving on a discrete lattice. Each cell stores 19 distribution functions (D3Q19) representing the probability of finding a particle moving in each lattice direction.

**Core equations:**

```
Macroscopic recovery:
  rho = SUM_i f_i
  u   = (1/rho) SUM_i f_i * c_i
  p   = rho * cs^2    (cs^2 = 1/3)

Equilibrium distribution:
  f_i^eq = w_i * rho * [1 + 3(c_i . u) + 9/2(c_i . u)^2 - 3/2(u . u)]

BGK collision + streaming:
  f_i(x + c_i, t+1) = f_i(x,t) - (1/tau)[f_i(x,t) - f_i^eq(x,t)]
```

### MRT Collision
Multiple Relaxation Time transforms populations into moment space, relaxes each moment independently with optimized rates, then transforms back. This provides better stability and accuracy than single-relaxation BGK, especially at higher Reynolds numbers.

### Smagorinsky LES
At coarse resolutions, subgrid turbulence is modeled by locally increasing the effective viscosity based on the strain rate tensor magnitude:
```
tau_eff = 0.5 * (tau + sqrt(tau^2 + 18 * Cs^2 * |S| / rho))
```

### Momentum Exchange Method
Forces on solid bodies are computed by summing the momentum transferred at each bounce-back link:
```
F = SUM_boundary_links (f_q + f_q_opp) * e_q
```
This is the standard Ladd (1994) method and is inherently more accurate than pressure integration on staircase boundaries because it captures both pressure and viscous contributions.

### Boundary Conditions
- **Bounce-back**: populations hitting solid walls reverse direction, enforcing no-slip condition
- **Prescribed velocity inlet**: populations set to equilibrium at desired flow speed
- **Zero-gradient outlet**: populations copied from upstream neighbor (direction-aware for all 6 wind directions)
- **Stability guards**: cells with density outside [0.3, 3.0] or NaN/Inf are reset to freestream equilibrium

---

## Performance

Typical frame rates on a modern GPU (RTX 3060+):

| Grid Size | Steps/Frame | Approx FPS | VRAM |
|-----------|-------------|------------|------|
| 100^3 | 10 | 60+ | ~160 MB |
| 200x100x100 | 10 | 40-60 | ~170 MB |
| 300x150x150 | 10 | 15-30 | ~570 MB |
| 400x200x200 | 5 | 5-15 | ~2.6 GB |

VRAM usage: ~83 bytes per cell (19x2 distribution functions + 7 field arrays + cell type + convergence buffers).

---

## Included Test Models

| Model | File | Use Case |
|-------|------|----------|
| F-104G Starfighter | `models/f104_starfighter.glb` | Aeronautical geometry, short wings + fuselage |
| Aston Martin AMR23 | `models/amr23_f1.glb` | F1 car, ground effect + wings |
| F-16C Falcon | `models/f16c_falcon.glb` | Classic fighter jet aerodynamics |

### Built-in Validation Models
- **Sphere** — Expected Cd ~ 0.47 (Re=100), well-documented reference
- **Cylinder** — Expected Cd ~ 1.2 (Re=100), von Karman vortex street
- **NACA 0012** — Expected Cd ~ 0.012 (Re=500k), symmetric airfoil profile

> Los modelos 3D externos se gestionan con [Git LFS](https://git-lfs.github.com/). Ejecuta `git lfs install` antes de clonar.

### Credits
- ["German F-104G Starfighter"](https://skfb.ly/pwGSY) by 42manako — [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/)
- ["Aston Martin F1 AMR23 2023"](https://skfb.ly/oSVBL) by Redgrund — [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/)
- ["F16-C Falcon"](https://skfb.ly/osUYX) by Carlos.Maciel — [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/)

---

## Limitations

- **Staircase boundaries**: voxelized surfaces create step-like geometry that overestimates drag on streamlined bodies. Interpolated Bounce-Back (IBB/Bouzidi) would improve this but is not yet implemented
- **Incompressible assumption**: valid only for Mach < 0.3 in lattice units (inlet velocity < ~0.1)
- **No thermal effects**: isothermal solver only
- **Single GPU**: no multi-GPU or distributed computing
- **NVIDIA only**: CUDA dependency — no AMD/Intel GPU support
- **No adaptive mesh**: uniform grid everywhere — no local refinement near surfaces

---

## Application Modes

### 1. Model Viewer
Basic 3D viewer with Blinn-Phong rendering, orbit camera, reference grid. No simulation.

### 2. LBM 2D
2D lid-driven cavity simulation (D2Q9) with velocity magnitude heatmap. Useful for learning and parameter exploration.

### 3. Simulation 3D
Full 3D wind tunnel around any loaded model. All visualization modes available simultaneously. Real-time aero coefficients with convergence monitoring.

---

## Logging

The application uses a leveled logging system:
- `[DEBUG]` — Detailed diagnostic info (disabled by default)
- `[INFO]`  — Normal operation messages
- `[WARN]`  — Non-fatal issues
- `[ERROR]` — Critical failures

Log level can be changed via `g_logLevel` in `src/core/log.h`.

---

## References

- Succi, S. — *The Lattice Boltzmann Equation for Fluid Dynamics and Beyond* (2001)
- d'Humieres, D. — [Multiple-relaxation-time lattice Boltzmann models](https://doi.org/10.1098/rsta.2001.0955) (2002)
- Smagorinsky, J. — [General circulation experiments with the primitive equations](https://doi.org/10.1175/1520-0493(1963)091<0099:GCEWTP>2.3.CO;2) (1963)
- Ladd, A.J.C. — [Numerical simulations of particulate suspensions](https://doi.org/10.1017/S0022112094001771) (1994)
- Bouzidi, M. et al. — Momentum transfer of a Boltzmann-lattice fluid with boundaries (2001)

---

## Author

**Alejandro Zabaleta** — [GitHub](https://github.com/ZabaHD4K)

## License

MIT
