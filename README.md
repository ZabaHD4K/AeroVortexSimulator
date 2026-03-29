# Aero3D Simulator

**Simulador aerodinámico 3D en tiempo real con CFD basado en Lattice Boltzmann Method (LBM) acelerado por GPU.**

Carga cualquier modelo 3D (.stl/.obj/.fbx), simula el flujo de aire a su alrededor y visualiza streamlines, presión, vorticidad y coeficientes aerodinámicos — todo en tiempo real.

> **Este es un proyecto de aprendizaje.** El objetivo es entender e implementar desde cero los fundamentos de Computational Fluid Dynamics (CFD), computación en GPU con CUDA, y visualización científica 3D. No es un reemplazo de herramientas profesionales como ANSYS u OpenFOAM, sino un ejercicio profundo de ingeniería de software, física computacional y programación de alto rendimiento.

---

## Estado Actual

### Fase 0 — Visor 3D base ✅

- Cargar cualquier modelo 3D (.stl, .obj, .fbx, .ply, .gltf...) via Assimp
- Renderizar con iluminación Blinn-Phong (ambient + diffuse + specular)
- Cámara orbital libre (rotar, zoom, pan)
- Grid de referencia con fade-out
- Panel ImGui con info del modelo (vértices, triángulos, meshes)
- Drag & drop de archivos directamente a la ventana
- Controles de render: wireframe, color del modelo, dirección de luz

### Fase 1 — LBM 2D en CUDA ✅

- Solver LBM D2Q9 implementado completamente en CUDA
- Simulación de lid-driven cavity (flujo con tapa móvil)
- Visualización del campo de velocidad como heatmap con colormap viridis
- Parámetros configurables: tau (viscosidad), velocidad de tapa, steps/frame
- Controles start/stop/resume desde la GUI

### Fase 2 — LBM 3D (D3Q19) en CUDA ✅

- Solver LBM 3D con lattice D3Q19 (19 velocidades discretas) en GPU
- Operador de colisión BGK con tau configurable
- Memory layout SoA (Structure of Arrays) para acceso coalescente en GPU
- Buffers host pinneados con lazy copy (GPU → CPU solo cuando se necesita)
- Condiciones de contorno:
  - Bounce-back en sólidos
  - Velocidad prescrita en inlet (f equilibrio)
  - Zero-gradient (Neumann) en outlet
  - Bounce-back en paredes Y/Z
- Campos macroscópicos: densidad, velocidad (ux/uy/uz), presión, vorticidad

### Fase 3 — Voxelización de modelos ✅

- Voxelización por conservative rasterization con Separating Axis Theorem (SAT) para intersección triángulo-AABB
- Dilatación de celdas sólidas (2 voxels) para modelos thin-shell
- Flood-fill desde esquinas exteriores para distinguir exterior de interior
- Marcado automático de caras inlet/outlet según dirección de viento
- Estadísticas de grid (% sólido, fluido, inlet, outlet)

### Fase 4 — Visualización de flujo 3D ✅

- **Streamlines 3D** — Integración RK4 del campo de velocidad, color por magnitud (azul→cian→verde→amarillo→naranja), fade-in/fade-out, coasting a través de zonas estancadas
- **Slice Plane** — Corte 2D del campo 3D (velocidad/presión/vorticidad) con colormap viridis, eje e índice configurables, opacidad 70%
- **Presión superficial** — Mapa de presión sobre el modelo con colormap cool-warm (azul=baja, rojo=alta) + iluminación Blinn-Phong
- **Partículas (jets)** — Jets de viento advectados por el flujo, inyección continua en inlet, ~120 jets × ~150 partículas/jet

### Fase 5 — Coeficientes aerodinámicos ✅

- Integración de presión en fronteras sólido-fluido con stencil de diferencias finitas (6 direcciones)
- Cálculo de área frontal por proyección de celdas sólidas
- Coeficientes: Cd (drag), Cl (lift), Cs (side force)
- Fuerzas brutas (Fx, Fy, Fz) en unidades lattice
- Historial temporal de Cd/Cl para gráficas de convergencia

---

## Limitaciones y errores conocidos

### Precisión de la simulación

- **Operador BGK solamente** — El README original planteaba MRT (Multiple Relaxation Time) que es más estable y preciso. Actualmente solo se usa BGK, lo que limita la estabilidad a Reynolds moderados y puede producir artefactos numéricos con geometrías complejas o velocidades altas.
- **Sin modelo de turbulencia** — No hay Smagorinsky LES implementado. Los vórtices pequeños no se modelan, lo que limita la precisión en flujos turbulentos.
- **Outlet zero-gradient aproximado** — La condición de contorno outlet copia f del vecino interior. Esto puede reflejar ondas de vuelta al dominio, causando inestabilidad a números de Reynolds altos.
- **Sin multi-resolución** — Grid uniforme en todo el dominio. No hay refinamiento adaptativo cerca del objeto, lo que obliga a elegir entre resolución global alta (lento) o baja resolución en la capa límite (impreciso).

### Voxelización

- **Dilatación hardcodeada** — La dilatación de 2 voxels es fija, no configurable. Puede ser excesiva para modelos gruesos o insuficiente para detalles finos.
- **CPU-only** — La voxelización se ejecuta en CPU, no en GPU como estaba planeado. Para grids grandes es lenta.
- **Modelos no watertight** — El flood-fill maneja modelos abiertos, pero no resuelve problemas topológicos complejos (genus > 0).
- **Precisión float** — Umbral de 1e-6f en SAT puede perder precisión con modelos muy grandes o muy pequeños.

### Visualización

- **Slice plane atraviesa sólidos** — El slice plane no filtra celdas sólidas, mostrando valores dentro del modelo que no tienen sentido físico.
- **Streamline seeding fijo** — El patrón de semillas es una grilla regular en el inlet. No se adapta al perfil de velocidad real, por lo que pueden sembrarse todas en zonas muertas o de recirculación.
- **Sin volume rendering** — No se implementó ray marching del campo 3D como estaba planeado.

### Plataforma y portabilidad

- **Windows-only** — Ruta de fuente hardcodeada a `C:\Windows\Fonts\segoeui.ttf` en `main.cpp:52`. Fallará en Linux/Mac o si la fuente no existe.
- **Requiere NVIDIA GPU** — CUDA no es portable a AMD/Intel. El solver depende completamente de CUDA.
- **Sin CUDA error checking** — Las operaciones GPU→CPU (getVelocityMagnitude, etc.) no verifican errores CUDA explícitamente; fallos silenciosos posibles.

### Otros

- **Sin export de datos** — No hay export a CSV, VTK, o screenshots como estaba planeado.
- **Sin validación con geometrías conocidas** — Los modelos de validación (esfera, cilindro, NACA 0012) no se han probado contra valores teóricos.
- **Cleanup en crash** — Si la aplicación crashea durante la simulación, la memoria GPU puede no liberarse correctamente.

---

## Por qué C++ y por qué desde cero

**Rendimiento:** CFD en tiempo real requiere iterar sobre millones de celdas cada frame. C++ da control directo sobre memoria (arrays contiguos, sin garbage collector) y el compilador optimiza agresivamente. En Python sería órdenes de magnitud más lento.

**CUDA:** Los kernels de simulación se escriben en CUDA, que es C/C++. Además, la interoperabilidad CUDA-OpenGL permite compartir buffers GPU sin copiar datos (zero-copy) — el LBM calcula y OpenGL renderiza desde la misma memoria.

**Aprendizaje profundo:** Usar librerías de alto nivel ocultaría exactamente lo que quiero entender. Implementar LBM, voxelización, volume rendering y streamlines desde cero obliga a dominar la física, las matemáticas y la arquitectura GPU.

---

## Core CFD — Lattice Boltzmann Method

### Implementación actual: D3Q19 + BGK

El solver implementa LBM en 3D con 19 velocidades discretas ejecutado en GPU con CUDA.

#### Algoritmo por timestep

1. **Streaming** — Cada distribución f_i se propaga al vecino según su velocidad discreta c_i. Layout SoA mantiene dirección X contigua para coalescing en GPU.
2. **Colisión BGK** — Calcula ρ y u macroscópicos, computa f_i^eq, relaja: `f_new = f - (1/τ)(f - f^eq)`. τ controla viscosidad: `ν = (τ - 0.5)/3`.
3. **Condiciones de contorno** — Bounce-back en sólidos, velocidad prescrita en inlet, zero-gradient en outlet.
4. **Extracción de campos** — Presión `p = ρ/3`, magnitud de velocidad, vorticidad.

#### Ecuaciones

```
Recuperación macroscópica:
  ρ(x,t) = Σᵢ fᵢ(x,t)
  u(x,t) = (1/ρ) Σᵢ fᵢ cᵢ
  p(x,t) = ρ cₛ²    (cₛ² = 1/3)

Distribución de equilibrio:
  fᵢ^eq = wᵢ ρ [1 + 3(cᵢ·u) + 9/2(cᵢ·u)² - 3/2(u·u)]
  Pesos D3Q19: w₀=1/3, w₁₋₆=1/18, w₇₋₁₈=1/36

Colisión BGK:
  fᵢ(x+cᵢ, t+1) = fᵢ(x,t) - (1/τ)[fᵢ(x,t) - fᵢ^eq(x,t)]
```

#### Parámetros por defecto

| Parámetro | Valor | Notas |
|---|---|---|
| Grid | 200×100×100 | 2M celdas |
| Domain scale | 0.4 | Modelo ocupa 40% del ancho |
| Inlet velocity | 0.05 | Unidades lattice (configurable 0.01–0.1) |
| Tau | 0.8 | Controla Re = u·L/ν |
| Direcciones de viento | ±X, ±Y, ±Z | 6 opciones |

#### Rendimiento típico

- 2M celdas × 19 distribuciones ≈ 38M operaciones/step
- 5–10 steps/frame a 60 FPS → 300–600 steps/seg
- Convergencia a estado estacionario: 1000–5000 steps según geometría y Re

### Planificado pero no implementado

- Operador de colisión MRT (Multiple Relaxation Time)
- Turbulencia LES con modelo Smagorinsky
- Bounce-back interpolado para superficies curvas
- Multi-resolución con grid refinement adaptativo

---

## Modos de aplicación

La aplicación tiene tres modos seleccionables desde la GUI:

### 1. Model Viewer
Visor 3D básico con render Blinn-Phong, cámara orbital, grid, drag & drop. Sin simulación.

### 2. LBM 2D
Simulación 2D de lid-driven cavity con D2Q9. Visualización fullscreen del campo de velocidad con heatmap viridis. Parámetros configurables desde la GUI.

### 3. Simulation 3D
Túnel de viento 3D alrededor del modelo cargado. Cinco modos de visualización simultáneos (toggleables):

| Modo | Colormap | Rendering | Descripción |
|---|---|---|---|
| Streamlines | Azul→Naranja | GL_LINE_STRIP | RK4, color por velocidad, fade |
| Slice Plane | Viridis | Quad texturizado, 70% opacidad | Corte por eje configurable |
| Surface Pressure | Cool-Warm | Mesh con Blinn-Phong | Presión interpolada en vértices |
| Particles | Viridis | GL_LINE_STRIP por jet | 120 jets advectados |
| Model | — | Mesh sólido/wireframe | Geometría del modelo |

---

## Tecnologías

| Componente | Tecnología |
|---|---|
| Lenguaje | C++17 |
| Simulación CFD | CUDA (LBM D2Q9 + D3Q19) |
| Render | OpenGL 4.6 |
| UI | Dear ImGui |
| Carga de modelos | Assimp (.stl, .obj, .fbx, .gltf, .ply) |
| Matemáticas | GLM |
| Build | CMake + FetchContent (deps automáticas) |

---

## Estructura del Proyecto

```
AeroVortexSimulator/
├── CMakeLists.txt
├── README.md
├── build_and_run.bat
├── build.bat
├── src/
│   ├── main.cpp                         # Entry point, ventana OpenGL + loop
│   ├── app.cpp/h                        # Aplicación principal (orquesta todo)
│   ├── core/
│   │   ├── lbm2d.cuh/cu                # Solver LBM 2D (D2Q9, CUDA)
│   │   ├── lbm3d.cuh/cu                # Solver LBM 3D (D3Q19, CUDA)
│   │   ├── voxelizer.h/cpp             # Voxelización con SAT (CPU)
│   │   └── aero_forces.h/cpp           # Cálculo de Cd/Cl/Cs
│   ├── geometry/
│   │   ├── model_loader.h/cpp          # Carga de modelos con Assimp
│   │   └── mesh.h                      # Estructuras de datos (Vertex, Mesh, Model)
│   ├── visualization/
│   │   ├── renderer.h/cpp              # Pipeline de render OpenGL (Blinn-Phong + grid)
│   │   ├── camera.h                    # Cámara orbital
│   │   ├── field2d.h/cpp               # Heatmap 2D (para LBM 2D)
│   │   ├── streamlines.h/cpp           # Líneas de corriente 3D (RK4)
│   │   ├── slice_plane.h/cpp           # Plano de corte 3D
│   │   ├── surface_pressure.h/cpp      # Presión sobre superficie del modelo
│   │   └── particles.h/cpp             # Jets de partículas advectadas
│   └── ui/
│       └── gui.h/cpp                   # Paneles ImGui
├── models/                             # Modelos 3D de prueba
└── extern/
    └── glad/                           # OpenGL loader
```

---

## Compilación

### Requisitos
- **Windows 10/11**
- **CMake 3.24+**
- **Visual Studio 2022** (MSVC)
- **CUDA Toolkit 12+** con GPU NVIDIA (Compute Capability 3.0+)
- **GPU** con OpenGL 4.6

### Build
```bash
# Automático
build_and_run.bat

# Manual
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/aero3d_simulator.exe
```

Las dependencias (GLFW, ImGui, Assimp, GLM) se descargan automáticamente con CMake FetchContent en la primera compilación.

---

## Controles

| Acción | Input |
|---|---|
| Rotar cámara | Click izquierdo + arrastrar |
| Zoom | Scroll |
| Pan | Click derecho + arrastrar |
| Reset cámara | R |
| Wireframe | W |
| Toggle grid | G |

---

## Roadmap

- [x] **Fase 0** — Visor 3D base (OpenGL + ImGui + Assimp + cámara orbital)
- [x] **Fase 1** — LBM 2D en CUDA (lid-driven cavity, D2Q9)
- [x] **Fase 2** — LBM 3D (D3Q19 + BGK) *(planificado: MRT)*
- [x] **Fase 3** — Voxelización de modelos en CPU con SAT *(planificado: GPU)*
- [x] **Fase 4** — Streamlines, slice planes, presión superficial, partículas
- [x] **Fase 5** — Cálculo de Cd/Cl/Cs + historial temporal
- [ ] **Fase 6** — Volume rendering + mejoras de visualización
- [ ] **Fase 7** — Export de datos (CSV, VTK, screenshots)
- [ ] **Fase 8** — MRT + Smagorinsky LES + multi-resolución
- [ ] **Fase 9** — Modelos de validación + comparación teórica
- [ ] **Fase 10** — Polish, optimización, portabilidad, documentación final

---

## Modelos de prueba

El proyecto incluye modelos 3D para probar el visor y validar la simulación CFD. Puedes cargarlos desde el botón "Load Model", la librería de modelos en la GUI, o arrastrándolos a la ventana.

| Modelo | Archivo | Uso |
|---|---|---|
| F-104G Starfighter | `models/f104_starfighter.glb` | Geometría aeronáutica, alas cortas + fuselaje |
| Aston Martin AMR23 | `models/amr23_f1.glb` | Coche F1, aerodinámica de suelo + alerones |
| F-16C Falcon | `models/f16c_falcon.glb` | Caza ligero, geometría aerodinámica clásica |

> Los modelos 3D se gestionan con [Git LFS](https://git-lfs.github.com/) debido a su tamaño. Ejecuta `git lfs install` antes de clonar el repositorio.

### Créditos

- ["German F-104G Starfighter"](https://skfb.ly/pwGSY) by 42manako — Licensed under [Creative Commons Attribution 4.0](http://creativecommons.org/licenses/by/4.0/)
- ["Aston Martin F1 AMR23 2023"](https://skfb.ly/oSVBL) by Redgrund — Licensed under [Creative Commons Attribution 4.0](http://creativecommons.org/licenses/by/4.0/)
- ["F16-C Falcon"](https://skfb.ly/osUYX) by Carlos.Maciel — Licensed under [Creative Commons Attribution 4.0](http://creativecommons.org/licenses/by/4.0/)

---

## Autor

**Alejandro Zabaleta** — [GitHub](https://github.com/ZabaHD4K)
