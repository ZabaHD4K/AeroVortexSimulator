# Aero3D Simulator

**Simulador aerodinámico 3D en tiempo real con CFD basado en Lattice Boltzmann Method (LBM) acelerado por GPU.**

Carga cualquier modelo 3D (.stl/.obj/.fbx), simula el flujo de aire a su alrededor y visualiza streamlines, presión, vorticidad y coeficientes aerodinámicos — todo en tiempo real.

> **Este es un proyecto de aprendizaje.** El objetivo es entender e implementar desde cero los fundamentos de Computational Fluid Dynamics (CFD), computación en GPU con CUDA, y visualización científica 3D. No es un reemplazo de herramientas profesionales como ANSYS u OpenFOAM, sino un ejercicio profundo de ingeniería de software, física computacional y programación de alto rendimiento.

---

## Estado Actual

### Fase 0 — Visor 3D base ✅

El proyecto arranca con una aplicación funcional en C++17/OpenGL que permite:

- Cargar cualquier modelo 3D (.stl, .obj, .fbx, .ply, .gltf...) via Assimp
- Renderizar con iluminación Blinn-Phong (ambient + diffuse + specular)
- Cámara orbital libre (rotar, zoom, pan)
- Grid de referencia con fade-out
- Panel ImGui con info del modelo (vértices, triángulos, meshes)
- Drag & drop de archivos directamente a la ventana
- Controles de render: wireframe, color del modelo, dirección de luz

**Próximo paso:** Implementar LBM 2D en CUDA como prototipo de la simulación.

---

## Por qué C++ y por qué desde cero

**Rendimiento:** CFD en tiempo real requiere iterar sobre millones de celdas cada frame. C++ da control directo sobre memoria (arrays contiguos, sin garbage collector) y el compilador optimiza agresivamente. En Python sería órdenes de magnitud más lento.

**CUDA:** Los kernels de simulación se escriben en CUDA, que es C/C++. Además, la interoperabilidad CUDA-OpenGL permite compartir buffers GPU sin copiar datos (zero-copy) — el LBM calcula y OpenGL renderiza desde la misma memoria.

**Aprendizaje profundo:** Usar librerías de alto nivel ocultaría exactamente lo que quiero entender. Implementar LBM, voxelización, volume rendering y streamlines desde cero obliga a dominar la física, las matemáticas y la arquitectura GPU.

---

## Qué se va a construir

### Core CFD — Lattice Boltzmann Method D3Q19

El corazón del simulador será una implementación de LBM en 3D con 19 velocidades discretas (D3Q19) ejecutada completamente en GPU con CUDA.

- **Modelo de colisión MRT** (Multiple Relaxation Time) — más estable y preciso que el operador BGK simple, permite controlar independientemente la relajación de cada momento físico
- **Turbulencia LES** con modelo Smagorinsky — Large Eddy Simulation que resuelve los vórtices grandes explícitamente y modela los pequeños con viscosidad turbulenta
- **Condiciones de contorno** — Bounce-back interpolado para superficies curvas, capturando geometrías complejas sin escalonamientos artificiales
- **Multi-resolución** — Grid refinement adaptativo cerca del objeto para resolver la capa límite con precisión

#### Ecuación fundamental

En LBM, en lugar de resolver Navier-Stokes directamente, se trabaja con funciones de distribución de partículas `f_i` que colisionan y se propagan en una grid:

```
f_i(x + c_i·dt, t + dt) = f_i(x, t) - M^(-1) · S · (m - m^eq)
```

Donde:
- `f_i` — función de distribución en la dirección `i` (19 direcciones en D3Q19)
- `c_i` — vectores de velocidad discretos
- `M` — matriz de transformación a espacio de momentos
- `S` — matriz diagonal de tasas de relajación
- `m, m^eq` — momentos y momentos de equilibrio

Las magnitudes macroscópicas se recuperan como:
```
densidad:  ρ = Σ f_i
velocidad: u = (1/ρ) · Σ f_i · c_i
presión:   p = ρ · c_s²    (c_s = 1/√3 en unidades lattice)
```

---

### Geometría y Voxelización

| Paso | Descripción |
|---|---|
| 1. Carga | Assimp importa .stl/.obj/.fbx y extrae la malla de triángulos |
| 2. Normalización | Escala y centra el modelo en el dominio de simulación |
| 3. Voxelización | Conservative rasterization en GPU — cada celda que intersecta un triángulo se marca como sólida |
| 4. Watertight check | Detección automática de mallas no cerradas + reparación (flood fill desde exterior) |
| 5. Boundary flags | Cada celda sólida vecina a fluido se marca como boundary con su normal interpolada |

---

### Visualización

Múltiples modos simultáneos para analizar el flujo:

- **Streamlines 3D** — Integración Runge-Kutta 4º orden, color por velocidad/presión/vorticidad
- **Volume Rendering** — Ray marching del campo 3D con transfer function configurable
- **Slice Planes** — Planos de corte arbitrarios con heatmap + vectores de velocidad
- **Presión superficial** — Mapa de Cp sobre el modelo como heatmap
- **Partículas advectadas** — Humo virtual transportado por el flujo, tipo túnel de viento

---

### Coeficientes Aerodinámicos

Cálculo en tiempo real de:
- `Cd` — Coeficiente de drag (resistencia)
- `Cl` — Coeficiente de lift (sustentación)
- `Cm` — Coeficiente de momento (pitch)
- Gráficas temporales de convergencia
- Número de Reynolds del flujo

---

### Modelos de Validación

Geometrías con soluciones teóricas conocidas para verificar la precisión:

| Modelo | Cd teórico | Uso |
|---|---|---|
| Esfera | 0.47 (Re ~10⁴) | Validación básica de drag |
| Cilindro infinito | 1.2 (Re ~10⁴) | Vórtices de Von Kármán |
| Placa plana | 1.28 (perpendicular) | Caso límite simple |
| Perfil NACA 0012 | ~0.006 (Re ~10⁶) | Validación aeronáutica |
| Cubo | ~1.05 | Geometría con aristas vivas |

---

## Tecnologías

| Componente | Tecnología |
|---|---|
| Lenguaje | C++17 |
| Simulación CFD | CUDA (LBM D3Q19) |
| Render | OpenGL 4.6 |
| UI | Dear ImGui |
| Carga de modelos | Assimp (.stl, .obj, .fbx, .gltf, .ply) |
| Matemáticas | GLM |
| Build | CMake + FetchContent (deps automáticas) |

---

## Estructura del Proyecto

```
aero3d-simulator/
├── CMakeLists.txt
├── README.md
├── build_and_run.bat
├── src/
│   ├── main.cpp                    # Entry point, ventana OpenGL + loop
│   ├── app.cpp/h                   # Aplicación principal (orquesta todo)
│   ├── core/                       # [TODO] Simulación CFD
│   │   ├── lbm_solver.cu           # Kernel CUDA — streaming + colisión MRT
│   │   ├── lbm_constants.h         # D3Q19 weights, velocidades, matrices
│   │   ├── boundary.cu             # Condiciones de contorno
│   │   └── turbulence.cu           # Modelo Smagorinsky LES
│   ├── geometry/
│   │   ├── model_loader.cpp/h      # Carga de modelos con Assimp
│   │   ├── mesh.h                  # Estructuras de datos (Vertex, Mesh, Model)
│   │   └── voxelizer.cu            # [TODO] Voxelización en GPU
│   ├── visualization/
│   │   ├── renderer.cpp/h          # Pipeline de render OpenGL
│   │   ├── camera.h                # Cámara orbital
│   │   ├── streamlines.cu          # [TODO] Líneas de corriente
│   │   ├── volume_render.cu        # [TODO] Volume rendering
│   │   └── particles.cu            # [TODO] Partículas advectadas
│   └── ui/
│       └── gui.cpp/h               # Paneles ImGui
├── models/                         # Modelos de validación
├── shaders/                        # [TODO] Shaders GLSL externos
└── extern/
    └── glad/                       # OpenGL loader
```

---

## Compilación

### Requisitos (fase actual)
- **Windows 10/11**
- **CMake 3.24+**
- **Visual Studio 2022** (MSVC)
- **GPU** con OpenGL 4.6

> En fases posteriores se necesitará CUDA Toolkit 12+ y GPU NVIDIA con Compute Capability 6.0+.

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
- [ ] **Fase 1** — LBM 2D en CUDA (validar con lid-driven cavity)
- [ ] **Fase 2** — Extender a 3D (D3Q19 + MRT)
- [ ] **Fase 3** — Voxelización de modelos en GPU
- [ ] **Fase 4** — Streamlines y visualización de presión
- [ ] **Fase 5** — Volume rendering + partículas
- [ ] **Fase 6** — Cálculo de Cd/Cl/Cm + gráficas
- [ ] **Fase 7** — Slice planes + export (CSV, VTK, screenshots)
- [ ] **Fase 8** — Multi-resolución + Smagorinsky LES
- [ ] **Fase 9** — Modelos de validación + comparación teórica
- [ ] **Fase 10** — Polish, optimización, documentación final

---

## Autor

**Alejandro Zabaleta** — [GitHub](https://github.com/ZabaHD4K)
