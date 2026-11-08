# @vibegameengine/terrain-kit

GPU-driven terrain for three.js WebGPU.

- The heightfield is computed once on the GPU from a function you give it.
- A quadtree is selected on the GPU every frame. Nothing is built on the main thread.
- Mesh density follows distance and a density map: paint roads, water and carved ground to refine them, and leave flat ground coarse.
- Vertices of a finer patch snap to the coarser grid before the boundary, so neighbouring levels share their edges without stitching geometry.

- Shadow passes use the same surface: the level of a vertex comes from the view, never from the camera drawing it. `terrainShadowCaster` hands a virtual shadow map the patch list and the vertex program.

In React, `<GpuTerrain>` from `@vibegameengine/terrain-kit/r3f` builds and mounts it; `createTerrain` does all the computing outside React, on the GPU. Terrain is static by default.

Needs `three` 0.185 or newer with `WebGPURenderer`. Kernels are compiled ahead of use when the renderer offers `compileComputeAsync`, and on first dispatch otherwise.

MIT licensed.
