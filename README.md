# @vibegameengine/terrain-kit

GPU-driven terrain for three.js WebGPU.

- The heightfield is computed once on the GPU from a function you give it.
- A quadtree is selected on the GPU every frame. Nothing is built on the main thread.
- Mesh density follows distance and a density map: paint roads, water and carved ground to refine them, and leave flat ground coarse.
- Vertices of a finer patch snap to the coarser grid before the boundary, so neighbouring levels share their edges without stitching geometry.

MIT licensed. Work in progress.
