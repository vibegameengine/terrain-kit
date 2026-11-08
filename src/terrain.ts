import type * as THREE from 'three/webgpu';
import { TerrainField, type FieldFunction, type TerrainDomain } from './field/terrainField.ts';
import type { LevelRule } from './lod/lodSelection.ts';
import { createTerrainMesh, terrainShadowCaster } from './mesh/terrainMesh.ts';
import { createTerrainProxy } from './mesh/terrainProxy.ts';
import { TerrainQuadtree, type TerrainBounds } from './quadtree/terrainQuadtree.ts';

const FIELD_RESOLUTION = 1024;
const LEAF_METRES = 3;
const PATCH_SEGMENTS = 16;
const PATCH_CAPACITY = 16384;
const DENSITY_LEVELS = 1;
const UNIFORM_LEVELS: LevelRule = { uniformLevel: 1 };

export interface TerrainProxyLattice {
  readonly metresPerVertex: number;
  readonly layer: number;
}

export interface TerrainOptions {
  readonly domain: TerrainDomain;
  readonly field: FieldFunction;
  readonly material: (field: TerrainField) => THREE.NodeMaterial;
  readonly proxy: TerrainProxyLattice;
  readonly bounds?: TerrainBounds;
  readonly resolution?: number;
  readonly maskTextures?: number;
  readonly levels?: LevelRule;
  readonly densityLevels?: number;
  readonly leafMetres?: number;
  readonly segments?: number;
  readonly capacity?: number;
  readonly toLight?: THREE.Vector3;
}

export interface Terrain {
  readonly field: TerrainField;
  readonly tree: TerrainQuadtree;
  readonly surface: THREE.Mesh;
  readonly proxy: THREE.Mesh;
  select(camera: THREE.Camera): void;
  dispose(): void;
}

export async function createTerrain(renderer: THREE.WebGPURenderer, options: TerrainOptions): Promise<Terrain> {
  const { domain } = options;
  const bounds = options.bounds ?? { minX: domain.minX, minZ: domain.minZ, maxX: domain.minX + domain.size, maxZ: domain.minZ + domain.size };
  const field = new TerrainField({ domain, resolution: options.resolution ?? FIELD_RESOLUTION, maskTextures: options.maskTextures ?? 1, field: options.field });
  await field.bake(renderer);
  const tree = new TerrainQuadtree({
    field, bounds,
    maxLevel: Math.ceil(Math.log2(domain.size / (options.leafMetres ?? LEAF_METRES))),
    reach: options.levels ?? UNIFORM_LEVELS,
    segments: options.segments ?? PATCH_SEGMENTS,
    densityLevels: options.densityLevels ?? DENSITY_LEVELS,
    capacity: options.capacity ?? PATCH_CAPACITY,
  });
  await tree.bakeBounds(renderer);
  const surfaceMaterial = options.material(field);
  const proxyMaterial = options.material(field);
  const surface = createTerrainMesh(tree, surfaceMaterial);
  const width = bounds.maxX - bounds.minX;
  const depth = bounds.maxZ - bounds.minZ;
  const { metresPerVertex, layer } = options.proxy;
  const proxy = await createTerrainProxy(renderer, field, proxyMaterial, {
    columns: Math.ceil(width / metresPerVertex) + 1, rows: Math.ceil(depth / metresPerVertex) + 1,
    span: { minX: bounds.minX, minZ: bounds.minZ, width, depth }, layer,
  });
  surface.raycast = proxy.raycast.bind(proxy);
  surface.userData.heightfieldCaster = terrainShadowCaster(tree);
  proxy.userData.standsInForHeightfield = true;
  return {
    field, tree, surface, proxy,
    select: (camera) => tree.select(renderer, camera, options.toLight),
    dispose: () => {
      tree.dispose();
      field.dispose();
      surface.geometry.dispose();
      proxy.geometry.dispose();
      surfaceMaterial.dispose();
      proxyMaterial.dispose();
    },
  };
}
