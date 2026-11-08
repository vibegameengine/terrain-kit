export { numberOps, type Ops } from './lod/ops.ts';
export { continuousLevel, distanceSquaredToBox, levelOfDistance, patchSubdivides, snappedLocal, type Box3, type LevelReach, type LevelRule, type PatchRequest, type UniformLevel, type Point3 } from './lod/lodSelection.ts';
export { levelBase, levelSide, patchCountOfTree } from './lod/patchAddress.ts';
export { tslOps } from './gpu/tslOps.ts';
export { TerrainField, type FieldFunction, type FieldLattice, type FieldSample, type TerrainDomain, type TerrainFieldOptions } from './field/terrainField.ts';
export { TerrainQuadtree, type PatchList, type TerrainBounds, type TerrainQuadtreeOptions } from './quadtree/terrainQuadtree.ts';
export { createTerrainMesh, terrainShadowCaster, terrainWorldPosition, type HeightfieldCasterSource } from './mesh/terrainMesh.ts';
export { planeForCasters, type Plane } from './lod/castingPlanes.ts';
export { createTerrainProxy, latticeGeometry, type TerrainProxyOptions, type TerrainProxySpan } from './mesh/terrainProxy.ts';
