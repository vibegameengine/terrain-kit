import * as THREE from 'three/webgpu';
import { Fn, clamp, float, instanceIndex, normalLocal, positionGeometry, positionPrevious, storage, vec2, vec3 } from 'three/tsl';
import { tslOps, type N } from '../gpu/tslOps.ts';
import { levelOfDistance, snappedLocal } from '../lod/lodSelection.ts';
import { patchLevel, patchX, patchY, type TerrainQuadtree } from '../quadtree/terrainQuadtree.ts';

export const STATIC_MOBILITY = 0;

export interface HeightfieldCasterSource {
  readonly geometry: THREE.BufferGeometry;
  readonly capacity: number;
  count(): N;
  patchAt(index: N): N;
  boxOf(packed: N): { low: N; high: N };
  worldOf(packed: N): N;
}

function patchGrid(segments: number): THREE.BufferGeometry {
  const side = segments + 1;
  const positions = new Float32Array(side * side * 3);
  const normals = new Float32Array(side * side * 3);
  for (let y = 0; y < side; y++) for (let x = 0; x < side; x++) {
    positions.set([x / segments, 0, y / segments], (y * side + x) * 3);
    normals.set([0, 1, 0], (y * side + x) * 3);
  }
  const indices = new Uint32Array(segments * segments * 6);
  for (let y = 0, at = 0; y < segments; y++) for (let x = 0; x < segments; x++, at += 6) {
    const corner = y * side + x;
    indices.set([corner, corner + side, corner + 1, corner + side, corner + side + 1, corner + 1], at);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  return geometry;
}

export function terrainWorldPosition(tree: TerrainQuadtree, patch: N): N {
  const { field, maxLevel, segments, reach, densityLevels, leafSize, bounds, viewOrigin } = tree;
  const { minX, minZ } = field.domain;
  const origin = vec2(minX, minZ);
  return Fn(() => {
    const packed = patch.toVar();
    const level = float(patchLevel(packed)).toVar();
    const size = float(leafSize).mul(tslOps.power2(level)).toVar();
    const corner = vec2(float(patchX(packed)), float(patchY(packed))).mul(size).add(origin).toVar();
    const local = vec2(positionGeometry.x, positionGeometry.z).toVar();
    const placed = corner.add(local.mul(size)).toVar();
    const toView = vec3(placed.x, field.heightAt(placed), placed.y).sub(viewOrigin).toVar();
    const wanted = levelOfDistance(tslOps, toView.dot(toView), reach).sub(field.densityAt(placed).mul(densityLevels));
    const levelsAbove = clamp(wanted, level, float(maxLevel)).floor().sub(level).toVar();
    const snapped = vec2(snappedLocal(tslOps, local.x, float(segments), levelsAbove), snappedLocal(tslOps, local.y, float(segments), levelsAbove)).toVar();
    const world = (corner.add(snapped.mul(size)) as N).clamp(vec2(bounds.minX, bounds.minZ), vec2(bounds.maxX, bounds.maxZ)).toVar();
    return vec3(world.x, field.heightAt(world), world.y);
  })();
}

export function createTerrainMesh(tree: TerrainQuadtree, material: THREE.NodeMaterial): THREE.Mesh {
  const geometry = patchGrid(tree.segments);
  geometry.setIndirect(tree.view.drawArgs);
  const leaves = storage(tree.view.leaves, 'uint', tree.capacity).toReadOnly();
  material.positionNode = Fn((builder: N) => {
    const world = terrainWorldPosition(tree, leaves.element(instanceIndex)).toVar();
    normalLocal.assign(tree.field.normalAt(world.xz));
    if (builder.needsPreviousData?.()) positionPrevious.assign(world);
    return world;
  })();
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = 'Terrain';
  mesh.frustumCulled = false;
  Object.assign(mesh, { drawsEveryFrame: true });
  mesh.userData.giExclude = true;
  mesh.userData.clusterLod = false;
  mesh.userData.mobility = STATIC_MOBILITY;
  return mesh;
}

export function terrainShadowCaster(tree: TerrainQuadtree): HeightfieldCasterSource {
  const geometry = patchGrid(tree.segments);
  const leaves = storage(tree.casters.leaves, 'uint', tree.capacity).toReadOnly();
  return {
    geometry,
    capacity: tree.capacity,
    count: () => tree.casterCountNode(),
    patchAt: (index) => leaves.element(index),
    boxOf: (packed) => tree.patchBoxOf(packed),
    worldOf: (packed) => terrainWorldPosition(tree, packed),
  };
}
