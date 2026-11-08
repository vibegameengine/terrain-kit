import * as THREE from 'three/webgpu';
import type { TerrainField } from '../field/terrainField.ts';
import { STATIC_MOBILITY } from './terrainMesh.ts';

export interface TerrainProxySpan {
  readonly minX: number;
  readonly minZ: number;
  readonly width: number;
  readonly depth: number;
}

export interface TerrainProxyOptions {
  readonly columns: number;
  readonly rows: number;
  readonly span: TerrainProxySpan;
  readonly layer: number;
}

function latticeIndices(columns: number, rows: number): Uint32Array {
  const indices = new Uint32Array((columns - 1) * (rows - 1) * 6);
  for (let row = 0, at = 0; row < rows - 1; row++) for (let column = 0; column < columns - 1; column++, at += 6) {
    const corner = row * columns + column;
    indices.set([corner, corner + columns, corner + 1, corner + columns, corner + columns + 1, corner + 1], at);
  }
  return indices;
}

export function latticeGeometry(heights: Float32Array, { columns, rows, span }: Omit<TerrainProxyOptions, 'layer'>): THREE.BufferGeometry {
  const positions = new Float32Array(columns * rows * 3);
  for (let row = 0; row < rows; row++) for (let column = 0; column < columns; column++) {
    const at = (row * columns + column) * 3;
    positions[at] = span.minX + (column / (columns - 1)) * span.width;
    positions[at + 1] = heights[row * columns + column];
    positions[at + 2] = span.minZ + (row / (rows - 1)) * span.depth;
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setIndex(new THREE.BufferAttribute(latticeIndices(columns, rows), 1));
  geometry.computeVertexNormals();
  return geometry;
}

export async function createTerrainProxy(renderer: THREE.WebGPURenderer, field: TerrainField, material: THREE.Material, options: TerrainProxyOptions): Promise<THREE.Mesh> {
  const heights = await field.readHeights(renderer, options);
  const mesh = new THREE.Mesh(latticeGeometry(heights, options), material);
  mesh.name = 'Terrain proxy';
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.layers.set(options.layer);
  mesh.userData.clusterLod = false;
  mesh.userData.mobility = STATIC_MOBILITY;
  return mesh;
}
