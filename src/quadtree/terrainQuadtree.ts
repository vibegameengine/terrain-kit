import * as THREE from 'three/webgpu';
import { Fn, If, Loop, atomicAdd, atomicLoad, atomicStore, float, instanceIndex, int, ivec2, max, min, storage, textureLoad, uint, uniform, uniformArray, vec2, vec4 } from 'three/tsl';
import type { TerrainField } from '../field/terrainField.ts';
import { tslOps, type N } from '../gpu/tslOps.ts';
import { continuousLevel, patchSubdivides, type LevelReach } from '../lod/lodSelection.ts';
import { planeForCasters } from '../lod/castingPlanes.ts';
import { levelBase, levelSide, patchCountOfTree } from '../lod/patchAddress.ts';

export interface TerrainQuadtreeOptions {
  readonly field: TerrainField;
  readonly maxLevel: number;
  readonly reach: LevelReach;
  readonly segments: number;
  readonly densityLevels: number;
  readonly capacity: number;
  readonly bounds?: TerrainBounds;
}

export interface TerrainBounds {
  readonly minX: number;
  readonly minZ: number;
  readonly maxX: number;
  readonly maxZ: number;
}

const KERNEL_GROUP = 64;
const COUNT_CANDIDATES = 0;
const COUNT_NEXT = 1;
const COUNT_LEAVES = 2;
const INSTANCE_COUNT_WORD = 1;
const INDEXED_INDIRECT_WORDS = 5;
const LEVEL_BITS = 5;
const COORDINATE_BITS = 13;

export const packPatch = (x: N, y: N, level: N): N => uint(x).shiftLeft(uint(COORDINATE_BITS + LEVEL_BITS)).bitOr(uint(y).shiftLeft(uint(LEVEL_BITS))).bitOr(uint(level));
export const patchLevel = (packed: N): N => packed.bitAnd(uint((1 << LEVEL_BITS) - 1));
export const patchY = (packed: N): N => packed.shiftRight(uint(LEVEL_BITS)).bitAnd(uint((1 << COORDINATE_BITS) - 1));
export const patchX = (packed: N): N => packed.shiftRight(uint(COORDINATE_BITS + LEVEL_BITS));

export class TerrainQuadtree {
  readonly field: TerrainField;
  readonly maxLevel: number;
  readonly reach: LevelReach;
  readonly segments: number;
  readonly densityLevels: number;
  readonly capacity: number;
  readonly leafSize: number;
  readonly bounds: TerrainBounds;
  readonly leaves: THREE.StorageBufferAttribute;
  readonly drawArgs: THREE.IndirectStorageBufferAttribute;
  private readonly patchBounds: THREE.StorageBufferAttribute;
  private readonly candidates: readonly THREE.StorageBufferAttribute[];
  private readonly counts: THREE.StorageBufferAttribute;
  private readonly boundsLevel = uniform(0, 'uint');
  readonly viewOrigin = uniform(new THREE.Vector3());
  private readonly planes = uniformArray(Array.from({ length: 6 }, () => new THREE.Vector4()));
  private readonly bases: N;
  private readonly leafBoundsKernel: N;
  private readonly parentBoundsKernel: N;
  private readonly resetKernel: N;
  private readonly walkKernels: readonly N[];
  private readonly finishLevelKernel: N;
  private readonly publishKernel: N;
  private readonly frustum = new THREE.Frustum();
  private readonly viewProjection = new THREE.Matrix4();

  constructor(options: TerrainQuadtreeOptions) {
    this.field = options.field;
    this.maxLevel = options.maxLevel;
    this.reach = options.reach;
    this.segments = options.segments;
    this.densityLevels = options.densityLevels;
    this.capacity = options.capacity;
    this.leafSize = options.field.domain.size / 2 ** options.maxLevel;
    const { minX, minZ, size } = options.field.domain;
    this.bounds = options.bounds ?? { minX, minZ, maxX: minX + size, maxZ: minZ + size };
    const patches = patchCountOfTree(options.maxLevel);
    this.patchBounds = new THREE.StorageBufferAttribute(new Float32Array(patches * 4), 4);
    this.candidates = [0, 1].map(() => new THREE.StorageBufferAttribute(new Uint32Array(options.capacity), 1));
    this.leaves = new THREE.StorageBufferAttribute(new Uint32Array(options.capacity), 1);
    this.counts = new THREE.StorageBufferAttribute(new Uint32Array(3), 1);
    this.drawArgs = new THREE.IndirectStorageBufferAttribute(new Uint32Array([segmentsIndexCount(options.segments), 0, 0, 0, 0]), 1);
    this.bases = uniformArray(Array.from({ length: options.maxLevel + 2 }, (_, level) => levelBase(options.maxLevel, level)), 'uint');
    this.leafBoundsKernel = this.buildLeafBounds().compute(levelSide(options.maxLevel, 0) ** 2, [KERNEL_GROUP]).setName('Terrain.LeafBounds');
    this.parentBoundsKernel = this.buildParentBounds().compute(1, [KERNEL_GROUP]).setName('Terrain.ParentBounds');
    this.resetKernel = this.buildReset().compute(1, [1]).setName('Terrain.Reset');
    this.walkKernels = [0, 1].map((side) => this.buildWalk(side).compute(options.capacity, [KERNEL_GROUP]).setName('Terrain.Walk'));
    this.finishLevelKernel = this.buildFinishLevel().compute(1, [1]).setName('Terrain.FinishLevel');
    this.publishKernel = this.buildPublish().compute(1, [1]).setName('Terrain.Publish');
  }

  private boundsNode(): N {
    return storage(this.patchBounds, 'vec4', this.patchBounds.count);
  }

  private buildLeafBounds(): N {
    const bounds = this.boundsNode();
    const side = levelSide(this.maxLevel, 0);
    const resolution = this.field.resolution;
    const texelsPerLeaf = Math.ceil(resolution / side);
    return Fn(() => {
      const index = instanceIndex.toVar();
      const leaf = ivec2(int(index.mod(uint(side))), int(index.div(uint(side)))).toVar();
      const first = leaf.mul(int(texelsPerLeaf)).toVar();
      const low = float(1e20).toVar();
      const high = float(-1e20).toVar();
      const most = float(0).toVar();
      Loop({ start: int(-1), end: int(texelsPerLeaf + 1), type: 'int', condition: '<', name: 'across' }, ({ across }: N) => {
        Loop({ start: int(-1), end: int(texelsPerLeaf + 1), type: 'int', condition: '<', name: 'along' }, ({ along }: N) => {
          const texel = (first.add(ivec2(across, along)) as N).clamp(ivec2(0), ivec2(resolution - 1)).toVar();
          const height = textureLoad(this.field.height, texel, int(0)).r.toVar();
          low.assign(min(low, height));
          high.assign(max(high, height));
          most.assign(max(most, textureLoad(this.field.density, texel, int(0)).r));
        });
      });
      bounds.element(index).assign(vec4(low, high, most, 0));
    })();
  }

  private buildParentBounds(): N {
    const bounds = this.boundsNode();
    return Fn(() => {
      const level = this.boundsLevel.toVar();
      const side = uint(1).shiftLeft(uint(this.maxLevel).sub(level)).toVar();
      const index = instanceIndex.toVar();
      const x = index.mod(side).toVar();
      const y = index.div(side).toVar();
      const childSide = side.mul(uint(2));
      const childBase = this.bases.element(level.sub(uint(1))).toVar();
      const low = float(1e20).toVar();
      const high = float(-1e20).toVar();
      const most = float(0).toVar();
      for (const [dx, dy] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
        const child = vec4(bounds.element(childBase.add(y.mul(uint(2)).add(uint(dy)).mul(childSide)).add(x.mul(uint(2)).add(uint(dx))))).toVar();
        low.assign(min(low, child.x));
        high.assign(max(high, child.y));
        most.assign(max(most, child.z));
      }
      bounds.element(this.bases.element(level).add(index)).assign(vec4(low, high, most, 0));
    })();
  }

  private countsNode(): N {
    return storage(this.counts, 'uint', this.counts.count).toAtomic();
  }

  private buildReset(): N {
    const counts = this.countsNode();
    const first = storage(this.candidates[0], 'uint', this.capacity);
    return Fn(() => {
      first.element(0).assign(packPatch(uint(0), uint(0), uint(this.maxLevel)));
      atomicStore(counts.element(uint(COUNT_CANDIDATES)), uint(1));
      atomicStore(counts.element(uint(COUNT_NEXT)), uint(0));
      atomicStore(counts.element(uint(COUNT_LEAVES)), uint(0));
    })();
  }

  private buildWalk(side: number): N {
    const counts = this.countsNode();
    const bounds = storage(this.patchBounds, 'vec4', this.patchBounds.count).toReadOnly();
    const input = storage(this.candidates[side], 'uint', this.capacity).toReadOnly();
    const output = storage(this.candidates[1 - side], 'uint', this.capacity);
    const leaves = storage(this.leaves, 'uint', this.capacity);
    const { minX, minZ } = this.field.domain;
    const leafSize = this.leafSize;
    return Fn(() => {
      If(instanceIndex.lessThan(atomicLoad(counts.element(uint(COUNT_CANDIDATES))) as N), () => {
        const packed = input.element(instanceIndex).toVar();
        const level = patchLevel(packed).toVar();
        const x = patchX(packed).toVar();
        const y = patchY(packed).toVar();
        const sideOfLevel = uint(1).shiftLeft(uint(this.maxLevel).sub(level)).toVar();
        const stored = vec4(bounds.element(this.bases.element(level).add(y.mul(sideOfLevel)).add(x))).toVar();
        const size = float(leafSize).mul(float(uint(1).shiftLeft(level))).toVar();
        const lowX = float(x).mul(size).add(minX).toVar();
        const lowZ = float(y).mul(size).add(minZ).toVar();
        const nearestX = this.viewOrigin.x.clamp(lowX, lowX.add(size));
        const nearestY = this.viewOrigin.y.clamp(stored.x, stored.y);
        const nearestZ = this.viewOrigin.z.clamp(lowZ, lowZ.add(size));
        const dx = this.viewOrigin.x.sub(nearestX);
        const dy = this.viewOrigin.y.sub(nearestY);
        const dz = this.viewOrigin.z.sub(nearestZ);
        const nearestLevel = continuousLevel(tslOps, dx.mul(dx).add(dy.mul(dy)).add(dz.mul(dz)), this.reach);
        const insideBounds = lowX.lessThan(this.bounds.maxX).and(lowZ.lessThan(this.bounds.maxZ)).and(lowX.add(size).greaterThan(this.bounds.minX)).and(lowZ.add(size).greaterThan(this.bounds.minZ));
        const visible = this.boxVisible(vec2(lowX, lowZ), size, vec2(stored.x, stored.y)).and(insideBounds).toVar();
        If(visible, () => {
          const subdivides = patchSubdivides(tslOps, { level: float(level), nearestLevel, densityBoost: stored.z.mul(this.densityLevels) });
          If(subdivides, () => {
            const slot = (atomicAdd(counts.element(uint(COUNT_NEXT)), uint(4)) as N).toVar();
            If(slot.add(uint(4)).lessThanEqual(uint(this.capacity)), () => {
              const finer = level.sub(uint(1));
              for (const [cx, cy] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
                output.element(slot.add(uint(cx + cy * 2))).assign(packPatch(x.mul(uint(2)).add(uint(cx)), y.mul(uint(2)).add(uint(cy)), finer));
              }
            });
          }).Else(() => {
            const slot = (atomicAdd(counts.element(uint(COUNT_LEAVES)), uint(1)) as N).toVar();
            If(slot.lessThan(uint(this.capacity)), () => { leaves.element(slot).assign(packed); });
          });
        });
      });
    })();
  }

  private boxVisible(low: N, size: N, heights: N): N {
    const inside = uint(1).toVar();
    Loop({ start: int(0), end: int(6), type: 'int', condition: '<' }, ({ i }: N) => {
      const plane = (this.planes.element(i) as N).toVar();
      const px = plane.x.greaterThan(0).select(low.x.add(size), low.x);
      const py = plane.y.greaterThan(0).select(heights.y, heights.x);
      const pz = plane.z.greaterThan(0).select(low.y.add(size), low.y);
      If(plane.x.mul(px).add(plane.y.mul(py)).add(plane.z.mul(pz)).add(plane.w).lessThan(0), () => { inside.assign(uint(0)); });
    });
    return inside.equal(uint(1));
  }

  private buildFinishLevel(): N {
    const counts = this.countsNode();
    return Fn(() => {
      const made = (atomicLoad(counts.element(uint(COUNT_NEXT))) as N).toVar();
      atomicStore(counts.element(uint(COUNT_CANDIDATES)), made.min(uint(this.capacity)));
      atomicStore(counts.element(uint(COUNT_NEXT)), uint(0));
    })();
  }

  private buildPublish(): N {
    const counts = this.countsNode();
    const args = storage(this.drawArgs, 'uint', INDEXED_INDIRECT_WORDS);
    return Fn(() => {
      args.element(uint(INSTANCE_COUNT_WORD)).assign((atomicLoad(counts.element(uint(COUNT_LEAVES))) as N).min(uint(this.capacity)));
    })();
  }

  async bakeBounds(renderer: THREE.WebGPURenderer): Promise<void> {
    await (renderer as N).compileComputeAsync([this.leafBoundsKernel, this.parentBoundsKernel, this.resetKernel, ...this.walkKernels, this.finishLevelKernel, this.publishKernel]);
    renderer.compute(this.leafBoundsKernel, levelSide(this.maxLevel, 0) ** 2);
    for (let level = 1; level <= this.maxLevel; level++) {
      this.boundsLevel.value = level;
      renderer.compute(this.parentBoundsKernel, levelSide(this.maxLevel, level) ** 2);
    }
  }

  select(renderer: THREE.WebGPURenderer, camera: THREE.Camera, toLight?: THREE.Vector3): void {
    this.viewOrigin.value.setFromMatrixPosition(camera.matrixWorld);
    this.viewProjection.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    this.frustum.setFromProjectionMatrix(this.viewProjection, camera.coordinateSystem);
    this.frustum.planes.forEach((plane, at) => {
      const viewPlane = { x: plane.normal.x, y: plane.normal.y, z: plane.normal.z, w: plane.constant };
      const { x, y, z, w } = toLight ? planeForCasters(viewPlane, toLight) : viewPlane;
      (this.planes.array[at] as THREE.Vector4).set(x, y, z, w);
    });
    renderer.compute(this.resetKernel);
    for (let walked = 0; walked <= this.maxLevel; walked++) {
      renderer.compute(this.walkKernels[walked & 1], this.capacity);
      renderer.compute(this.finishLevelKernel);
    }
    renderer.compute(this.publishKernel);
  }

  async leafCount(renderer: THREE.WebGPURenderer): Promise<number> {
    return new Uint32Array(await renderer.getArrayBufferAsync(this.counts))[COUNT_LEAVES];
  }

  dispose(): void {
    for (const kernel of [this.leafBoundsKernel, this.parentBoundsKernel, this.resetKernel, ...this.walkKernels, this.finishLevelKernel, this.publishKernel]) kernel.dispose();
  }
}

function segmentsIndexCount(segments: number): number {
  return segments * segments * 6;
}
