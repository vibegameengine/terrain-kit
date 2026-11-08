import * as THREE from 'three/webgpu';
import { Fn, clamp, float, instanceIndex, int, ivec2, mix, normalize, storage, textureLoad, textureStore, texture, uniform, uint, vec2, vec3, vec4 } from 'three/tsl';
import { compileKernels } from '../gpu/compileKernels.ts';
import type { N } from '../gpu/tslOps.ts';

export interface FieldLattice {
  readonly columns: number;
  readonly rows: number;
  readonly span: { readonly minX: number; readonly minZ: number; readonly width: number; readonly depth: number };
}

export interface TerrainDomain {
  readonly minX: number;
  readonly minZ: number;
  readonly size: number;
}

export interface FieldSample {
  readonly height: N;
  readonly density: N;
  readonly masks: readonly N[];
}

export type FieldFunction = (x: N, z: N) => FieldSample;

export interface TerrainFieldOptions {
  readonly domain: TerrainDomain;
  readonly resolution: number;
  readonly maskTextures: number;
  readonly field: FieldFunction;
}

function storageTexture(side: number, format: THREE.PixelFormat, type: THREE.TextureDataType, filter: THREE.MagnificationTextureFilter, name: string): THREE.StorageTexture {
  const made = new THREE.StorageTexture(side, side);
  made.format = format;
  made.type = type;
  made.minFilter = filter;
  made.magFilter = filter;
  made.wrapS = THREE.ClampToEdgeWrapping;
  made.wrapT = THREE.ClampToEdgeWrapping;
  made.generateMipmaps = false;
  made.name = name;
  return made;
}

export class TerrainField {
  readonly domain: TerrainDomain;
  readonly resolution: number;
  readonly height: THREE.StorageTexture;
  readonly slope: THREE.StorageTexture;
  readonly density: THREE.StorageTexture;
  readonly masks: readonly THREE.StorageTexture[];
  private readonly bakeKernel: N;
  private readonly slopeKernel: N;
  private readonly side = uniform(1, 'uint');

  constructor(options: TerrainFieldOptions) {
    this.domain = options.domain;
    this.resolution = options.resolution;
    this.side.value = options.resolution;
    const side = options.resolution;
    this.height = storageTexture(side, THREE.RedFormat, THREE.FloatType, THREE.NearestFilter, 'TerrainHeight');
    this.slope = storageTexture(side, THREE.RGBAFormat, THREE.HalfFloatType, THREE.LinearFilter, 'TerrainSlope');
    this.density = storageTexture(side, THREE.RGBAFormat, THREE.HalfFloatType, THREE.LinearFilter, 'TerrainDensity');
    this.masks = Array.from({ length: options.maskTextures }, (_, at) => storageTexture(side, THREE.RGBAFormat, THREE.UnsignedByteType, THREE.LinearFilter, `TerrainMask${at}`));
    this.bakeKernel = this.buildBake(options.field).compute(side * side, [64]).setName('Terrain.Bake');
    this.slopeKernel = this.buildSlope().compute(side * side, [64]).setName('Terrain.Slope');
  }

  private texelOf(index: N): N {
    return ivec2(int(index.mod(this.side)), int(index.div(this.side)));
  }

  private buildBake(field: FieldFunction): N {
    const { minX, minZ, size } = this.domain;
    return Fn(() => {
      const texel = this.texelOf(instanceIndex).toVar();
      const x = float(texel.x).add(0.5).div(float(this.side)).mul(size).add(minX).toVar();
      const z = float(texel.y).add(0.5).div(float(this.side)).mul(size).add(minZ).toVar();
      const sample = field(x, z);
      textureStore(this.height, texel, vec4(sample.height, 0, 0, 1));
      textureStore(this.density, texel, vec4(sample.density, 0, 0, 1));
      sample.masks.forEach((mask, at) => textureStore(this.masks[at], texel, mask));
    })();
  }

  private buildSlope(): N {
    const metresPerTexel = this.domain.size / this.resolution;
    return Fn(() => {
      const texel = this.texelOf(instanceIndex).toVar();
      const last = int(this.side).sub(1);
      const heightAt = (dx: number, dy: number) => textureLoad(this.height, (texel.add(ivec2(dx, dy)) as N).clamp(ivec2(0), ivec2(last)), int(0)).r;
      const dhdx = heightAt(1, 0).sub(heightAt(-1, 0)).div(metresPerTexel * 2);
      const dhdz = heightAt(0, 1).sub(heightAt(0, -1)).div(metresPerTexel * 2);
      textureStore(this.slope, texel, vec4(dhdx, dhdz, 0, 1));
    })();
  }

  async bake(renderer: THREE.WebGPURenderer): Promise<void> {
    await compileKernels(renderer, [this.bakeKernel, this.slopeKernel]);
    const texels = this.resolution * this.resolution;
    renderer.compute(this.bakeKernel, texels);
    renderer.compute(this.slopeKernel, texels);
  }

  dispose(): void {
    this.bakeKernel.dispose();
    this.slopeKernel.dispose();
    for (const made of [this.height, this.slope, this.density, ...this.masks]) made.dispose();
  }

  private uvOf(xz: N): N {
    return xz.sub(vec2(this.domain.minX, this.domain.minZ)).div(this.domain.size);
  }

  heightAt(xz: N): N {
    const last = float(this.resolution - 1);
    const cell = clamp(this.uvOf(xz).mul(float(this.resolution)).sub(0.5), vec2(0), vec2(last)).toVar();
    const base = cell.floor().toVar();
    const blend = cell.sub(base).toVar();
    const load = (dx: number, dy: number) => textureLoad(this.height, ivec2((base.add(vec2(dx, dy)) as N).clamp(vec2(0), vec2(last))), int(0)).r;
    return mix(mix(load(0, 0), load(1, 0), blend.x), mix(load(0, 1), load(1, 1), blend.x), blend.y);
  }

  normalAt(xz: N): N {
    const slope = (texture(this.slope, this.uvOf(xz)) as N).level(0).rg.toVar();
    return normalize(vec3(slope.x.negate(), 1, slope.y.negate()));
  }

  densityAt(xz: N): N {
    return (texture(this.density, this.uvOf(xz)) as N).level(0).r;
  }

  maskAt(index: number, xz: N): N {
    return texture(this.masks[index], this.uvOf(xz));
  }

  readHeights(renderer: THREE.WebGPURenderer, lattice: FieldLattice): Promise<Float32Array> {
    return this.readLattice(renderer, lattice, (xz) => this.heightAt(xz));
  }

  readMask(renderer: THREE.WebGPURenderer, index: number, channel: number, lattice: FieldLattice): Promise<Float32Array> {
    return this.readLattice(renderer, lattice, (xz) => (texture(this.masks[index], this.uvOf(xz)) as N).level(0).element(int(channel)));
  }

  private async readLattice(renderer: THREE.WebGPURenderer, { columns, rows, span: spans }: FieldLattice, valueAt: (xz: N) => N): Promise<Float32Array> {
    const output = new THREE.StorageBufferAttribute(new Float32Array(columns * rows), 1);
    const target = storage(output, 'float', columns * rows);
    const across = uint(columns);
    const kernel = Fn(() => {
      const i = instanceIndex;
      const column = float(i.mod(across)).div(float(Math.max(1, columns - 1)));
      const row = float(i.div(across)).div(float(Math.max(1, rows - 1)));
      const x = column.mul(spans.width).add(spans.minX);
      const z = row.mul(spans.depth).add(spans.minZ);
      target.element(i).assign(valueAt(vec2(x, z)));
    })().compute(columns * rows, [64]).setName('Terrain.ReadHeights');
    renderer.compute(kernel, columns * rows);
    const bytes = await renderer.getArrayBufferAsync(output);
    kernel.dispose();
    return new Float32Array(bytes.slice(0));
  }
}
