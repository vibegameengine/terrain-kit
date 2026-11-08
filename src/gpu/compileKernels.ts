import type * as THREE from 'three/webgpu';
import type { N } from './tslOps.ts';

export async function compileKernels(renderer: THREE.WebGPURenderer, kernels: readonly N[]): Promise<void> {
  await (renderer as N).compileComputeAsync?.(kernels);
}
