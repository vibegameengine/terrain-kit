import { float, floor, log2, max, min, pow, sqrt } from 'three/tsl';
import type { Ops } from '../lod/ops.ts';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type N = any;

export const tslOps: Ops<N, N> = {
  constant: (value) => float(value),
  add: (a, b) => a.add(b),
  subtract: (a, b) => a.sub(b),
  multiply: (a, b) => a.mul(b),
  divide: (a, b) => a.div(b),
  minimum: (a, b) => min(a, b),
  maximum: (a, b) => max(a, b),
  floor: (a) => floor(a),
  sqrt: (a) => sqrt(a),
  log2: (a) => log2(a),
  power2: (a) => pow(float(2), a),
  below: (a, b) => a.lessThan(b),
  both: (a, b) => a.and(b),
};
