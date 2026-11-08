import type { Ops } from './ops.ts';

export interface LevelReach {
  readonly fineReach: number;
  readonly growth: number;
}

export type Point3<S> = readonly [S, S, S];

export interface Box3<S> {
  readonly min: Point3<S>;
  readonly max: Point3<S>;
}

export interface PatchRequest<S> {
  readonly level: S | number;
  readonly nearestLevel: S | number;
  readonly densityBoost: S | number;
}

const lift = <S, B>(k: Ops<S, B>, value: S | number): S => (typeof value === 'number' ? k.constant(value) : value);

export function continuousLevel<S, B>(k: Ops<S, B>, distanceSquared: S, reach: { fineReach: S | number; growth: S | number }): S {
  const fine = lift(k, reach.fineReach);
  const stepped = k.divide(k.sqrt(distanceSquared), fine);
  const inFineReach = k.minimum(stepped, k.constant(1));
  const beyond = k.maximum(stepped, k.constant(1));
  return k.add(inFineReach, k.divide(k.log2(beyond), k.log2(lift(k, reach.growth))));
}

export function distanceSquaredToBox<S, B>(k: Ops<S, B>, point: Point3<S | number>, box: Box3<S | number>): S {
  return [0, 1, 2].reduce((sum, axis) => {
    const coordinate = lift(k, point[axis]);
    const below = k.subtract(lift(k, box.min[axis]), coordinate);
    const above = k.subtract(coordinate, lift(k, box.max[axis]));
    const gap = k.maximum(k.maximum(below, above), k.constant(0));
    return k.add(sum, k.multiply(gap, gap));
  }, k.constant(0));
}

export function patchSubdivides<S, B>(k: Ops<S, B>, patch: PatchRequest<S>): B {
  const level = lift(k, patch.level);
  return k.both(k.below(k.constant(0), level), k.below(k.subtract(lift(k, patch.nearestLevel), lift(k, patch.densityBoost)), level));
}

const SNAP_TOLERANCE = 1e-4;

export function snappedLocal<S, B>(k: Ops<S, B>, local: S | number, segments: S | number, levelsAbove: S | number): S {
  const cells = k.maximum(k.divide(lift(k, segments), k.power2(lift(k, levelsAbove))), k.constant(1));
  return k.divide(k.floor(k.add(k.multiply(lift(k, local), cells), k.constant(SNAP_TOLERANCE))), cells);
}
