import { describe, expect, it } from 'vitest';
import { planeForCasters, type Plane } from './castingPlanes.ts';

const toLight = { x: 0, y: 1, z: 0 };
const insideOf = (plane: Plane, point: { x: number; y: number; z: number }) => plane.x * point.x + plane.y * point.y + plane.z * point.z + plane.w >= 0;

describe('a view plane used to cull shadow casters', () => {
  it('still rejects a caster when its shadow can only move further outside', () => {
    const facingTheLight = { x: 0, y: 1, z: 0, w: 0 };
    expect(insideOf(planeForCasters(facingTheLight, toLight), { x: 0, y: -1, z: 0 })).toBe(false);
  });

  it('accepts everything when the shadow of an outside caster can fall back inside', () => {
    const facingAwayFromTheLight = { x: 0, y: -1, z: 0, w: 0 };
    expect(insideOf(planeForCasters(facingAwayFromTheLight, toLight), { x: 0, y: 50, z: 0 })).toBe(true);
  });

  it('keeps a plane parallel to the light, which the shadow can never cross', () => {
    const sidePlane = { x: 1, y: 0, z: 0, w: 0 };
    expect(insideOf(planeForCasters(sidePlane, toLight), { x: -1, y: 0, z: 0 })).toBe(false);
  });
});
