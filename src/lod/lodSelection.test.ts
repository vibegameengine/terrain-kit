import { describe, expect, it } from 'vitest';
import { numberOps } from './ops.ts';
import { continuousLevel, distanceSquaredToBox, patchSubdivides, snappedLocal, type LevelReach } from './lodSelection.ts';

const reach: LevelReach = { fineReach: 40, growth: 2 };
const levelAt = (distance: number) => continuousLevel(numberOps, distance * distance, reach);

describe('continuous level', () => {
  it('is zero at the viewer and rises to one at the end of the fine reach', () => {
    expect(levelAt(0)).toBe(0);
    expect(levelAt(20)).toBeCloseTo(0.5);
    expect(levelAt(40)).toBeCloseTo(1);
  });

  it('gains one level each time the distance multiplies by the growth', () => {
    expect(levelAt(80)).toBeCloseTo(2);
    expect(levelAt(160)).toBeCloseTo(3);
  });

  it('never decreases with distance, so a patch boundary cannot flip back', () => {
    let previous = -1;
    for (let distance = 0; distance < 2000; distance += 3) {
      expect(levelAt(distance)).toBeGreaterThanOrEqual(previous);
      previous = levelAt(distance);
    }
  });
});

describe('distance to a box', () => {
  const box = { min: [0, 0, 0] as const, max: [10, 2, 10] as const };

  it('is zero inside', () => {
    expect(distanceSquaredToBox(numberOps, [5, 1, 5], box)).toBe(0);
  });

  it('measures to the nearest face, edge or corner', () => {
    expect(distanceSquaredToBox(numberOps, [15, 1, 5], box)).toBe(25);
    expect(distanceSquaredToBox(numberOps, [13, 6, 14], box)).toBe(9 + 16 + 16);
  });
});

describe('patch subdivision', () => {
  it('subdivides while the nearest point wants a finer level than the patch', () => {
    expect(patchSubdivides(numberOps, { level: 3, nearestLevel: 2.5, densityBoost: 0 })).toBe(true);
    expect(patchSubdivides(numberOps, { level: 3, nearestLevel: 3, densityBoost: 0 })).toBe(false);
  });

  it('subdivides further where the density map boosts the ground, at the same distance', () => {
    expect(patchSubdivides(numberOps, { level: 3, nearestLevel: 4, densityBoost: 0 })).toBe(false);
    expect(patchSubdivides(numberOps, { level: 3, nearestLevel: 4, densityBoost: 2 })).toBe(true);
  });

  it('never subdivides the finest level', () => {
    expect(patchSubdivides(numberOps, { level: 0, nearestLevel: 0, densityBoost: 5 })).toBe(false);
  });
});

describe('vertex snapping', () => {
  const segments = 16;

  it('leaves a vertex alone when the level it wants is the patch level', () => {
    for (let i = 0; i <= segments; i++) expect(snappedLocal(numberOps, i / segments, segments, 0)).toBeCloseTo(i / segments);
  });

  it('moves a vertex onto the grid of the coarser level', () => {
    expect(snappedLocal(numberOps, 3 / 16, 16, 1)).toBeCloseTo(2 / 16);
    expect(snappedLocal(numberOps, 7 / 16, 16, 2)).toBeCloseTo(4 / 16);
    expect(snappedLocal(numberOps, 1, 16, 2)).toBeCloseTo(1);
  });

  it('closes the seam: on the edge between a fine and a coarse patch every vertex lands on a vertex of the coarse one', () => {
    const patchSize = (level: number) => 2 ** level;
    const worldLevelField = (x: number) => 3 + x / 40;
    const edgeLevel = 3;
    const coarseOrigin = 0;
    const coarseVertices = new Set<string>();
    for (let i = 0; i <= segments; i++) coarseVertices.add((coarseOrigin + (i / segments) * patchSize(edgeLevel + 1)).toFixed(6));
    for (let i = 0; i <= segments; i++) {
      const world = coarseOrigin + (i / segments) * patchSize(edgeLevel);
      const wanted = Math.max(edgeLevel + 1, worldLevelField(world));
      const levelsAbove = Math.floor(wanted) - edgeLevel;
      const local = snappedLocal(numberOps, i / segments, segments, levelsAbove);
      expect(coarseVertices.has((coarseOrigin + local * patchSize(edgeLevel)).toFixed(6))).toBe(true);
    }
  });
});
