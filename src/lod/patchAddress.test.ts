import { describe, expect, it } from 'vitest';
import { levelBase, levelSide, patchCountOfTree } from './patchAddress.ts';

describe('patch pyramid layout', () => {
  it('a level holds a square of patches, one at the root and doubling on the way down', () => {
    expect(levelSide(4, 4)).toBe(1);
    expect(levelSide(4, 3)).toBe(2);
    expect(levelSide(4, 0)).toBe(16);
  });

  it('levels sit one after another with the finest first and never overlap', () => {
    const maxLevel = 4;
    let end = 0;
    for (let level = 0; level <= maxLevel; level++) {
      expect(levelBase(maxLevel, level)).toBe(end);
      end += levelSide(maxLevel, level) ** 2;
    }
    expect(patchCountOfTree(maxLevel)).toBe(end);
  });
});
