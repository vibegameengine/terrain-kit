export function levelSide(maxLevel: number, level: number): number {
  return 2 ** (maxLevel - level);
}

export function levelBase(maxLevel: number, level: number): number {
  let base = 0;
  for (let finer = 0; finer < level; finer++) base += levelSide(maxLevel, finer) ** 2;
  return base;
}

export function patchCountOfTree(maxLevel: number): number {
  return levelBase(maxLevel, maxLevel + 1);
}
