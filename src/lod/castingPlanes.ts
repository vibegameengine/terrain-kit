export interface Plane {
  readonly x: number;
  readonly y: number;
  readonly z: number;
  readonly w: number;
}

const ACCEPTS_EVERYTHING: Plane = { x: 0, y: 0, z: 0, w: 1 };

export function planeForCasters(plane: Plane, toLight: { readonly x: number; readonly y: number; readonly z: number }): Plane {
  const shadowEntersFromOutside = plane.x * toLight.x + plane.y * toLight.y + plane.z * toLight.z < 0;
  return shadowEntersFromOutside ? ACCEPTS_EVERYTHING : plane;
}
