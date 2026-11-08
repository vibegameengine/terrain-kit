import { Fragment, createElement, useEffect, useRef, useState, type ReactElement } from 'react';
import { useFrame, useThree } from '@vibegameengine/react-three-fiber';
import type * as THREE from 'three/webgpu';
import { createTerrain, type Terrain, type TerrainOptions } from './terrain.ts';

const AFTER_THE_CAMERA_MOVED = 1;

export function TerrainMesh({ terrain }: { terrain: Terrain }): ReactElement {
  useFrame(({ camera }) => { terrain.select(camera as unknown as THREE.Camera); }, AFTER_THE_CAMERA_MOVED);
  return createElement(Fragment, null,
    createElement('primitive', { object: terrain.surface, dispose: null }),
    createElement('primitive', { object: terrain.proxy, dispose: null }));
}

export interface TerrainProps extends TerrainOptions {
  readonly onReady?: (terrain: Terrain) => void;
}

export function GpuTerrain(props: TerrainProps): ReactElement | null {
  const renderer = useThree((state) => state.gl) as unknown as THREE.WebGPURenderer;
  const latest = useRef(props);
  latest.current = props;
  const [terrain, setTerrain] = useState<Terrain | null>(null);
  const { field } = props;

  useEffect(() => {
    let mounted = true;
    let built: Terrain | null = null;
    void createTerrain(renderer, latest.current).then((made) => {
      if (!mounted) { made.dispose(); return; }
      built = made;
      setTerrain(made);
      latest.current.onReady?.(made);
    });
    return () => {
      mounted = false;
      setTerrain(null);
      built?.dispose();
    };
  }, [renderer, field]);

  return terrain ? createElement(TerrainMesh, { terrain }) : null;
}
