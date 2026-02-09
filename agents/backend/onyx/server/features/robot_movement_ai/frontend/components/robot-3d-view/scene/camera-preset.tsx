/**
 * Camera Preset Component
 * @module robot-3d-view/scene/camera-preset
 */

'use client';

import { useEffect } from 'react';
import { useThree } from '@react-three/fiber';
import type { Position3D, CameraPreset } from '../types';
import { CAMERA_PRESETS } from '../constants';

/**
 * Props for CameraPreset component
 */
interface CameraPresetProps {
  preset: CameraPreset;
  target: Position3D;
}

/**
 * Camera Preset Component
 * 
 * Applies camera position and rotation based on preset configuration.
 * 
 * @param props - Camera preset configuration
 * @returns null (side effect component)
 * 
 * @example
 * ```tsx
 * <CameraPreset preset="front" target={[0, 0, 0]} />
 * ```
 */
export function CameraPreset({ preset, target }: CameraPresetProps) {
  const { camera } = useThree();

  useEffect(() => {
    if (!preset) return;

    const presetConfig = CAMERA_PRESETS[preset];
    if (!presetConfig) return;

    // Adjust target position
    const adjustedPosition: Position3D = [
      presetConfig.position[0] + target[0],
      presetConfig.position[1] + target[1],
      presetConfig.position[2] + target[2],
    ];

    camera.position.set(...adjustedPosition);
    camera.lookAt(...target);
  }, [camera, preset, target]);

  return null;
}



