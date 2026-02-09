/**
 * Lighting Setup Component
 * 
 * Separated lighting configuration for better organization and reusability.
 * 
 * @module robot-3d-view/scene/lighting-setup
 */

'use client';

import { memo } from 'react';
import { LIGHTING } from '../constants';

/**
 * Lighting Setup Component
 * 
 * Configures all lights for the 3D scene.
 * Separated for better organization and potential reuse.
 * 
 * @returns Lighting setup component
 */
export const LightingSetup = memo(() => {
  return (
    <>
      <ambientLight intensity={LIGHTING.ambient.intensity} />
      
      <directionalLight
        position={LIGHTING.directional.main.position}
        intensity={LIGHTING.directional.main.intensity}
        castShadow
        shadow-mapSize-width={2048}
        shadow-mapSize-height={2048}
      />
      
      <directionalLight
        position={LIGHTING.directional.secondary.position}
        intensity={LIGHTING.directional.secondary.intensity}
      />
      
      <pointLight 
        position={LIGHTING.point.position} 
        intensity={LIGHTING.point.intensity} 
      />
      
      <spotLight
        position={LIGHTING.spot.position}
        angle={LIGHTING.spot.angle}
        penumbra={LIGHTING.spot.penumbra}
        intensity={LIGHTING.spot.intensity}
        castShadow
      />
    </>
  );
});

LightingSetup.displayName = 'LightingSetup';



