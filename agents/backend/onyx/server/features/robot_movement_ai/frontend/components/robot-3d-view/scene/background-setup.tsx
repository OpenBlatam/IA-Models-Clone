/**
 * Background Setup Component
 * 
 * Separated background/sky configuration for better organization.
 * 
 * @module robot-3d-view/scene/background-setup
 */

'use client';

import { memo } from 'react';
import dynamic from 'next/dynamic';
import { lazyDreiComponents } from '../lib/drei-imports';

// Lazy load heavy background components
const Stars = dynamic(() => lazyDreiComponents.Stars(), { ssr: false });
const Sky = dynamic(() => lazyDreiComponents.Sky(), { ssr: false });

/**
 * Background Setup Component
 * 
 * Handles sky and stars rendering when enabled.
 * 
 * @param showStars - Whether to show stars and sky
 * @returns Background setup component
 */
export const BackgroundSetup = memo(({ showStars }: { showStars: boolean }) => {
  if (!showStars) {
    return null;
  }

  return (
    <>
      <Stars
        radius={100}
        depth={50}
        count={5000}
        factor={4}
        saturation={0}
        fade
        speed={1}
      />
      <Sky
        distance={450000}
        sunPosition={[0, 1, 0]}
        inclination={0}
        azimuth={0.25}
      />
    </>
  );
});

BackgroundSetup.displayName = 'BackgroundSetup';



