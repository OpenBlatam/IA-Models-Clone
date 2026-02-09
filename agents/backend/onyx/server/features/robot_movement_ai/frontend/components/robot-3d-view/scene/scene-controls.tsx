/**
 * Scene Controls Component
 * 
 * Separated controls configuration for better organization.
 * 
 * @module robot-3d-view/scene/scene-controls
 */

'use client';

import { memo } from 'react';
import {
  PresentationControls,
  OrbitControls,
  GizmoHelper,
  GizmoViewport,
} from '../lib/drei-imports';
import type { SceneConfig } from '../types';

/**
 * Scene Controls Component
 * 
 * Handles all camera and navigation controls for the 3D scene.
 * 
 * @param config - Scene configuration
 * @returns Scene controls component
 */
export const SceneControls = memo(({ config }: { config: SceneConfig }) => {
  return (
    <>
      <PresentationControls
        global
        zoom={0.8}
        rotation={[0, 0, 0]}
        polar={[-Math.PI / 3, Math.PI / 3]}
        azimuth={[-Math.PI / 1.4, Math.PI / 2]}
      >
        <OrbitControls
          enableDamping
          dampingFactor={0.05}
          minDistance={1}
          maxDistance={20}
          enablePan
          enableZoom
          enableRotate
          autoRotate={config.autoRotate}
          autoRotateSpeed={0.5}
        />
      </PresentationControls>

      {config.showGizmo && (
        <GizmoHelper alignment="bottom-right" margin={[80, 80]}>
          <GizmoViewport
            axisColors={['#ef4444', '#10b981', '#3b82f6']}
            labelColor="white"
          />
        </GizmoHelper>
      )}
    </>
  );
});

SceneControls.displayName = 'SceneControls';



