/**
 * Environment Setup Component
 * @module robot-3d-view/scene/environment-setup
 */

'use client';

import { memo } from 'react';
import dynamic from 'next/dynamic';
import {
  WorkTable,
  ToolContainer,
  SafetyZone,
  CoordinateMarkers,
  StatusLights,
  BoundaryMarkers,
} from '../objects/environment-objects';
import {
  CameraSensor,
  ProximitySensor,
} from '../objects/sensor-objects';
import type { Position3D } from '../types';

// Lazy load heavy components
const Monitor = dynamic(
  () => import('../objects/industrial-objects').then((mod) => ({ default: mod.Monitor })),
  { ssr: false }
);
const IndustrialLight = dynamic(
  () => import('../objects/industrial-objects').then((mod) => ({ default: mod.IndustrialLight })),
  { ssr: false }
);
const Ventilator = dynamic(
  () => import('../objects/industrial-objects').then((mod) => ({ default: mod.Ventilator })),
  { ssr: false }
);

/**
 * Props for EnvironmentSetup component
 */
interface EnvironmentSetupProps {
  showObjects: boolean;
  currentPos: Position3D;
}

/**
 * Environment Setup Component
 * 
 * Renders all environment objects (tables, sensors, lights, etc.)
 * when showObjects is enabled.
 * 
 * @param props - Environment configuration
 * @returns Environment objects component
 */
export const EnvironmentSetup = memo(({ showObjects, currentPos }: EnvironmentSetupProps) => {
  if (!showObjects) {
    return null;
  }

  return (
    <>
      {/* Work table */}
      <WorkTable />

      {/* Tool containers */}
      <ToolContainer position={[-1.5, 0, -1.5]} />
      <ToolContainer position={[1.5, 0, -1.5]} />

      {/* Safety zone */}
      <SafetyZone radius={2.5} />

      {/* Coordinate markers */}
      <CoordinateMarkers />

      {/* Status indicator lights */}
      <StatusLights position={[-1.8, 0.3, -1.8]} status="idle" />
      <StatusLights position={[1.8, 0.3, -1.8]} status="moving" />

      {/* Boundary markers */}
      <BoundaryMarkers />

      {/* Camera sensors */}
      <CameraSensor position={[-1.5, 0.5, 1.5]} rotation={[0, Math.PI / 4, 0]} />
      <CameraSensor position={[1.5, 0.5, -1.5]} rotation={[0, -Math.PI / 4, 0]} />

      {/* Proximity sensors */}
      <ProximitySensor position={[-1, 0.2, 0]} />
      <ProximitySensor position={[1, 0.2, 0]} />
      <ProximitySensor position={[0, 0.2, -1]} />

      {/* Industrial lights */}
      <IndustrialLight position={[-1, 1.2, -1]} intensity={1.2} />
      <IndustrialLight position={[1, 1.2, 1]} intensity={1.2} />
      <IndustrialLight position={[0, 1.2, 0]} intensity={0.8} />

      {/* Ventilators */}
      <Ventilator position={[-1.5, 1.5, -1.5]} />
      <Ventilator position={[1.5, 1.5, 1.5]} />

      {/* Monitors */}
      <Monitor position={[1.8, 0.3, 0]} rotation={[0, -Math.PI / 4, 0]} content="STATUS" />
      <Monitor position={[-1.8, 0.3, 0]} rotation={[0, Math.PI / 4, 0]} content="DATA" />
    </>
  );
});

EnvironmentSetup.displayName = 'EnvironmentSetup';



