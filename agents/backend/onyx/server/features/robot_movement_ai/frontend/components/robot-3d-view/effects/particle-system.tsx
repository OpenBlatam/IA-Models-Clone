/**
 * Advanced Particle System Component
 * @module robot-3d-view/effects/particle-system
 */

'use client';

import { useRef, useMemo, memo } from 'react';
import { useFrame } from '@react-three/fiber';
import { Sparkles } from '@react-three/drei';
import * as THREE from 'three';
import type { Position3D } from '../types';

/**
 * Props for ParticleSystem component
 */
interface ParticleSystemProps {
  position: Position3D;
  count?: number;
  color?: string;
  speed?: number;
  size?: number;
  scale?: [number, number, number];
  enabled?: boolean;
}

/**
 * Advanced Particle System
 * 
 * Creates an enhanced particle system with customizable properties.
 * 
 * @param props - Particle system configuration
 * @returns Particle system component
 */
export const ParticleSystem = memo(({
  position,
  count = 20,
  color = '#0ea5e9',
  speed = 0.4,
  size = 2,
  scale = [2, 2, 2],
  enabled = true,
}: ParticleSystemProps) => {
  if (!enabled) return null;

  return (
    <Sparkles
      count={count}
      scale={scale}
      size={size}
      speed={speed}
      opacity={0.6}
      color={color}
      position={position}
    />
  );
});

ParticleSystem.displayName = 'ParticleSystem';

/**
 * Trail Effect Component
 * 
 * Creates a trailing effect behind moving objects.
 */
export const TrailEffect = memo(({ 
  positions, 
  color = '#0ea5e9',
  width = 0.02,
}: { 
  positions: Position3D[];
  color?: string;
  width?: number;
}) => {
  const points = useMemo(() => {
    return positions.map((pos) => new THREE.Vector3(...pos));
  }, [positions]);

  if (points.length < 2) return null;

  return (
    <line>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={points.length}
          array={new Float32Array(points.flatMap((p) => [p.x, p.y, p.z]))}
          itemSize={3}
        />
      </bufferGeometry>
      <lineBasicMaterial color={color} linewidth={width} />
    </line>
  );
});

TrailEffect.displayName = 'TrailEffect';



