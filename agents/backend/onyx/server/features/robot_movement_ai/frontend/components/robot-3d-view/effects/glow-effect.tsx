/**
 * Glow Effect Component
 * @module robot-3d-view/effects/glow-effect
 */

'use client';

import { useRef, memo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import type { Position3D } from '../types';

/**
 * Props for GlowEffect component
 */
interface GlowEffectProps {
  position: Position3D;
  color?: string;
  intensity?: number;
  size?: number;
  pulseSpeed?: number;
  enabled?: boolean;
}

/**
 * Glow Effect Component
 * 
 * Creates a pulsing glow effect around objects.
 * 
 * @param props - Glow effect configuration
 * @returns Glow effect component
 */
export const GlowEffect = memo(({
  position,
  color = '#0ea5e9',
  intensity = 0.5,
  size = 0.1,
  pulseSpeed = 2,
  enabled = true,
}: GlowEffectProps) => {
  const glowRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (!glowRef.current || !enabled) return;

    const pulse = Math.sin(state.clock.elapsedTime * pulseSpeed) * 0.3 + 0.7;
    glowRef.current.scale.setScalar(size * pulse);
    
    const material = glowRef.current.material as THREE.MeshStandardMaterial;
    if (material) {
      material.emissiveIntensity = intensity * pulse;
    }
  });

  if (!enabled) return null;

  return (
    <mesh ref={glowRef} position={position}>
      <sphereGeometry args={[size, 16, 16]} />
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={intensity}
        transparent
        opacity={0.6}
      />
    </mesh>
  );
});

GlowEffect.displayName = 'GlowEffect';



