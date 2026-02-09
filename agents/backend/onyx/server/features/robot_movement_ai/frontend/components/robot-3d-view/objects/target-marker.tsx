/**
 * Target Marker 3D Component
 * @module robot-3d-view/objects/target-marker
 */

'use client';

import { useRef, memo } from 'react';
import { useFrame } from '@react-three/fiber';
import { Float, Html } from '@react-three/drei';
import * as THREE from 'three';
import type { TargetMarkerProps } from '../types';
import { MATERIALS, ANIMATIONS } from '../constants';

/**
 * Target position marker with animation
 * 
 * Displays an animated marker at the target position with pulsing animation
 * and optional label.
 * 
 * @param props - Target marker component props
 * @returns Target marker 3D component
 * 
 * @example
 * ```tsx
 * <TargetMarker position={[1, 1, 1]} label="Objetivo" />
 * ```
 */
export const TargetMarker = memo(({ position, label = 'Objetivo' }: TargetMarkerProps) => {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (meshRef.current) {
      // Pulsing animation
      const scale = 1 + Math.sin(state.clock.elapsedTime * 3) * 0.2;
      meshRef.current.scale.setScalar(scale);
    }
  });

  return (
    <Float
      speed={ANIMATIONS.targetMarker.speed}
      rotationIntensity={ANIMATIONS.targetMarker.rotationIntensity}
      floatIntensity={ANIMATIONS.targetMarker.floatIntensity}
    >
      <group position={position}>
        <mesh ref={meshRef}>
          <sphereGeometry args={[0.05, 16, 16]} />
          <meshStandardMaterial
            color={MATERIALS.target.color}
            emissive={MATERIALS.target.emissive}
            emissiveIntensity={MATERIALS.target.emissiveIntensity}
          />
        </mesh>
        {/* Ring indicator */}
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <ringGeometry args={[0.08, 0.12, 32]} />
          <meshStandardMaterial
            color={MATERIALS.target.color}
            emissive={MATERIALS.target.emissive}
            emissiveIntensity={0.5}
            transparent
            opacity={0.6}
          />
        </mesh>
        {/* Label */}
        {label && (
          <Html position={[0, 0.2, 0]} center>
            <div className="bg-yellow-500/90 backdrop-blur-sm px-2 py-1 rounded text-xs font-semibold text-black whitespace-nowrap">
              {label}
            </div>
          </Html>
        )}
      </group>
    </Float>
  );
});

TargetMarker.displayName = 'TargetMarker';



