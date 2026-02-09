/**
 * Sensor Objects - Camera and Proximity Sensors
 * @module robot-3d-view/objects/sensor-objects
 */

'use client';

import { useRef, memo } from 'react';
import { useFrame } from '@react-three/fiber';
import { Float, Html } from '@react-three/drei';
import * as THREE from 'three';
import type { BaseObjectProps } from '../types';

/**
 * Camera Sensor Component
 */
export const CameraSensor = memo(({ position, rotation }: BaseObjectProps) => {
  const meshRef = useRef<THREE.Group>(null);

  useFrame((state) => {
    if (meshRef.current) {
      // Subtle scanning animation
      meshRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.5) * 0.2;
    }
  });

  return (
    <Float speed={0.5} rotationIntensity={0.1} floatIntensity={0.1}>
      <group ref={meshRef} position={position} rotation={rotation || [0, 0, 0]}>
        {/* Camera body */}
        <mesh>
          <boxGeometry args={[0.08, 0.08, 0.12]} />
          <meshStandardMaterial color="#1f2937" metalness={0.8} roughness={0.2} />
        </mesh>
        {/* Lens */}
        <mesh position={[0, 0, 0.07]}>
          <cylinderGeometry args={[0.04, 0.04, 0.02, 16]} />
          <meshStandardMaterial color="#000000" metalness={0.9} roughness={0.1} />
        </mesh>
        {/* LED indicator */}
        <mesh position={[0.03, 0.03, 0.05]}>
          <sphereGeometry args={[0.01, 8, 8]} />
          <meshStandardMaterial color="#10b981" emissive="#10b981" emissiveIntensity={1} />
        </mesh>
        {/* View cone */}
        <mesh position={[0, 0, 0.15]} rotation={[0, 0, 0]}>
          <coneGeometry args={[0.2, 0.3, 8]} />
          <meshStandardMaterial
            color="#3b82f6"
            transparent
            opacity={0.1}
            side={THREE.DoubleSide}
          />
        </mesh>
      </group>
    </Float>
  );
});

CameraSensor.displayName = 'CameraSensor';

/**
 * Proximity Sensor Component
 */
export const ProximitySensor = memo(({ position }: BaseObjectProps) => {
  const sensorRef = useRef<THREE.Group>(null);

  useFrame((state) => {
    if (sensorRef.current) {
      // Pulsing detection animation
      const scale = 1 + Math.sin(state.clock.elapsedTime * 4) * 0.3;
      sensorRef.current.scale.setScalar(scale);
    }
  });

  return (
    <group ref={sensorRef} position={position}>
      {/* Sensor body */}
      <mesh>
        <cylinderGeometry args={[0.03, 0.03, 0.05, 8]} />
        <meshStandardMaterial color="#6366f1" metalness={0.7} roughness={0.3} />
      </mesh>
      {/* Detection ring */}
      <mesh position={[0, 0, 0.03]}>
        <ringGeometry args={[0.05, 0.08, 16]} />
        <meshStandardMaterial
          color="#6366f1"
          emissive="#6366f1"
          emissiveIntensity={0.8}
          transparent
          opacity={0.6}
        />
      </mesh>
    </group>
  );
});

ProximitySensor.displayName = 'ProximitySensor';



