/**
 * Industrial Objects - Monitors, Lights, Ventilators
 * @module robot-3d-view/objects/industrial-objects
 */

'use client';

import { useRef, useState, memo } from 'react';
import { useFrame } from '@react-three/fiber';
import { Float, Html } from '@react-three/drei';
import * as THREE from 'three';
import type { BaseObjectProps } from '../types';

/**
 * Industrial Light Component
 */
export const IndustrialLight = memo(({ position, intensity = 1 }: BaseObjectProps & { intensity?: number }) => {
  return (
    <group position={position}>
      {/* Light fixture body */}
      <mesh>
        <boxGeometry args={[0.2, 0.1, 0.2]} />
        <meshStandardMaterial color="#1f2937" metalness={0.7} roughness={0.3} />
      </mesh>
      {/* Light bulb */}
      <mesh position={[0, -0.05, 0]}>
        <sphereGeometry args={[0.06, 16, 16]} />
        <meshStandardMaterial
          color="#fbbf24"
          emissive="#fbbf24"
          emissiveIntensity={intensity}
        />
      </mesh>
      {/* Light cone */}
      <mesh position={[0, -0.15, 0]} rotation={[Math.PI, 0, 0]}>
        <coneGeometry args={[0.15, 0.2, 8]} />
        <meshStandardMaterial
          color="#fbbf24"
          emissive="#fbbf24"
          emissiveIntensity={intensity * 0.3}
          transparent
          opacity={0.3}
          side={THREE.DoubleSide}
        />
      </mesh>
      {/* Point light */}
      <pointLight position={[0, -0.1, 0]} intensity={intensity * 0.5} color="#fbbf24" />
    </group>
  );
});

IndustrialLight.displayName = 'IndustrialLight';

/**
 * Monitor Component
 */
export const Monitor = memo(({ position, rotation, content = 'MONITOR' }: BaseObjectProps & { content?: string }) => {
  const [hovered, setHovered] = useState(false);

  return (
    <Float speed={0.2} rotationIntensity={0.05} floatIntensity={0.05}>
      <group
        position={position}
        rotation={rotation || [0, 0, 0]}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
      >
        {/* Monitor stand */}
        <mesh position={[0, -0.15, 0]}>
          <boxGeometry args={[0.05, 0.1, 0.05]} />
          <meshStandardMaterial color="#374151" metalness={0.6} roughness={0.4} />
        </mesh>
        {/* Monitor base */}
        <mesh position={[0, -0.2, 0]}>
          <boxGeometry args={[0.15, 0.02, 0.15]} />
          <meshStandardMaterial color="#1f2937" metalness={0.7} roughness={0.3} />
        </mesh>
        {/* Monitor body */}
        <mesh position={[0, 0, 0]}>
          <boxGeometry args={[0.3, 0.2, 0.05]} />
          <meshStandardMaterial color="#1f2937" metalness={0.6} roughness={0.4} />
        </mesh>
        {/* Screen */}
        <mesh position={[0, 0, 0.03]}>
          <boxGeometry args={[0.28, 0.18, 0.01]} />
          <meshStandardMaterial
            color={hovered ? '#0ea5e9' : '#0f172a'}
            emissive={hovered ? '#0ea5e9' : '#1e293b'}
            emissiveIntensity={hovered ? 0.8 : 0.3}
          />
        </mesh>
        {/* Screen content */}
        <Html position={[0, 0, 0.04]} center>
          <div className="bg-slate-900/95 backdrop-blur-sm px-2 py-1 rounded text-[8px] text-cyan-400 font-mono whitespace-nowrap">
            {content}
          </div>
        </Html>
      </group>
    </Float>
  );
});

Monitor.displayName = 'Monitor';

/**
 * Ventilator Component
 */
export const Ventilator = memo(({ position }: BaseObjectProps) => {
  const fanRef = useRef<THREE.Group>(null);

  useFrame((state) => {
    if (fanRef.current) {
      fanRef.current.rotation.z = state.clock.elapsedTime * 5;
    }
  });

  return (
    <group position={position}>
      {/* Mount */}
      <mesh position={[0, -0.1, 0]}>
        <cylinderGeometry args={[0.05, 0.05, 0.2, 8]} />
        <meshStandardMaterial color="#4b5563" metalness={0.6} roughness={0.4} />
      </mesh>
      {/* Motor housing */}
      <mesh>
        <cylinderGeometry args={[0.08, 0.08, 0.1, 16]} />
        <meshStandardMaterial color="#1f2937" metalness={0.8} roughness={0.2} />
      </mesh>
      {/* Fan blades */}
      <group ref={fanRef}>
        {[0, 1, 2].map((i) => (
          <mesh key={i} rotation={[0, 0, (i * Math.PI * 2) / 3]}>
            <boxGeometry args={[0.3, 0.05, 0.01]} />
            <meshStandardMaterial color="#6b7280" metalness={0.5} roughness={0.5} />
          </mesh>
        ))}
      </group>
    </group>
  );
});

Ventilator.displayName = 'Ventilator';



