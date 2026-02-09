/**
 * Environment Objects - Modular 3D Objects
 * @module robot-3d-view/objects/environment-objects
 */

'use client';

import { memo } from 'react';
import { Float, Html, Sparkles } from '@react-three/drei';
import * as THREE from 'three';
import type { BaseObjectProps } from '../types';

/**
 * Work Table Component
 * 
 * Industrial work table with legs and surface.
 */
export const WorkTable = memo(() => {
  return (
    <group position={[0, -0.15, 0]}>
      {/* Table top */}
      <mesh position={[0, 0, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <boxGeometry args={[4, 4, 0.05]} />
        <meshStandardMaterial color="#4b5563" metalness={0.3} roughness={0.7} />
      </mesh>
      {/* Table legs */}
      {[[-1.8, -1.8], [1.8, -1.8], [-1.8, 1.8], [1.8, 1.8]].map(([x, z], i) => (
        <mesh key={i} position={[x, -0.1, z]}>
          <boxGeometry args={[0.1, 0.2, 0.1]} />
          <meshStandardMaterial color="#374151" metalness={0.5} roughness={0.5} />
        </mesh>
      ))}
    </group>
  );
});

WorkTable.displayName = 'WorkTable';

/**
 * Tool Container Component
 */
export const ToolContainer = memo(({ position }: BaseObjectProps) => {
  return (
    <Float speed={0.5} rotationIntensity={0.1} floatIntensity={0.1}>
      <group position={position}>
        {/* Container box */}
        <mesh>
          <boxGeometry args={[0.3, 0.2, 0.3]} />
          <meshStandardMaterial color="#6b7280" metalness={0.4} roughness={0.6} />
        </mesh>
        {/* Lid */}
        <mesh position={[0, 0.12, 0]}>
          <boxGeometry args={[0.32, 0.02, 0.32]} />
          <meshStandardMaterial color="#9ca3af" metalness={0.5} roughness={0.5} />
        </mesh>
        {/* Label */}
        <Html position={[0, 0.3, 0]} center>
          <div className="bg-gray-700/90 backdrop-blur-sm px-2 py-1 rounded text-[10px] font-semibold text-white whitespace-nowrap">
            Herramientas
          </div>
        </Html>
      </group>
    </Float>
  );
});

ToolContainer.displayName = 'ToolContainer';

/**
 * Safety Zone Component
 */
export const SafetyZone = memo(({ radius = 2.5 }: { radius?: number }) => {
  return (
    <group>
      {/* Safety zone ring */}
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[radius - 0.1, radius, 64]} />
        <meshStandardMaterial
          color="#ef4444"
          emissive="#ef4444"
          emissiveIntensity={0.3}
          transparent
          opacity={0.4}
        />
      </mesh>
      {/* Safety zone posts */}
      {Array.from({ length: 8 }).map((_, i) => {
        const angle = (i / 8) * Math.PI * 2;
        const x = Math.cos(angle) * radius;
        const z = Math.sin(angle) * radius;
        return (
          <mesh key={i} position={[x, 0.15, z]}>
            <cylinderGeometry args={[0.02, 0.02, 0.3, 8]} />
            <meshStandardMaterial color="#ef4444" emissive="#ef4444" emissiveIntensity={0.5} />
          </mesh>
        );
      })}
    </group>
  );
});

SafetyZone.displayName = 'SafetyZone';

/**
 * Coordinate Markers Component
 */
export const CoordinateMarkers = memo(() => {
  const markerSize = 0.1;
  return (
    <group>
      {/* X axis marker */}
      <group position={[2, 0, 0]}>
        <mesh>
          <boxGeometry args={[markerSize, 0.02, 0.02]} />
          <meshStandardMaterial color="#ef4444" emissive="#ef4444" emissiveIntensity={0.5} />
        </mesh>
        <Html position={[0, 0.15, 0]} center>
          <div className="text-red-400 text-xs font-bold">X</div>
        </Html>
      </group>
      {/* Y axis marker */}
      <group position={[0, 2, 0]}>
        <mesh>
          <boxGeometry args={[0.02, markerSize, 0.02]} />
          <meshStandardMaterial color="#10b981" emissive="#10b981" emissiveIntensity={0.5} />
        </mesh>
        <Html position={[0, 0.15, 0]} center>
          <div className="text-green-400 text-xs font-bold">Y</div>
        </Html>
      </group>
      {/* Z axis marker */}
      <group position={[0, 0, 2]}>
        <mesh>
          <boxGeometry args={[0.02, 0.02, markerSize]} />
          <meshStandardMaterial color="#3b82f6" emissive="#3b82f6" emissiveIntensity={0.5} />
        </mesh>
        <Html position={[0, 0.15, 0]} center>
          <div className="text-blue-400 text-xs font-bold">Z</div>
        </Html>
      </group>
    </group>
  );
});

CoordinateMarkers.displayName = 'CoordinateMarkers';

/**
 * Status Lights Component
 */
export const StatusLights = memo(({ position, status = 'idle' }: BaseObjectProps & { status?: 'idle' | 'moving' | 'error' }) => {
  const colors = {
    idle: '#3b82f6',
    moving: '#10b981',
    error: '#ef4444',
  };

  const color = colors[status];

  return (
    <group position={position}>
      <mesh>
        <sphereGeometry args={[0.05, 16, 16]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={status === 'moving' ? 1.5 : 0.8}
        />
      </mesh>
      {/* Glow effect */}
      <mesh>
        <sphereGeometry args={[0.08, 16, 16]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.3}
          transparent
          opacity={0.5}
        />
      </mesh>
    </group>
  );
});

StatusLights.displayName = 'StatusLights';

/**
 * Boundary Markers Component
 */
export const BoundaryMarkers = memo(() => {
  const positions: [number, number, number][] = [
    [-2, 0, -2],
    [2, 0, -2],
    [-2, 0, 2],
    [2, 0, 2],
  ];

  return (
    <group>
      {positions.map((pos, i) => (
        <group key={i} position={pos}>
          <mesh>
            <cylinderGeometry args={[0.05, 0.05, 0.3, 8]} />
            <meshStandardMaterial color="#6b7280" metalness={0.5} roughness={0.5} />
          </mesh>
          <mesh position={[0, 0.2, 0]}>
            <coneGeometry args={[0.08, 0.1, 8]} />
            <meshStandardMaterial color="#ef4444" emissive="#ef4444" emissiveIntensity={0.5} />
          </mesh>
        </group>
      ))}
    </group>
  );
});

BoundaryMarkers.displayName = 'BoundaryMarkers';



