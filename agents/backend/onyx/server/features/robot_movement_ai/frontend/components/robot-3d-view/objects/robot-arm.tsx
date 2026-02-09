/**
 * Robot Arm 3D Component
 * @module robot-3d-view/objects/robot-arm
 */

'use client';

import { useRef, useState, memo } from 'react';
import { useFrame } from '@react-three/fiber';
import { Float } from '@react-three/drei';
import * as THREE from 'three';
import type { RobotArmProps } from '../types';
import { MATERIALS, ANIMATIONS } from '../constants';

/**
 * Advanced Robot Arm with better materials and animations
 * 
 * @param props - Robot arm component props
 * @returns Robot arm 3D component
 * 
 * @example
 * ```tsx
 * <RobotArm position={[0, 0, 0]} />
 * ```
 */
export const RobotArm = memo(({ position, hovered: externalHovered }: RobotArmProps) => {
  const meshRef = useRef<THREE.Group>(null);
  const [internalHovered, setInternalHovered] = useState(false);
  const hovered = externalHovered ?? internalHovered;

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.position.set(...position);
      // Subtle rotation animation
      meshRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.5) * 0.1;
    }
  });

  return (
    <Float
      speed={ANIMATIONS.robotArm.speed}
      rotationIntensity={ANIMATIONS.robotArm.rotationIntensity}
      floatIntensity={ANIMATIONS.robotArm.floatIntensity}
    >
      <group
        ref={meshRef}
        position={position}
        onPointerOver={() => setInternalHovered(true)}
        onPointerOut={() => setInternalHovered(false)}
      >
        {/* Base */}
        <mesh position={[0, 0.05, 0]}>
          <boxGeometry args={[0.2, 0.1, 0.2]} />
          <meshStandardMaterial
            color={hovered ? '#38bdf8' : MATERIALS.robot.base.color}
            metalness={MATERIALS.robot.base.metalness}
            roughness={MATERIALS.robot.base.roughness}
            envMapIntensity={1}
          />
        </mesh>

        {/* Link 1 */}
        <mesh position={[0, 0.25, 0]}>
          <boxGeometry args={[0.05, 0.3, 0.05]} />
          <meshStandardMaterial
            color={MATERIALS.robot.link.color}
            metalness={MATERIALS.robot.link.metalness}
            roughness={MATERIALS.robot.link.roughness}
          />
        </mesh>

        {/* Joint 1 */}
        <mesh position={[0, 0.4, 0]}>
          <sphereGeometry args={[0.06, 16, 16]} />
          <meshStandardMaterial
            color={MATERIALS.robot.joint.color}
            metalness={MATERIALS.robot.joint.metalness}
            roughness={MATERIALS.robot.joint.roughness}
          />
        </mesh>

        {/* Link 2 */}
        <mesh position={[0.15, 0.4, 0]}>
          <boxGeometry args={[0.3, 0.05, 0.05]} />
          <meshStandardMaterial
            color={MATERIALS.robot.link.color}
            metalness={MATERIALS.robot.link.metalness}
            roughness={MATERIALS.robot.link.roughness}
          />
        </mesh>

        {/* Joint 2 */}
        <mesh position={[0.3, 0.4, 0]}>
          <sphereGeometry args={[0.06, 16, 16]} />
          <meshStandardMaterial
            color={MATERIALS.robot.joint.color}
            metalness={MATERIALS.robot.joint.metalness}
            roughness={MATERIALS.robot.joint.roughness}
          />
        </mesh>

        {/* Link 3 - Additional segment */}
        <mesh position={[0.3, 0.4, 0.1]}>
          <boxGeometry args={[0.05, 0.05, 0.15]} />
          <meshStandardMaterial
            color={MATERIALS.robot.link.color}
            metalness={MATERIALS.robot.link.metalness}
            roughness={MATERIALS.robot.link.roughness}
          />
        </mesh>

        {/* Joint 3 */}
        <mesh position={[0.3, 0.4, 0.2]}>
          <sphereGeometry args={[0.05, 16, 16]} />
          <meshStandardMaterial
            color={MATERIALS.robot.joint.color}
            metalness={MATERIALS.robot.joint.metalness}
            roughness={MATERIALS.robot.joint.roughness}
          />
        </mesh>

        {/* End Effector with glow effect */}
        <mesh position={[0.3, 0.4, 0.25]}>
          <boxGeometry args={[0.08, 0.08, 0.08]} />
          <meshStandardMaterial
            color={MATERIALS.robot.effector.color}
            emissive={MATERIALS.robot.effector.color}
            emissiveIntensity={hovered ? 0.8 : 0.5}
            metalness={MATERIALS.robot.effector.metalness}
            roughness={MATERIALS.robot.effector.roughness}
          />
        </mesh>

        {/* Gripper fingers */}
        <mesh position={[0.3, 0.35, 0.25]}>
          <boxGeometry args={[0.02, 0.06, 0.04]} />
          <meshStandardMaterial
            color="#059669"
            metalness={0.6}
            roughness={0.3}
          />
        </mesh>
        <mesh position={[0.3, 0.45, 0.25]}>
          <boxGeometry args={[0.02, 0.06, 0.04]} />
          <meshStandardMaterial
            color="#059669"
            metalness={0.6}
            roughness={0.3}
          />
        </mesh>

        {/* Cable/wire simulation */}
        <mesh position={[0, 0.1, 0]}>
          <cylinderGeometry args={[0.01, 0.01, 0.3, 8]} />
          <meshStandardMaterial
            color="#1f2937"
            metalness={0.3}
            roughness={0.8}
          />
        </mesh>
      </group>
    </Float>
  );
});

RobotArm.displayName = 'RobotArm';



