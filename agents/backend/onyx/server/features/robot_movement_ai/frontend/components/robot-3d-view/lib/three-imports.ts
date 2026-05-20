/**
 * Optimized Three.js imports
 * 
 * This file provides optimized imports for Three.js to improve tree-shaking
 * and reduce bundle size. Instead of importing the entire namespace,
 * we import only the specific classes and functions needed.
 * 
 * @module robot-3d-view/lib/three-imports
 */

// Core Three.js classes - import only what we need
export {
  Vector3,
  type Vector3 as Vector3Type,
  Group,
  type Group as GroupType,
  Mesh,
  type Mesh as MeshType,
  SphereGeometry,
  BoxGeometry,
  CylinderGeometry,
  ConeGeometry,
  RingGeometry,
  PlaneGeometry,
  BufferGeometry,
  BufferAttribute,
  LineBasicMaterial,
  MeshStandardMaterial,
  type MeshStandardMaterial as MeshStandardMaterialType,
  DoubleSide,
  CatmullRomCurve3,
  QuadraticBezierCurve3,
  type Curve as CurveType,
} from 'three';

// Re-export commonly used types
export type { Object3D } from 'three';

/**
 * Three.js namespace for cases where we need the full namespace
 * Use sparingly - prefer specific imports above
 */
export { default as THREE } from 'three';

/**
 * Helper to create optimized geometry instances
 */
export const createGeometry = {
  sphere: (radius: number, segments = 16) =>
    new SphereGeometry(radius, segments, segments),
  box: (width: number, height: number, depth: number) =>
    new BoxGeometry(width, height, depth),
  cylinder: (radiusTop: number, radiusBottom: number, height: number, segments = 8) =>
    new CylinderGeometry(radiusTop, radiusBottom, height, segments),
  cone: (radius: number, height: number, segments = 8) =>
    new ConeGeometry(radius, height, segments),
  ring: (innerRadius: number, outerRadius: number, segments = 32) =>
    new RingGeometry(innerRadius, outerRadius, segments),
  plane: (width: number, height: number) =>
    new PlaneGeometry(width, height),
};

/**
 * Helper to create optimized material instances
 */
export const createMaterial = {
  standard: (options?: {
    color?: string | number;
    metalness?: number;
    roughness?: number;
    emissive?: string | number;
    emissiveIntensity?: number;
    opacity?: number;
    transparent?: boolean;
  }) => {
    const material = new MeshStandardMaterial();
    if (options) {
      if (options.color) material.color.set(options.color);
      if (options.metalness !== undefined) material.metalness = options.metalness;
      if (options.roughness !== undefined) material.roughness = options.roughness;
      if (options.emissive) material.emissive.set(options.emissive);
      if (options.emissiveIntensity !== undefined)
        material.emissiveIntensity = options.emissiveIntensity;
      if (options.opacity !== undefined) material.opacity = options.opacity;
      if (options.transparent !== undefined) material.transparent = options.transparent;
    }
    return material;
  },
  line: (color?: string | number, linewidth = 1) => {
    const material = new LineBasicMaterial({ color, linewidth });
    return material;
  },
};



