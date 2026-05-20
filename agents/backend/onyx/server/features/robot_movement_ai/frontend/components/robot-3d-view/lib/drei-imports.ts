/**
 * Optimized @react-three/drei imports
 * 
 * This file provides optimized imports for @react-three/drei to improve
 * tree-shaking and reduce bundle size. Components are organized by category
 * and can be imported individually.
 * 
 * @module robot-3d-view/lib/drei-imports
 */

// Core components - most commonly used
export {
  Grid,
  Environment,
  ContactShadows,
  OrbitControls,
  PerspectiveCamera,
  Stats,
  GizmoHelper,
  GizmoViewport,
} from '@react-three/drei';

// Presentation and controls
export {
  PresentationControls,
} from '@react-three/drei';

// Visual effects
export {
  Sparkles,
  Stars,
  Sky,
} from '@react-three/drei';

// UI and overlays
export {
  Html,
  Text,
} from '@react-three/drei';

// Animation helpers
export {
  Float,
} from '@react-three/drei';

// Geometry helpers
export {
  Line,
} from '@react-three/drei';

/**
 * Lazy-loaded drei components
 * Use these for components that are not always needed
 */
export const lazyDreiComponents = {
  Sky: () => import('@react-three/drei').then((mod) => ({ default: mod.Sky })),
  Stars: () => import('@react-three/drei').then((mod) => ({ default: mod.Stars })),
  Text: () => import('@react-three/drei').then((mod) => ({ default: mod.Text })),
  // Add more lazy-loaded components as needed
} as const;

/**
 * Component categories for better organization
 */
export const dreiComponents = {
  // Core scene components
  core: {
    Grid,
    Environment,
    ContactShadows,
    OrbitControls,
    PerspectiveCamera,
  },
  // UI and overlays
  ui: {
    Html,
    Text,
    Stats,
  },
  // Controls
  controls: {
    PresentationControls,
    GizmoHelper,
    GizmoViewport,
  },
  // Effects
  effects: {
    Sparkles,
    Stars,
    Sky,
  },
  // Animation
  animation: {
    Float,
  },
  // Geometry
  geometry: {
    Line,
  },
} as const;



