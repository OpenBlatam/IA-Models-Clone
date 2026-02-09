/**
 * Import optimization utilities
 * @module robot-3d-view/utils/import-optimization
 */

/**
 * Lazy import configuration for code splitting
 */
export const lazyImports = {
  // Heavy 3D components - only load when needed
  scene3D: () =>
    import('../scene/scene-3d').then((mod) => ({ default: mod.Scene3D })),
  
  // UI controls - can be loaded separately
  viewControls: () =>
    import('../controls/view-controls').then((mod) => ({ default: mod.ViewControls })),
  
  infoOverlay: () =>
    import('../controls/info-overlay').then((mod) => ({ default: mod.InfoOverlay })),
  
  instructionsOverlay: () =>
    import('../controls/instructions-overlay').then((mod) => ({
      default: mod.InstructionsOverlay,
    })),
  
  screenshotControls: () =>
    import('../controls/screenshot-controls').then((mod) => ({
      default: mod.ScreenshotControls,
    })),
  
  // Effects - load on demand
  particleSystem: () =>
    import('../effects/particle-system').then((mod) => ({
      default: mod.ParticleSystem,
    })),
  
  glowEffect: () =>
    import('../effects/glow-effect').then((mod) => ({ default: mod.GlowEffect })),
  
  // Objects - lazy load heavy 3D objects
  robotArm: () =>
    import('../objects/robot-arm').then((mod) => ({ default: mod.RobotArm })),
  
  targetMarker: () =>
    import('../objects/target-marker').then((mod) => ({
      default: mod.TargetMarker,
    })),
  
  trajectoryPath: () =>
    import('../objects/trajectory-path').then((mod) => ({
      default: mod.TrajectoryPath,
    })),
} as const;

/**
 * Preload critical components
 * 
 * @param components - Components to preload
 */
export async function preloadComponents(
  ...components: Array<keyof typeof lazyImports>
): Promise<void> {
  await Promise.all(components.map((key) => lazyImports[key]()));
}

/**
 * Checks if a component is already loaded
 * 
 * @param component - Component key
 * @returns true if component is loaded
 */
export function isComponentLoaded(
  component: keyof typeof lazyImports
): boolean {
  // This is a simplified check - in production you might want
  // to track loaded modules more accurately
  return typeof window !== 'undefined' && component in window;
}



