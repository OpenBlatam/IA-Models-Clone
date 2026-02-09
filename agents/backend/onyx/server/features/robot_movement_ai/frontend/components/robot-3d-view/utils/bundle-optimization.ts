/**
 * Bundle optimization utilities
 * @module robot-3d-view/utils/bundle-optimization
 */

/**
 * Analyzes bundle size and provides optimization suggestions
 */
export interface BundleAnalysis {
  totalSize: number;
  chunks: Array<{
    name: string;
    size: number;
    percentage: number;
  }>;
  suggestions: string[];
}

/**
 * Gets bundle size information (if available in development)
 * 
 * @returns Bundle analysis or null if not available
 */
export function analyzeBundle(): BundleAnalysis | null {
  if (typeof window === 'undefined' || process.env.NODE_ENV !== 'development') {
    return null;
  }

  // This would integrate with webpack-bundle-analyzer or similar
  // For now, return a placeholder structure
  return {
    totalSize: 0,
    chunks: [],
    suggestions: [
      'Use dynamic imports for heavy 3D components',
      'Optimize Three.js imports to use specific classes',
      'Lazy load @react-three/drei components',
      'Consider code splitting for different view modes',
    ],
  };
}

/**
 * Preloads critical components for better perceived performance
 * 
 * @param components - List of component keys to preload
 */
export async function preloadCriticalComponents(
  components: string[]
): Promise<void> {
  if (typeof window === 'undefined') return;

  const { lazyImports } = await import('./import-optimization');
  
  await Promise.all(
    components
      .filter((key): key is keyof typeof lazyImports => key in lazyImports)
      .map((key) => lazyImports[key]())
  );
}

/**
 * Checks if a library is loaded
 * 
 * @param library - Library name to check
 * @returns true if library is loaded
 */
export function isLibraryLoaded(library: 'three' | 'drei' | 'fiber'): boolean {
  if (typeof window === 'undefined') return false;

  switch (library) {
    case 'three':
      return 'THREE' in window || typeof (window as any).THREE !== 'undefined';
    case 'drei':
      return typeof (window as any).drei !== 'undefined';
    case 'fiber':
      return typeof (window as any).fiber !== 'undefined';
    default:
      return false;
  }
}

/**
 * Gets estimated bundle size for a component
 * 
 * @param componentName - Name of the component
 * @returns Estimated size in KB
 */
export function getEstimatedSize(componentName: string): number {
  // Rough estimates based on typical component sizes
  const estimates: Record<string, number> = {
    'Scene3D': 150,
    'RobotArm': 50,
    'TargetMarker': 20,
    'TrajectoryPath': 30,
    'EnvironmentSetup': 100,
    'ViewControls': 15,
    'InfoOverlay': 10,
  };

  return estimates[componentName] || 25;
}

/**
 * Optimization recommendations based on current usage
 */
export function getOptimizationRecommendations(): string[] {
  const recommendations: string[] = [];

  // Check for namespace imports
  if (typeof window !== 'undefined') {
    recommendations.push(
      'Use specific imports from three-imports.ts instead of import * as THREE',
      'Use lazy-loaded drei components for Sky and Stars',
      'Consider using dynamic imports for heavy effects',
      'Enable compression in Next.js config',
      'Use WebP format for any textures or images',
    );
  }

  return recommendations;
}



