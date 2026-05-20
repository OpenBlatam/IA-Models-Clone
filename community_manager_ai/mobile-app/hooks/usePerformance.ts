import { useEffect, useRef } from 'react';
import { InteractionManager } from 'react-native';

/**
 * Performance monitoring and optimization hooks
 */

/**
 * Measure component render time
 */
export function useRenderTime(componentName: string) {
  const renderStart = useRef<number>(Date.now());

  useEffect(() => {
    if (__DEV__) {
      renderStart.current = Date.now();
      
      return () => {
        const renderTime = Date.now() - renderStart.current;
        if (renderTime > 16) {
          // Warn if render takes longer than one frame (16ms)
          console.warn(`${componentName} took ${renderTime}ms to render`);
        }
      };
    }
  }, [componentName]);
}

/**
 * Defer heavy operations until after interactions
 */
export function useDeferredCallback(callback: () => void, deps: React.DependencyList) {
  useEffect(() => {
    const interaction = InteractionManager.runAfterInteractions(() => {
      callback();
    });

    return () => {
      interaction.cancel();
    };
  }, deps);
}

/**
 * Batch state updates
 */
export function useBatchedUpdates() {
  const updatesRef = useRef<Array<() => void>>([]);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  const batchUpdate = (update: () => void) => {
    updatesRef.current.push(update);

    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }

    timeoutRef.current = setTimeout(() => {
      const updates = updatesRef.current;
      updatesRef.current = [];
      
      InteractionManager.runAfterInteractions(() => {
        updates.forEach((update) => update());
      });
    }, 0);
  };

  return batchUpdate;
}

/**
 * Monitor memory usage (development only)
 * Note: performance.memory is not available in React Native
 * This is a placeholder for future memory monitoring integration
 */
export function useMemoryMonitor(componentName: string) {
  useEffect(() => {
    if (__DEV__) {
      // Memory monitoring would require native modules
      // This is a placeholder for future implementation
      console.log(`Memory monitoring enabled for ${componentName}`);
    }
  }, [componentName]);
}

