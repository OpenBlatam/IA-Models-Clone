import { useEffect, useRef } from 'react';

/**
 * Hook to track render count for debugging performance issues.
 * Only works in development mode.
 */
export function useRenderCount(componentName?: string): number {
  const countRef = useRef(0);

  useEffect(() => {
    if (__DEV__) {
      countRef.current += 1;
      if (componentName) {
        console.log(`[Render Count] ${componentName}: ${countRef.current}`);
      }
    }
  });

  return countRef.current;
}

