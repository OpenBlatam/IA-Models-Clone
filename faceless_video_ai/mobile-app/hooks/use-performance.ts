import { useEffect, useRef, useCallback } from 'react';
import { InteractionManager } from 'react-native';

export function usePerformanceMonitor() {
  const renderStartTime = useRef<number>(0);
  const renderCount = useRef<number>(0);

  useEffect(() => {
    renderStartTime.current = performance.now();
    renderCount.current += 1;

    const renderTime = performance.now() - renderStartTime.current;
    if (renderTime > 16) {
      // More than one frame (16ms at 60fps)
      console.warn(`Slow render detected: ${renderTime.toFixed(2)}ms`);
    }
  });

  return {
    renderCount: renderCount.current,
  };
}

export function useDeferredCallback<T extends (...args: any[]) => any>(
  callback: T,
  delay = 0
): T {
  const timeoutRef = useRef<NodeJS.Timeout>();

  const deferredCallback = useCallback(
    ((...args: Parameters<T>) => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }

      timeoutRef.current = setTimeout(() => {
        callback(...args);
      }, delay);
    }) as T,
    [callback, delay]
  );

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  return deferredCallback;
}

export function useInteractionCallback(callback: () => void) {
  const callbackRef = useRef(callback);

  useEffect(() => {
    callbackRef.current = callback;
  }, [callback]);

  return useCallback(() => {
    InteractionManager.runAfterInteractions(() => {
      callbackRef.current();
    });
  }, []);
}

export function useRenderCount(componentName?: string) {
  const renderCount = useRef(0);

  useEffect(() => {
    renderCount.current += 1;
    if (componentName && __DEV__) {
      console.log(`${componentName} rendered ${renderCount.current} times`);
    }
  });

  return renderCount.current;
}

export function useMemoizedCallback<T extends (...args: any[]) => any>(
  callback: T,
  deps: React.DependencyList
): T {
  const callbackRef = useRef(callback);

  useEffect(() => {
    callbackRef.current = callback;
  }, [callback, ...deps]);

  return useCallback(
    ((...args: Parameters<T>) => {
      return callbackRef.current(...args);
    }) as T,
    []
  );
}


