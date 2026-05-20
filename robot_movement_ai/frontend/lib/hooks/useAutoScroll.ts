import { useEffect, useRef } from 'react';

export interface UseAutoScrollOptions {
  enabled?: boolean;
  behavior?: ScrollBehavior;
  threshold?: number; // Distance from bottom to trigger auto-scroll
}

export function useAutoScroll<T extends HTMLElement = HTMLDivElement>(
  dependencies: React.DependencyList = [],
  options: UseAutoScrollOptions = {}
) {
  const {
    enabled = true,
    behavior = 'smooth',
    threshold = 100,
  } = options;

  const containerRef = useRef<T>(null);

  useEffect(() => {
    if (!enabled || !containerRef.current) return;

    const container = containerRef.current;
    const isNearBottom =
      container.scrollHeight - container.scrollTop - container.clientHeight < threshold;

    if (isNearBottom) {
      container.scrollTo({
        top: container.scrollHeight,
        behavior,
      });
    }
  }, [enabled, behavior, threshold, ...dependencies]);

  return containerRef;
}



