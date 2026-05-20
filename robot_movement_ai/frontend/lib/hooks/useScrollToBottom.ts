import { useEffect, useRef, useCallback } from 'react';

export function useScrollToBottom<T extends HTMLElement = HTMLDivElement>(
  dependencies: React.DependencyList = []
) {
  const containerRef = useRef<T>(null);

  const scrollToBottom = useCallback(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [scrollToBottom, ...dependencies]);

  return { containerRef, scrollToBottom };
}



