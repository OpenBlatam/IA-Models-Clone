import { useState, useEffect, useRef, useCallback } from 'react';
import { View, LayoutChangeEvent } from 'react-native';
import { useWindowDimensions } from 'react-native';

interface UseIntersectionOptions {
  threshold?: number;
  rootMargin?: number;
}

/**
 * Hook to detect element intersection with viewport
 * Returns intersection state and ref
 * React Native compatible version
 */
export function useIntersection({
  threshold = 0,
  rootMargin = 0,
}: UseIntersectionOptions = {}) {
  const [isIntersecting, setIsIntersecting] = useState(false);
  const elementRef = useRef<View>(null);
  const layoutRef = useRef<{ x: number; y: number; width: number; height: number } | null>(null);
  const { width: windowWidth, height: windowHeight } = useWindowDimensions();

  const checkIntersection = useCallback(() => {
    if (!layoutRef.current) return;

    const { x, y, width, height } = layoutRef.current;
    const visibleWidth = Math.max(0, Math.min(x + width, windowWidth) - Math.max(x, 0));
    const visibleHeight = Math.max(0, Math.min(y + height, windowHeight) - Math.max(y, 0));
    const visibleArea = visibleWidth * visibleHeight;
    const totalArea = width * height;
    const intersectionRatio = totalArea > 0 ? visibleArea / totalArea : 0;

    const intersecting = intersectionRatio >= threshold;
    setIsIntersecting(intersecting);
  }, [threshold, windowWidth, windowHeight]);

  const handleLayout = useCallback((event: LayoutChangeEvent) => {
    const { x, y, width, height } = event.nativeEvent.layout;
    layoutRef.current = { x, y, width, height };
    checkIntersection();
  }, [checkIntersection]);

  useEffect(() => {
    checkIntersection();
  }, [checkIntersection]);

  return {
    ref: elementRef,
    onLayout: handleLayout,
    isIntersecting,
  };
}

