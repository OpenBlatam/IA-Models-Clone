import { useState, useEffect } from 'react';
import { Dimensions, ScaledSize } from 'react-native';

interface ViewportState {
  width: number;
  height: number;
  scale: number;
  fontScale: number;
  isSmall: boolean;
  isMedium: boolean;
  isLarge: boolean;
  isTablet: boolean;
  isLandscape: boolean;
}

/**
 * Hook to monitor viewport dimensions
 * Provides responsive breakpoints
 */
export function useViewport(): ViewportState {
  const [dimensions, setDimensions] = useState<ScaledSize>(
    Dimensions.get('window')
  );

  useEffect(() => {
    const subscription = Dimensions.addEventListener('change', ({ window }) => {
      setDimensions(window);
    });

    return () => subscription?.remove();
  }, []);

  const { width, height, scale, fontScale } = dimensions;
  const isSmall = width < 375;
  const isMedium = width >= 375 && width < 768;
  const isLarge = width >= 768;
  const isTablet = width >= 768;
  const isLandscape = width > height;

  return {
    width,
    height,
    scale,
    fontScale,
    isSmall,
    isMedium,
    isLarge,
    isTablet,
    isLandscape,
  };
}

