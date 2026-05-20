import { useState, useEffect } from 'react';
import { Dimensions, ScaledSize } from 'react-native';

interface WindowDimensions {
  width: number;
  height: number;
  scale: number;
  fontScale: number;
}

export function useWindowDimensions(): WindowDimensions {
  const [dimensions, setDimensions] = useState(() => {
    const { width, height, scale, fontScale } = Dimensions.get('window');
    return { width, height, scale, fontScale };
  });

  useEffect(() => {
    const subscription = Dimensions.addEventListener('change', ({ window }: { window: ScaledSize }) => {
      setDimensions({
        width: window.width,
        height: window.height,
        scale: window.scale,
        fontScale: window.fontScale,
      });
    });

    return () => subscription?.remove();
  }, []);

  return dimensions;
}

/**
 * Breakpoint helpers for responsive design
 */
export function useBreakpoints() {
  const { width } = useWindowDimensions();

  return {
    isSmall: width < 375,
    isMedium: width >= 375 && width < 768,
    isLarge: width >= 768,
    isTablet: width >= 768,
    isPhone: width < 768,
  };
}


