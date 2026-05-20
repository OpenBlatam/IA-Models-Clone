import { useWindowDimensions } from 'react-native';
import { useMemo } from 'react';

export interface ResponsiveBreakpoints {
  isSmall: boolean;
  isMedium: boolean;
  isLarge: boolean;
  isTablet: boolean;
  isPhone: boolean;
  width: number;
  height: number;
}

export function useResponsive(): ResponsiveBreakpoints {
  const { width, height } = useWindowDimensions();

  return useMemo(
    () => ({
      isSmall: width < 375,
      isMedium: width >= 375 && width < 768,
      isLarge: width >= 768,
      isTablet: width >= 768,
      isPhone: width < 768,
      width,
      height,
    }),
    [width, height]
  );
}


