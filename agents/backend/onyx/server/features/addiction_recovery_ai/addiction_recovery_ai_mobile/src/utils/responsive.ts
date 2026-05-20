import { useWindowDimensions } from 'react-native';

export interface ResponsiveBreakpoints {
  isSmallDevice: boolean;
  isMediumDevice: boolean;
  isLargeDevice: boolean;
  isTablet: boolean;
  width: number;
  height: number;
}

const BREAKPOINTS = {
  small: 375,
  medium: 768,
  large: 1024,
  tablet: 768,
} as const;

export function useResponsive(): ResponsiveBreakpoints {
  const { width, height } = useWindowDimensions();

  return {
    isSmallDevice: width < BREAKPOINTS.small,
    isMediumDevice: width >= BREAKPOINTS.small && width < BREAKPOINTS.medium,
    isLargeDevice: width >= BREAKPOINTS.large,
    isTablet: width >= BREAKPOINTS.tablet,
    width,
    height,
  };
}

export function getResponsiveValue<T>(
  values: {
    small?: T;
    medium?: T;
    large?: T;
    default: T;
  },
  width: number
): T {
  if (width < BREAKPOINTS.small && values.small !== undefined) {
    return values.small;
  }
  if (width < BREAKPOINTS.medium && values.medium !== undefined) {
    return values.medium;
  }
  if (width >= BREAKPOINTS.large && values.large !== undefined) {
    return values.large;
  }
  return values.default;
}

