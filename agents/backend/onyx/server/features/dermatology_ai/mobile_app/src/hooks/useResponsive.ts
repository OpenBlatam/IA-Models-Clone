import { useState, useEffect, useMemo } from 'react';
import { Dimensions, ScaledSize } from 'react-native';

interface Breakpoints {
  xs: number;
  sm: number;
  md: number;
  lg: number;
  xl: number;
}

const DEFAULT_BREAKPOINTS: Breakpoints = {
  xs: 0,
  sm: 576,
  md: 768,
  lg: 992,
  xl: 1200,
};

/**
 * Hook para obtener dimensiones responsivas
 */
export const useResponsive = (breakpoints: Breakpoints = DEFAULT_BREAKPOINTS) => {
  const [dimensions, setDimensions] = useState<ScaledSize>(
    Dimensions.get('window')
  );

  useEffect(() => {
    const subscription = Dimensions.addEventListener('change', ({ window }) => {
      setDimensions(window);
    });

    return () => subscription?.remove();
  }, []);

  const isXs = useMemo(
    () => dimensions.width < breakpoints.sm,
    [dimensions.width, breakpoints.sm]
  );
  const isSm = useMemo(
    () =>
      dimensions.width >= breakpoints.sm && dimensions.width < breakpoints.md,
    [dimensions.width, breakpoints.sm, breakpoints.md]
  );
  const isMd = useMemo(
    () =>
      dimensions.width >= breakpoints.md && dimensions.width < breakpoints.lg,
    [dimensions.width, breakpoints.md, breakpoints.lg]
  );
  const isLg = useMemo(
    () =>
      dimensions.width >= breakpoints.lg && dimensions.width < breakpoints.xl,
    [dimensions.width, breakpoints.lg, breakpoints.xl]
  );
  const isXl = useMemo(
    () => dimensions.width >= breakpoints.xl,
    [dimensions.width, breakpoints.xl]
  );

  const isPortrait = useMemo(
    () => dimensions.height > dimensions.width,
    [dimensions.height, dimensions.width]
  );
  const isLandscape = useMemo(
    () => dimensions.width > dimensions.height,
    [dimensions.width, dimensions.height]
  );

  return {
    width: dimensions.width,
    height: dimensions.height,
    isXs,
    isSm,
    isMd,
    isLg,
    isXl,
    isPortrait,
    isLandscape,
    breakpoint: useMemo(() => {
      if (isXl) return 'xl';
      if (isLg) return 'lg';
      if (isMd) return 'md';
      if (isSm) return 'sm';
      return 'xs';
    }, [isXl, isLg, isMd, isSm]),
  };
};

/**
 * Hook para obtener valores responsivos basados en breakpoints
 */
export const useResponsiveValue = <T,>(
  values: {
    xs?: T;
    sm?: T;
    md?: T;
    lg?: T;
    xl?: T;
  },
  breakpoints: Breakpoints = DEFAULT_BREAKPOINTS
): T | undefined => {
  const { width } = useResponsive(breakpoints);

  return useMemo(() => {
    if (width >= breakpoints.xl && values.xl !== undefined) return values.xl;
    if (width >= breakpoints.lg && values.lg !== undefined) return values.lg;
    if (width >= breakpoints.md && values.md !== undefined) return values.md;
    if (width >= breakpoints.sm && values.sm !== undefined) return values.sm;
    return values.xs;
  }, [width, values, breakpoints]);
};

