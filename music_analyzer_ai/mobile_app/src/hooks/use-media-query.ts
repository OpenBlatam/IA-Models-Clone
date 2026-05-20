import { useState, useEffect } from 'react';
import { Dimensions, ScaledSize } from 'react-native';

interface MediaQuery {
  minWidth?: number;
  maxWidth?: number;
  minHeight?: number;
  maxHeight?: number;
  orientation?: 'portrait' | 'landscape';
}

/**
 * Hook for responsive media queries
 * Matches CSS media query behavior
 */
export function useMediaQuery(query: MediaQuery): boolean {
  const [matches, setMatches] = useState(() => {
    const { width, height } = Dimensions.get('window');
    return matchQuery(query, width, height);
  });

  useEffect(() => {
    const subscription = Dimensions.addEventListener('change', ({ window }) => {
      const newMatches = matchQuery(query, window.width, window.height);
      setMatches(newMatches);
    });

    return () => subscription?.remove();
  }, [query]);

  return matches;
}

function matchQuery(
  query: MediaQuery,
  width: number,
  height: number
): boolean {
  if (query.minWidth !== undefined && width < query.minWidth) {
    return false;
  }

  if (query.maxWidth !== undefined && width > query.maxWidth) {
    return false;
  }

  if (query.minHeight !== undefined && height < query.minHeight) {
    return false;
  }

  if (query.maxHeight !== undefined && height > query.maxHeight) {
    return false;
  }

  if (query.orientation) {
    const isLandscape = width > height;
    if (
      (query.orientation === 'landscape' && !isLandscape) ||
      (query.orientation === 'portrait' && isLandscape)
    ) {
      return false;
    }
  }

  return true;
}

/**
 * Predefined media query hooks
 */
export function useIsSmallScreen(): boolean {
  return useMediaQuery({ maxWidth: 375 });
}

export function useIsMediumScreen(): boolean {
  return useMediaQuery({ minWidth: 376, maxWidth: 767 });
}

export function useIsLargeScreen(): boolean {
  return useMediaQuery({ minWidth: 768 });
}

export function useIsTablet(): boolean {
  return useMediaQuery({ minWidth: 768 });
}

export function useIsLandscape(): boolean {
  return useMediaQuery({ orientation: 'landscape' });
}

export function useIsPortrait(): boolean {
  return useMediaQuery({ orientation: 'portrait' });
}

