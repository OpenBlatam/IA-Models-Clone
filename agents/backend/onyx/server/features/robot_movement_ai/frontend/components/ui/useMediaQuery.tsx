'use client';

import { useMedia } from 'react-use';

export function useMediaQuery(query: string): boolean {
  return useMedia(query, false);
}

export function useBreakpoint() {
  const isXs = useMediaQuery('(min-width: 475px)');
  const isSm = useMediaQuery('(min-width: 640px)');
  const isMd = useMediaQuery('(min-width: 768px)');
  const isLg = useMediaQuery('(min-width: 1024px)');
  const isXl = useMediaQuery('(min-width: 1280px)');
  const is2Xl = useMediaQuery('(min-width: 1536px)');
  const is3Xl = useMediaQuery('(min-width: 1920px)');

  return {
    isXs,
    isSm,
    isMd,
    isLg,
    isXl,
    is2Xl,
    is3Xl,
    current: is3Xl ? '3xl' : is2Xl ? '2xl' : isXl ? 'xl' : isLg ? 'lg' : isMd ? 'md' : isSm ? 'sm' : 'xs',
  };
}

