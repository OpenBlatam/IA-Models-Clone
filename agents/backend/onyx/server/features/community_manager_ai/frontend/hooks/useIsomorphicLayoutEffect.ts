/**
 * useIsomorphicLayoutEffect Hook
 * Hook that uses useLayoutEffect on client and useEffect on server
 */

import { useEffect, useLayoutEffect } from 'react';

/**
 * useLayoutEffect on client, useEffect on server
 * Prevents hydration mismatches in SSR
 */
export const useIsomorphicLayoutEffect =
  typeof window !== 'undefined' ? useLayoutEffect : useEffect;


