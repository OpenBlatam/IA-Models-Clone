/**
 * Custom hook for generating unique IDs.
 * Provides stable IDs for components that need unique identifiers.
 */

import { useMemo } from 'react';

let idCounter = 0;

/**
 * Generates a unique ID.
 * @param prefix - ID prefix (default: 'id')
 * @returns Unique ID string
 */
function generateId(prefix: string = 'id'): string {
  return `${prefix}-${++idCounter}`;
}

/**
 * Custom hook for generating unique IDs.
 * Returns a stable ID that persists across re-renders.
 *
 * @param prefix - ID prefix (default: 'id')
 * @returns Unique ID string
 */
export function useId(prefix: string = 'id'): string {
  return useMemo(() => generateId(prefix), [prefix]);
}

