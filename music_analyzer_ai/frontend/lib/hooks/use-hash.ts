/**
 * Custom hook for hash generation.
 * Provides reactive hash generation.
 */

import { useMemo } from 'react';
import { hash, hashString, hashObject, hashObjectString } from '../utils/hash';

/**
 * Custom hook for hash generation.
 * Generates hash from value.
 *
 * @param value - Value to hash
 * @returns Hash value
 */
export function useHash(value: string | number | object): number {
  return useMemo(() => {
    if (typeof value === 'string') {
      return hash(value);
    } else if (typeof value === 'number') {
      return hash(value.toString());
    } else {
      return hashObject(value);
    }
  }, [value]);
}

/**
 * Custom hook for hash string generation.
 * Generates hash string from value.
 *
 * @param value - Value to hash
 * @returns Hash string
 */
export function useHashString(value: string | number | object): string {
  return useMemo(() => {
    if (typeof value === 'string') {
      return hashString(value);
    } else if (typeof value === 'number') {
      return hashString(value.toString());
    } else {
      return hashObjectString(value);
    }
  }, [value]);
}

