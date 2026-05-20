import { useEffect, useRef } from 'react';
import { useFocusEffect as useExpoFocusEffect } from 'expo-router';
import type { EffectCallback } from 'react';

/**
 * Enhanced focus effect hook with cleanup
 * Runs callback when screen comes into focus
 */
export function useFocusEffect(callback: EffectCallback): void {
  useExpoFocusEffect(
    useEffect(() => {
      return callback();
    }, [callback])
  );
}

/**
 * Hook that runs callback only once when screen first focuses
 */
export function useFocusOnce(callback: () => void): void {
  const hasFocusedRef = useRef(false);

  useFocusEffect(() => {
    if (!hasFocusedRef.current) {
      hasFocusedRef.current = true;
      callback();
    }
  });
}

