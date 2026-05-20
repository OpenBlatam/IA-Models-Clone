/**
 * Custom hook for mount/unmount callbacks.
 * Provides callbacks for component mount and unmount.
 */

import { useEffect } from 'react';

/**
 * Options for useMount hook.
 */
export interface UseMountOptions {
  onMount?: () => void;
  onUnmount?: () => void;
}

/**
 * Custom hook for mount/unmount callbacks.
 * Executes callbacks when component mounts or unmounts.
 *
 * @param options - Hook options
 */
export function useMount(options: UseMountOptions): void {
  const { onMount, onUnmount } = options;

  useEffect(() => {
    onMount?.();

    return () => {
      onUnmount?.();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
}

