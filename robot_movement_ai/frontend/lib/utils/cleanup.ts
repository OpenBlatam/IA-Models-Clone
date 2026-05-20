/**
 * Cleanup utilities
 */

/**
 * Create cleanup function
 */
export function createCleanup(): () => void {
  const cleanups: Array<() => void> = [];

  return {
    add: (cleanup: () => void) => {
      cleanups.push(cleanup);
    },
    run: () => {
      cleanups.forEach((cleanup) => cleanup());
      cleanups.length = 0;
    },
  } as any;
}

/**
 * Combine multiple cleanup functions
 */
export function combineCleanups(...cleanups: Array<() => void>): () => void {
  return () => {
    cleanups.forEach((cleanup) => cleanup());
  };
}

/**
 * Safe cleanup (handles errors)
 */
export function safeCleanup(cleanup: () => void): () => void {
  return () => {
    try {
      cleanup();
    } catch (error) {
      console.error('Error in cleanup:', error);
    }
  };
}



