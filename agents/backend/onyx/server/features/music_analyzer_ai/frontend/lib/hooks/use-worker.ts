/**
 * Custom hook for Web Worker.
 * Provides reactive Web Worker functionality.
 */

import { useRef, useEffect, useCallback } from 'react';
import {
  createWorker,
  createWorkerFromString,
  createWorkerFromURL,
  terminateWorker,
  sendMessage,
} from '../utils/worker';

/**
 * Options for useWorker hook.
 */
export interface UseWorkerOptions {
  timeout?: number;
}

/**
 * Custom hook for Web Worker.
 * Provides reactive Web Worker functionality.
 *
 * @param workerFn - Worker function or URL
 * @param options - Worker options
 * @returns Worker operations
 */
export function useWorker(
  workerFn: Function | string,
  options: UseWorkerOptions = {}
) {
  const workerRef = useRef<Worker | null>(null);
  const { timeout } = options;

  useEffect(() => {
    if (typeof workerFn === 'string') {
      // Assume it's a URL if it starts with http:// or https://
      if (workerFn.startsWith('http://') || workerFn.startsWith('https://')) {
        workerRef.current = createWorkerFromURL(workerFn);
      } else {
        workerRef.current = createWorkerFromString(workerFn);
      }
    } else {
      workerRef.current = createWorker(workerFn);
    }

    return () => {
      if (workerRef.current) {
        terminateWorker(workerRef.current);
        workerRef.current = null;
      }
    };
  }, [workerFn]);

  const postMessage = useCallback(
    <T, R>(message: T): Promise<R> => {
      if (!workerRef.current) {
        throw new Error('Worker not initialized');
      }
      return sendMessage<T, R>(workerRef.current, message, timeout);
    },
    [timeout]
  );

  const terminate = useCallback(() => {
    if (workerRef.current) {
      terminateWorker(workerRef.current);
      workerRef.current = null;
    }
  }, []);

  return {
    postMessage,
    terminate,
    worker: workerRef.current,
  };
}

