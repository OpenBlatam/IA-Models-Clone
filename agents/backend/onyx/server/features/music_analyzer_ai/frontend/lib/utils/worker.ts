/**
 * Web Worker utility functions.
 * Provides helper functions for Web Worker operations.
 */

/**
 * Creates a Web Worker from a function.
 */
export function createWorker(fn: Function): Worker {
  const blob = new Blob([`(${fn.toString()})()`], {
    type: 'application/javascript',
  });
  return new Worker(URL.createObjectURL(blob));
}

/**
 * Creates a Web Worker from a string.
 */
export function createWorkerFromString(code: string): Worker {
  const blob = new Blob([code], { type: 'application/javascript' });
  return new Worker(URL.createObjectURL(blob));
}

/**
 * Creates a Web Worker from a URL.
 */
export function createWorkerFromURL(url: string): Worker {
  return new Worker(url);
}

/**
 * Terminates a Web Worker and cleans up.
 */
export function terminateWorker(worker: Worker): void {
  worker.terminate();
  // Note: URL.revokeObjectURL should be called if worker was created from blob
}

/**
 * Sends a message to a Web Worker and returns a promise.
 */
export function sendMessage<T, R>(
  worker: Worker,
  message: T,
  timeout?: number
): Promise<R> {
  return new Promise((resolve, reject) => {
    const messageHandler = (event: MessageEvent<R>) => {
      worker.removeEventListener('message', messageHandler);
      worker.removeEventListener('error', errorHandler);
      resolve(event.data);
    };

    const errorHandler = (error: ErrorEvent) => {
      worker.removeEventListener('message', messageHandler);
      worker.removeEventListener('error', errorHandler);
      reject(error);
    };

    worker.addEventListener('message', messageHandler);
    worker.addEventListener('error', errorHandler);

    worker.postMessage(message);

    if (timeout) {
      setTimeout(() => {
        worker.removeEventListener('message', messageHandler);
        worker.removeEventListener('error', errorHandler);
        reject(new Error('Worker message timeout'));
      }, timeout);
    }
  });
}

