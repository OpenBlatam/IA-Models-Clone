/**
 * Web Worker utilities for heavy computations
 * @module robot-3d-view/utils/web-worker
 */

/**
 * Worker message
 */
export interface WorkerMessage<T = unknown> {
  type: string;
  data: T;
  id?: string;
}

/**
 * Worker response
 */
export interface WorkerResponse<T = unknown> {
  success: boolean;
  data?: T;
  error?: string;
  id?: string;
}

/**
 * Creates a Web Worker from a function
 * 
 * @param fn - Function to run in worker
 * @returns Worker instance
 */
export function createWorker(fn: Function): Worker {
  const blob = new Blob([`(${fn.toString()})()`], {
    type: 'application/javascript',
  });
  return new Worker(URL.createObjectURL(blob));
}

/**
 * Worker for trajectory calculations
 */
export function createTrajectoryWorker(): Worker {
  return createWorker(() => {
    self.onmessage = (e: MessageEvent<WorkerMessage>) => {
      const { type, data, id } = e.data;

      if (type === 'calculate-trajectory') {
        try {
          // Heavy trajectory calculation
          const { start, end, steps } = data as {
            start: number[];
            end: number[];
            steps: number;
          };

          const trajectory: number[][] = [];
          for (let i = 0; i <= steps; i++) {
            const t = i / steps;
            const point = start.map((s, idx) => s + (end[idx] - s) * t);
            trajectory.push(point);
          }

          self.postMessage({
            success: true,
            data: trajectory,
            id,
          } as WorkerResponse);
        } catch (error) {
          self.postMessage({
            success: false,
            error: error instanceof Error ? error.message : 'Unknown error',
            id,
          } as WorkerResponse);
        }
      }
    };
  });
}

/**
 * Worker for path optimization
 */
export function createPathOptimizationWorker(): Worker {
  return createWorker(() => {
    self.onmessage = (e: MessageEvent<WorkerMessage>) => {
      const { type, data, id } = e.data;

      if (type === 'optimize-path') {
        try {
          // Path optimization algorithm
          const { points } = data as { points: number[][] };

          // Simple optimization (could be more complex)
          const optimized = points.filter((_, index) => index % 2 === 0 || index === points.length - 1);

          self.postMessage({
            success: true,
            data: optimized,
            id,
          } as WorkerResponse);
        } catch (error) {
          self.postMessage({
            success: false,
            error: error instanceof Error ? error.message : 'Unknown error',
            id,
          } as WorkerResponse);
        }
      }
    };
  });
}

/**
 * Hook-like function to use workers
 */
export function useWorker<T, R>(
  worker: Worker,
  message: WorkerMessage<T>
): Promise<WorkerResponse<R>> {
  return new Promise((resolve, reject) => {
    const messageId = `${Date.now()}-${Math.random()}`;
    const messageWithId = { ...message, id: messageId };

    const handler = (e: MessageEvent<WorkerResponse<R>>) => {
      if (e.data.id === messageId) {
        worker.removeEventListener('message', handler);
        if (e.data.success) {
          resolve(e.data);
        } else {
          reject(new Error(e.data.error || 'Worker error'));
        }
      }
    };

    worker.addEventListener('message', handler);
    worker.postMessage(messageWithId);

    // Timeout after 30 seconds
    setTimeout(() => {
      worker.removeEventListener('message', handler);
      reject(new Error('Worker timeout'));
    }, 30000);
  });
}



