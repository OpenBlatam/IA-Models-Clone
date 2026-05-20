import { InteractionManager } from 'react-native';

const preloadQueue: Array<() => Promise<unknown>> = [];
let isProcessing = false;

async function processPreloadQueue(): Promise<void> {
  if (isProcessing || preloadQueue.length === 0) {
    return;
  }

  isProcessing = true;

  while (preloadQueue.length > 0) {
    const task = preloadQueue.shift();
    if (task) {
      try {
        await task();
      } catch (error) {
        console.warn('Preload task failed:', error);
      }
    }
    
    // Yield to UI thread
    await new Promise((resolve) => setTimeout(resolve, 0));
  }

  isProcessing = false;
}

export function preload<T>(importFn: () => Promise<T>): void {
  preloadQueue.push(importFn);
  
  InteractionManager.runAfterInteractions(() => {
    processPreloadQueue();
  });
}

export function preloadCritical<T>(importFn: () => Promise<T>): Promise<T> {
  return importFn();
}

export function preloadBatch<T>(
  importFns: Array<() => Promise<T>>
): void {
  importFns.forEach((fn) => preload(fn));
}

