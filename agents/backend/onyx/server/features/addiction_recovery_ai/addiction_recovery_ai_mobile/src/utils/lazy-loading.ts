import { lazy, ComponentType } from 'react';

export function createLazyScreen<T extends ComponentType<any>>(
  importFn: () => Promise<{ default: T }>
) {
  return lazy(importFn);
}

export function preloadComponent<T extends ComponentType<any>>(
  importFn: () => Promise<{ default: T }>
): Promise<void> {
  return importFn().then(() => {});
}

export function createPreloadableComponent<T extends ComponentType<any>>(
  importFn: () => Promise<{ default: T }>
) {
  const LazyComponent = lazy(importFn);
  
  // Preload on idle
  if (typeof requestIdleCallback !== 'undefined') {
    requestIdleCallback(() => {
      preloadComponent(importFn);
    });
  }

  return LazyComponent;
}

