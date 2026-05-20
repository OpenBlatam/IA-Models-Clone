import React from 'react';

export const lazyLoad = <T extends (...args: any[]) => Promise<any>>(
  importFn: () => Promise<{ default: T }>
): T => {
  return (async (...args: Parameters<T>) => {
    const module = await importFn();
    return module.default(...args);
  }) as T;
};

export const createLazyComponent = <T extends React.ComponentType<any>>(
  importFn: () => Promise<{ default: T }>
): React.LazyExoticComponent<T> => {
  return React.lazy(importFn) as React.LazyExoticComponent<T>;
};

