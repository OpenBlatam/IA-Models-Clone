/**
 * React component optimization utilities.
 * Provides utilities for memoization, lazy loading, and performance optimization.
 */

import React, { type ComponentType, type ReactElement, memo, lazy, Suspense } from 'react';

/**
 * Options for component memoization.
 */
export interface MemoizeOptions {
  /**
   * Custom comparison function for props.
   * If provided, this will be used instead of shallow comparison.
   */
  areEqual?: (prevProps: unknown, nextProps: unknown) => boolean;
  /**
   * Display name for the memoized component (for debugging).
   */
  displayName?: string;
}

/**
 * Memoizes a component with optional custom comparison.
 *
 * @param Component - Component to memoize
 * @param options - Memoization options
 * @returns Memoized component
 *
 * @example
 * ```tsx
 * const OptimizedButton = memoizeComponent(Button, {
 *   displayName: 'OptimizedButton',
 *   areEqual: (prev, next) => prev.id === next.id
 * });
 * ```
 */
export function memoizeComponent<P extends object>(
  Component: ComponentType<P>,
  options?: MemoizeOptions
): ComponentType<P> {
  const MemoizedComponent = options?.areEqual
    ? memo(Component, options.areEqual)
    : memo(Component);

  if (options?.displayName) {
    MemoizedComponent.displayName = options.displayName;
  } else if (Component.displayName) {
    MemoizedComponent.displayName = `Memoized(${Component.displayName})`;
  } else if (Component.name) {
    MemoizedComponent.displayName = `Memoized(${Component.name})`;
  }

  return MemoizedComponent;
}

/**
 * Options for lazy component loading.
 */
export interface LazyLoadOptions {
  /**
   * Fallback component to show while loading.
   */
  fallback?: ReactElement | null;
  /**
   * Whether to preload the component.
   */
  preload?: boolean;
}

/**
 * Creates a lazy-loaded component with a fallback.
 *
 * @param importFn - Function that returns a promise resolving to the component
 * @param options - Lazy loading options
 * @returns Lazy component wrapped with Suspense
 *
 * @example
 * ```tsx
 * const LazyComponent = lazyLoadComponent(
 *   () => import('./HeavyComponent'),
 *   { fallback: <Loading /> }
 * );
 * ```
 */
export function lazyLoadComponent<P extends object>(
  importFn: () => Promise<{ default: ComponentType<P> }>,
  options?: LazyLoadOptions
): ComponentType<P> {
  const LazyComponent = lazy(importFn);

  if (options?.preload) {
    // Preload the component
    importFn().catch(() => {
      // Silently handle preload errors
    });
  }

  const WrappedComponent = (props: P) => (
    <Suspense fallback={options?.fallback ?? null}>
      <LazyComponent {...props} />
    </Suspense>
  );

  WrappedComponent.displayName = `LazyLoaded(${LazyComponent.name || 'Component'})`;

  return WrappedComponent as ComponentType<P>;
}

/**
 * Creates a higher-order component that conditionally renders based on a predicate.
 *
 * @param predicate - Function that determines if component should render
 * @param FallbackComponent - Component to render if predicate is false
 * @returns HOC that conditionally renders the component
 *
 * @example
 * ```tsx
 * const ConditionalComponent = conditionalRender(
 *   (props) => props.isVisible,
 *   () => <div>Hidden</div>
 * )(MyComponent);
 * ```
 */
export function conditionalRender<P extends object>(
  predicate: (props: P) => boolean,
  FallbackComponent?: ComponentType<P>
) {
  return (Component: ComponentType<P>): ComponentType<P> => {
    const ConditionalComponent = (props: P) => {
      if (predicate(props)) {
        return <Component {...props} />;
      }

      return FallbackComponent ? <FallbackComponent {...props} /> : null;
    };

    ConditionalComponent.displayName = `Conditional(${Component.displayName || Component.name || 'Component'})`;

    return ConditionalComponent;
  };
}

/**
 * Creates a component that only renders on client side (avoids SSR hydration issues).
 *
 * @param Component - Component to wrap
 * @param fallback - Fallback to render during SSR
 * @returns Client-only component
 *
 * @example
 * ```tsx
 * const ClientOnlyComponent = clientOnly(MyComponent, <div>Loading...</div>);
 * ```
 */
export function clientOnly<P extends object>(
  Component: ComponentType<P>,
  fallback: ReactElement | null = null
): ComponentType<P> {
  const ClientOnlyComponent = (props: P) => {
    const [isMounted, setIsMounted] = React.useState(false);

    React.useEffect(() => {
      setIsMounted(true);
    }, []);

    if (!isMounted) {
      return fallback;
    }

    return <Component {...props} />;
  };

  ClientOnlyComponent.displayName = `ClientOnly(${Component.displayName || Component.name || 'Component'})`;

  return ClientOnlyComponent;
}

