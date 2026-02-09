import { useMemo, useRef, DependencyList } from 'react';
import { shallowEqual } from './render-optimization';

export function useShallowMemo<T>(
  factory: () => T,
  deps: DependencyList
): T {
  const ref = useRef<{ deps: DependencyList; value: T }>();
  
  if (!ref.current || !shallowEqual(ref.current.deps, deps)) {
    ref.current = {
      deps,
      value: factory(),
    };
  }
  
  return ref.current.value;
}

export function useDeepMemo<T>(
  factory: () => T,
  deps: DependencyList
): T {
  return useMemo(factory, deps);
}

export function useStableRef<T>(value: T): React.MutableRefObject<T> {
  const ref = useRef<T>(value);
  ref.current = value;
  return ref as React.MutableRefObject<T>;
}

