import { useRef } from 'react';

export function useShallowEqual<T>(value: T): T {
  const ref = useRef<{ value: T; deps: unknown[] }>({
    value,
    deps: [],
  });

  const deps = Array.isArray(value) ? value : Object.values(value as object);
  const depsChanged =
    ref.current.deps.length !== deps.length ||
    deps.some((dep, i) => dep !== ref.current.deps[i]);

  if (depsChanged) {
    ref.current = { value, deps };
  }

  return ref.current.value;
}

