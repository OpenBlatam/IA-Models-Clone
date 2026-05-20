import { useEffect, useRef } from 'react';

export const useUpdateEffect = (
  effect: React.EffectCallback,
  deps?: React.DependencyList
): void => {
  const isFirstRender = useRef(true);

  useEffect(() => {
    if (isFirstRender.current) {
      isFirstRender.current = false;
      return;
    }
    return effect();
  }, deps);
};

