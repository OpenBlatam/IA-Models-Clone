import { useEffect, useRef } from 'react';

export const useUpdateEffect = (effect: React.EffectCallback, deps?: React.DependencyList): void => {
  const isFirst = useRef(true);

  useEffect(() => {
    if (isFirst.current) {
      isFirst.current = false;
      return;
    }

    return effect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
};



