import { useEffect } from 'react';

export const useConditionalEffect = (
  effect: React.EffectCallback,
  condition: boolean,
  deps?: React.DependencyList
): void => {
  useEffect(() => {
    if (condition) {
      return effect();
    }
  }, [condition, effect, deps]);
};

