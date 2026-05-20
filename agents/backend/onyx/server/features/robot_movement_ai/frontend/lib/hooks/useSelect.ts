import { useState, useCallback, useMemo } from 'react';

export interface UseSelectOptions<T> {
  multiple?: boolean;
  initialValue?: T | T[];
  compareFn?: (a: T, b: T) => boolean;
}

/**
 * Hook for selection (single or multiple)
 */
export function useSelect<T>(
  options: UseSelectOptions<T> = {}
): {
  selected: T | T[] | null;
  isSelected: (item: T) => boolean;
  select: (item: T) => void;
  deselect: (item: T) => void;
  toggle: (item: T) => void;
  clear: () => void;
  selectAll: (items: T[]) => void;
} {
  const {
    multiple = false,
    initialValue,
    compareFn = (a, b) => a === b,
  } = options;

  const [selected, setSelected] = useState<T | T[] | null>(
    initialValue ?? (multiple ? [] : null)
  );

  const isSelected = useCallback(
    (item: T): boolean => {
      if (multiple) {
        return (selected as T[]).some((s) => compareFn(s, item));
      }
      return selected !== null && compareFn(selected as T, item);
    },
    [selected, multiple, compareFn]
  );

  const select = useCallback(
    (item: T) => {
      if (multiple) {
        setSelected((prev) => {
          const prevArray = (prev as T[]) || [];
          if (prevArray.some((s) => compareFn(s, item))) {
            return prevArray;
          }
          return [...prevArray, item];
        });
      } else {
        setSelected(item);
      }
    },
    [multiple, compareFn]
  );

  const deselect = useCallback(
    (item: T) => {
      if (multiple) {
        setSelected((prev) => {
          const prevArray = (prev as T[]) || [];
          return prevArray.filter((s) => !compareFn(s, item));
        });
      } else {
        setSelected(null);
      }
    },
    [multiple, compareFn]
  );

  const toggle = useCallback(
    (item: T) => {
      if (isSelected(item)) {
        deselect(item);
      } else {
        select(item);
      }
    },
    [isSelected, select, deselect]
  );

  const clear = useCallback(() => {
    setSelected(multiple ? [] : null);
  }, [multiple]);

  const selectAll = useCallback(
    (items: T[]) => {
      if (multiple) {
        setSelected(items);
      }
    },
    [multiple]
  );

  return {
    selected,
    isSelected,
    select,
    deselect,
    toggle,
    clear,
    selectAll,
  };
}



