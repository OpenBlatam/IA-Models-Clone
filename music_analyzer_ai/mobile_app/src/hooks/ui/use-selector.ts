import { useState, useCallback, useMemo } from 'react';

export function useSelector<T>(
  items: T[],
  initialSelected: T[] = [],
  keyExtractor?: (item: T) => string
) {
  const [selected, setSelected] = useState<T[]>(initialSelected);

  const getKey = useCallback(
    (item: T) => {
      if (keyExtractor) {
        return keyExtractor(item);
      }
      return String(item);
    },
    [keyExtractor]
  );

  const isSelected = useCallback(
    (item: T) => {
      const key = getKey(item);
      return selected.some((s) => getKey(s) === key);
    },
    [selected, getKey]
  );

  const toggle = useCallback(
    (item: T) => {
      setSelected((prev) => {
        const key = getKey(item);
        const isCurrentlySelected = prev.some((s) => getKey(s) === key);

        if (isCurrentlySelected) {
          return prev.filter((s) => getKey(s) !== key);
        } else {
          return [...prev, item];
        }
      });
    },
    [getKey]
  );

  const select = useCallback(
    (item: T) => {
      if (!isSelected(item)) {
        setSelected((prev) => [...prev, item]);
      }
    },
    [isSelected]
  );

  const deselect = useCallback(
    (item: T) => {
      const key = getKey(item);
      setSelected((prev) => prev.filter((s) => getKey(s) !== key));
    },
    [getKey]
  );

  const selectAll = useCallback(() => {
    setSelected([...items]);
  }, [items]);

  const deselectAll = useCallback(() => {
    setSelected([]);
  }, []);

  const selectedCount = selected.length;
  const allSelected = selectedCount === items.length && items.length > 0;
  const someSelected = selectedCount > 0 && selectedCount < items.length;

  return {
    selected,
    isSelected,
    toggle,
    select,
    deselect,
    selectAll,
    deselectAll,
    selectedCount,
    allSelected,
    someSelected,
  };
}


