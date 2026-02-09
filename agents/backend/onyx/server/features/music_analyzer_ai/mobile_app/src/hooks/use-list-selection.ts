import { useState, useCallback, useMemo } from 'react';

interface UseListSelectionOptions<T> {
  items: T[];
  keyExtractor: (item: T) => string;
  multiSelect?: boolean;
  initialSelection?: string[];
  onSelectionChange?: (selectedKeys: string[]) => void;
}

export function useListSelection<T>(options: UseListSelectionOptions<T>) {
  const {
    items,
    keyExtractor,
    multiSelect = false,
    initialSelection = [],
    onSelectionChange,
  } = options;

  const [selectedKeys, setSelectedKeys] = useState<string[]>(initialSelection);

  const toggleSelection = useCallback(
    (item: T) => {
      const key = keyExtractor(item);
      setSelectedKeys((prev) => {
        let newSelection: string[];

        if (multiSelect) {
          if (prev.includes(key)) {
            newSelection = prev.filter((k) => k !== key);
          } else {
            newSelection = [...prev, key];
          }
        } else {
          newSelection = prev.includes(key) ? [] : [key];
        }

        onSelectionChange?.(newSelection);
        return newSelection;
      });
    },
    [keyExtractor, multiSelect, onSelectionChange]
  );

  const selectItem = useCallback(
    (item: T) => {
      const key = keyExtractor(item);
      if (!selectedKeys.includes(key)) {
        setSelectedKeys((prev) => {
          const newSelection = multiSelect ? [...prev, key] : [key];
          onSelectionChange?.(newSelection);
          return newSelection;
        });
      }
    },
    [keyExtractor, selectedKeys, multiSelect, onSelectionChange]
  );

  const deselectItem = useCallback(
    (item: T) => {
      const key = keyExtractor(item);
      if (selectedKeys.includes(key)) {
        setSelectedKeys((prev) => {
          const newSelection = prev.filter((k) => k !== key);
          onSelectionChange?.(newSelection);
          return newSelection;
        });
      }
    },
    [keyExtractor, selectedKeys, onSelectionChange]
  );

  const selectAll = useCallback(() => {
    const allKeys = items.map(keyExtractor);
    setSelectedKeys(allKeys);
    onSelectionChange?.(allKeys);
  }, [items, keyExtractor, onSelectionChange]);

  const deselectAll = useCallback(() => {
    setSelectedKeys([]);
    onSelectionChange?.([]);
  }, [onSelectionChange]);

  const isSelected = useCallback(
    (item: T) => {
      const key = keyExtractor(item);
      return selectedKeys.includes(key);
    },
    [keyExtractor, selectedKeys]
  );

  const selectedItems = useMemo(() => {
    return items.filter((item) => isSelected(item));
  }, [items, isSelected]);

  const selectionCount = useMemo(() => selectedKeys.length, [selectedKeys.length]);

  const hasSelection = useMemo(() => selectedKeys.length > 0, [selectedKeys.length]);

  return {
    selectedKeys,
    selectedItems,
    selectionCount,
    hasSelection,
    toggleSelection,
    selectItem,
    deselectItem,
    selectAll,
    deselectAll,
    isSelected,
    setSelection: (keys: string[]) => {
      setSelectedKeys(keys);
      onSelectionChange?.(keys);
    },
  };
}

