import { useState, useCallback } from 'react';

export function useAccordion(initialOpenItems: string[] = []) {
  const [openItems, setOpenItems] = useState<string[]>(initialOpenItems);

  const isOpen = useCallback(
    (itemId: string) => {
      return openItems.includes(itemId);
    },
    [openItems]
  );

  const toggle = useCallback((itemId: string) => {
    setOpenItems((prev) => {
      if (prev.includes(itemId)) {
        return prev.filter((id) => id !== itemId);
      } else {
        return [...prev, itemId];
      }
    });
  }, []);

  const open = useCallback((itemId: string) => {
    setOpenItems((prev) => {
      if (!prev.includes(itemId)) {
        return [...prev, itemId];
      }
      return prev;
    });
  }, []);

  const close = useCallback((itemId: string) => {
    setOpenItems((prev) => prev.filter((id) => id !== itemId));
  }, []);

  const openAll = useCallback((itemIds: string[]) => {
    setOpenItems([...itemIds]);
  }, []);

  const closeAll = useCallback(() => {
    setOpenItems([]);
  }, []);

  return {
    openItems,
    isOpen,
    toggle,
    open,
    close,
    openAll,
    closeAll,
  };
}


