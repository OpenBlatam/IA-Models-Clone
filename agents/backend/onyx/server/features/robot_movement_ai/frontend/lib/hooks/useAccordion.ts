import { useState, useCallback } from 'react';

export interface UseAccordionOptions {
  allowMultiple?: boolean;
  defaultOpen?: string[];
}

export interface UseAccordionReturn {
  openItems: Set<string>;
  toggle: (item: string) => void;
  open: (item: string) => void;
  close: (item: string) => void;
  isOpen: (item: string) => boolean;
  openAll: (items: string[]) => void;
  closeAll: () => void;
}

/**
 * Hook for accordion state management
 */
export function useAccordion(
  options: UseAccordionOptions = {}
): UseAccordionReturn {
  const { allowMultiple = false, defaultOpen = [] } = options;
  const [openItems, setOpenItems] = useState<Set<string>>(
    new Set(defaultOpen)
  );

  const toggle = useCallback(
    (item: string) => {
      setOpenItems((prev) => {
        const next = new Set(prev);
        if (next.has(item)) {
          next.delete(item);
        } else {
          if (!allowMultiple) {
            next.clear();
          }
          next.add(item);
        }
        return next;
      });
    },
    [allowMultiple]
  );

  const open = useCallback((item: string) => {
    setOpenItems((prev) => {
      const next = allowMultiple ? new Set(prev) : new Set();
      next.add(item);
      return next;
    });
  }, [allowMultiple]);

  const close = useCallback((item: string) => {
    setOpenItems((prev) => {
      const next = new Set(prev);
      next.delete(item);
      return next;
    });
  }, []);

  const isOpen = useCallback(
    (item: string) => openItems.has(item),
    [openItems]
  );

  const openAll = useCallback((items: string[]) => {
    if (allowMultiple) {
      setOpenItems(new Set(items));
    }
  }, [allowMultiple]);

  const closeAll = useCallback(() => {
    setOpenItems(new Set());
  }, []);

  return {
    openItems,
    toggle,
    open,
    close,
    isOpen,
    openAll,
    closeAll,
  };
}



