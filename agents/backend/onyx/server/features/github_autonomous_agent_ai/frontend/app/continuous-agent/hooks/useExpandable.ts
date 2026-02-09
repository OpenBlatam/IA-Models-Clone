import { useState, useCallback } from "react";

type UseExpandableOptions = {
  readonly initialExpanded?: boolean;
  readonly onToggle?: (expanded: boolean) => void;
};

type UseExpandableReturn = {
  readonly isExpanded: boolean;
  readonly expand: () => void;
  readonly collapse: () => void;
  readonly toggle: () => void;
};

/**
 * Hook for managing expand/collapse state
 * 
 * Features:
 * - Simple expand/collapse state management
 * - Optional callback on toggle
 * - Convenience methods
 */
export const useExpandable = (
  options: UseExpandableOptions = {}
): UseExpandableReturn => {
  const { initialExpanded = false, onToggle } = options;
  const [isExpanded, setIsExpanded] = useState(initialExpanded);

  const expand = useCallback((): void => {
    setIsExpanded(true);
    onToggle?.(true);
  }, [onToggle]);

  const collapse = useCallback((): void => {
    setIsExpanded(false);
    onToggle?.(false);
  }, [onToggle]);

  const toggle = useCallback((): void => {
    setIsExpanded((prev) => {
      const newValue = !prev;
      onToggle?.(newValue);
      return newValue;
    });
  }, [onToggle]);

  return {
    isExpanded,
    expand,
    collapse,
    toggle,
  };
};



