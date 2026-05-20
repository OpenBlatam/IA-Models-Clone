/**
 * Custom hook for modal/dialog state management.
 * Provides convenient methods for opening, closing, and toggling modals.
 */

import { useState, useCallback } from 'react';

/**
 * Return type for useModal hook.
 */
export interface UseModalReturn {
  isOpen: boolean;
  open: () => void;
  close: () => void;
  toggle: () => void;
}

/**
 * Custom hook for modal state management.
 * Provides convenient methods for modal control.
 *
 * @param initialOpen - Initial open state (default: false)
 * @returns Modal state and handlers
 */
export function useModal(initialOpen: boolean = false): UseModalReturn {
  const [isOpen, setIsOpen] = useState(initialOpen);

  const open = useCallback(() => {
    setIsOpen(true);
  }, []);

  const close = useCallback(() => {
    setIsOpen(false);
  }, []);

  const toggle = useCallback(() => {
    setIsOpen((prev) => !prev);
  }, []);

  return {
    isOpen,
    open,
    close,
    toggle,
  };
}

