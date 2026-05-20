"use client";

import { useState, useCallback } from "react";

/**
 * Custom hook for managing modal open/close state
 * 
 * @param initialOpen - Initial open state (default: false)
 * @returns Object with open state and handlers
 */
export const useModalState = (initialOpen = false) => {
  const [isOpen, setIsOpen] = useState(initialOpen);

  const open = useCallback((): void => {
    setIsOpen(true);
  }, []);

  const close = useCallback((): void => {
    setIsOpen(false);
  }, []);

  const toggle = useCallback((): void => {
    setIsOpen((prev) => !prev);
  }, []);

  const onOpenChange = useCallback((open: boolean): void => {
    setIsOpen(open);
  }, []);

  return {
    isOpen,
    open,
    close,
    toggle,
    setIsOpen,
    onOpenChange,
  };
};

