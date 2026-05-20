import { useState, useCallback } from "react";
import { useModalState } from "./useModalState";

/**
 * Hook for managing create agent modal state
 * 
 * Provides:
 * - Modal open/close state
 * - Handlers for opening and closing
 * - Convenience methods
 */
export const useCreateAgentModal = () => {
  const modalState = useModalState(false);

  return {
    isOpen: modalState.isOpen,
    open: modalState.open,
    close: modalState.close,
    toggle: modalState.toggle,
    onOpenChange: modalState.onOpenChange,
  };
};



