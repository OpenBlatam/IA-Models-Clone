import { useState, useCallback } from 'react';

export interface UseBottomSheetOptions {
  initialSnapPoint?: number;
  snapPoints?: number[];
}

export function useBottomSheet(options: UseBottomSheetOptions = {}) {
  const { initialSnapPoint = 0, snapPoints = [0, 0.5, 1] } = options;
  const [isOpen, setIsOpen] = useState(false);
  const [currentSnapPoint, setCurrentSnapPoint] = useState(initialSnapPoint);

  const open = useCallback(
    (snapPoint?: number) => {
      setIsOpen(true);
      if (snapPoint !== undefined) {
        setCurrentSnapPoint(snapPoint);
      }
    },
    []
  );

  const close = useCallback(() => {
    setIsOpen(false);
    setCurrentSnapPoint(initialSnapPoint);
  }, [initialSnapPoint]);

  const toggle = useCallback(() => {
    if (isOpen) {
      close();
    } else {
      open();
    }
  }, [isOpen, open, close]);

  const snapTo = useCallback(
    (index: number) => {
      if (index >= 0 && index < snapPoints.length) {
        setCurrentSnapPoint(index);
      }
    },
    [snapPoints]
  );

  return {
    isOpen,
    currentSnapPoint,
    snapPoints,
    open,
    close,
    toggle,
    snapTo,
  };
}


