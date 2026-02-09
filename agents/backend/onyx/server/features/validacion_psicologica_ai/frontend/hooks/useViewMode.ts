/**
 * Custom hook for view mode persistence
 */

import { useState, useEffect } from 'react';
import { useLocalStorage } from './useLocalStorage';
import type { ViewType } from '@/components/features/ViewSwitcher';

export const useViewMode = (defaultMode: ViewType = 'list') => {
  const [storedMode, setStoredMode] = useLocalStorage<ViewType>('validation-view-mode', defaultMode);
  const [viewMode, setViewMode] = useState<ViewType>(storedMode);

  useEffect(() => {
    setViewMode(storedMode);
  }, [storedMode]);

  const handleViewChange = (mode: ViewType) => {
    setViewMode(mode);
    setStoredMode(mode);
  };

  return [viewMode, handleViewChange] as const;
};


