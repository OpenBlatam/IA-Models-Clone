/**
 * Custom hook for view type persistence (list, table, timeline)
 */

import { useState, useEffect } from 'react';
import { useLocalStorage } from './useLocalStorage';
import type { ViewType } from '@/components/features/ViewSwitcher';

export const useViewType = (defaultType: ViewType = 'list') => {
  const [storedType, setStoredType] = useLocalStorage<ViewType>('validation-view-type', defaultType);
  const [viewType, setViewType] = useState<ViewType>(storedType);

  useEffect(() => {
    setViewType(storedType);
  }, [storedType]);

  const handleViewChange = (type: ViewType) => {
    setViewType(type);
    setStoredType(type);
  };

  return [viewType, handleViewChange] as const;
};




