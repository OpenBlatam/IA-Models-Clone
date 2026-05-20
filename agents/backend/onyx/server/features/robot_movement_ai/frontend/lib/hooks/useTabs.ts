import { useState, useCallback } from 'react';

export interface UseTabsOptions {
  initialTab?: string;
  onChange?: (tab: string) => void;
}

export interface UseTabsReturn {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  isActive: (tab: string) => boolean;
}

/**
 * Hook for tab navigation
 */
export function useTabs(
  tabs: string[],
  options: UseTabsOptions = {}
): UseTabsReturn {
  const { initialTab, onChange } = options;
  const [activeTab, setActiveTabState] = useState(
    initialTab || tabs[0] || ''
  );

  const setActiveTab = useCallback(
    (tab: string) => {
      if (tabs.includes(tab)) {
        setActiveTabState(tab);
        if (onChange) {
          onChange(tab);
        }
      }
    },
    [tabs, onChange]
  );

  const isActive = useCallback(
    (tab: string) => activeTab === tab,
    [activeTab]
  );

  return {
    activeTab,
    setActiveTab,
    isActive,
  };
}



