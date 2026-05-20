import { useState, useCallback } from 'react';

interface UseTabsOptions {
  defaultValue?: string;
}

interface UseTabsReturn {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  isActive: (tab: string) => boolean;
}

export const useTabs = ({ defaultValue = '' }: UseTabsOptions = {}): UseTabsReturn => {
  const [activeTab, setActiveTab] = useState(defaultValue);

  const isActive = useCallback((tab: string): boolean => {
    return activeTab === tab;
  }, [activeTab]);

  return {
    activeTab,
    setActiveTab,
    isActive,
  };
};

