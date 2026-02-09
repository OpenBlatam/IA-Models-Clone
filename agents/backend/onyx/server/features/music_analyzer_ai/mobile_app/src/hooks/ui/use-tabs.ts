import { useState, useCallback } from 'react';

export function useTabs<T extends string>(
  defaultTab: T,
  tabs: T[]
) {
  const [activeTab, setActiveTab] = useState<T>(defaultTab);

  const changeTab = useCallback((tab: T) => {
    if (tabs.includes(tab)) {
      setActiveTab(tab);
    }
  }, [tabs]);

  const nextTab = useCallback(() => {
    const currentIndex = tabs.indexOf(activeTab);
    const nextIndex = (currentIndex + 1) % tabs.length;
    setActiveTab(tabs[nextIndex]);
  }, [activeTab, tabs]);

  const previousTab = useCallback(() => {
    const currentIndex = tabs.indexOf(activeTab);
    const prevIndex = currentIndex === 0 ? tabs.length - 1 : currentIndex - 1;
    setActiveTab(tabs[prevIndex]);
  }, [activeTab, tabs]);

  return {
    activeTab,
    changeTab,
    nextTab,
    previousTab,
    tabs,
  };
}


