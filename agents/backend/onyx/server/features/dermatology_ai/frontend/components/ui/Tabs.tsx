'use client';

import React, { useState, ReactNode, memo, useCallback, useMemo } from 'react';
import { clsx } from 'clsx';

interface Tab {
  id: string;
  label: string;
  content: ReactNode;
  icon?: React.ReactNode;
  badge?: number | string;
  disabled?: boolean;
}

interface TabsProps {
  tabs: Tab[];
  defaultTab?: string;
  className?: string;
  variant?: 'default' | 'pills' | 'underline';
  orientation?: 'horizontal' | 'vertical';
  onTabChange?: (tabId: string) => void;
}

export const Tabs: React.FC<TabsProps> = memo(({
  tabs,
  defaultTab,
  className,
  variant = 'default',
  orientation = 'horizontal',
  onTabChange,
}) => {
  const [activeTab, setActiveTab] = useState(defaultTab || tabs[0]?.id);

  const handleTabChange = useCallback((tabId: string) => {
    setActiveTab(tabId);
    if (onTabChange) {
      onTabChange(tabId);
    }
  }, [onTabChange]);

  const activeTabContent = useMemo(
    () => tabs.find((tab) => tab.id === activeTab)?.content,
    [tabs, activeTab]
  );

  const isVertical = orientation === 'vertical';

  if (variant === 'pills') {
    return (
      <div className={clsx('w-full', isVertical && 'flex gap-4', className)}>
        <nav
          className={clsx(
            'flex',
            isVertical ? 'flex-col space-y-2' : 'space-x-2'
          )}
          aria-label="Tabs"
        >
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => !tab.disabled && handleTabChange(tab.id)}
              disabled={tab.disabled}
              className={clsx(
                'px-4 py-2 rounded-lg font-medium text-sm transition-colors',
                activeTab === tab.id
                  ? 'bg-primary-600 text-white'
                  : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700',
                tab.disabled && 'opacity-50 cursor-not-allowed',
                'flex items-center space-x-2'
              )}
            >
              {tab.icon && <span>{tab.icon}</span>}
              <span>{tab.label}</span>
              {tab.badge && (
                <span className="px-2 py-0.5 text-xs bg-white/20 dark:bg-gray-900/20 rounded-full">
                  {tab.badge}
                </span>
              )}
            </button>
          ))}
        </nav>
        <div className={clsx(isVertical ? 'flex-1' : 'mt-4')}>
          {activeTabContent}
        </div>
      </div>
    );
  }

  return (
    <div className={clsx('w-full', isVertical && 'flex gap-4', className)}>
      <div
        className={clsx(
          isVertical
            ? 'border-r border-gray-200 dark:border-gray-700'
            : 'border-b border-gray-200 dark:border-gray-700'
        )}
      >
        <nav
          className={clsx(
            'flex',
            isVertical ? 'flex-col space-y-1' : 'space-x-8'
          )}
          aria-label="Tabs"
        >
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => !tab.disabled && handleTabChange(tab.id)}
              disabled={tab.disabled}
              className={clsx(
                isVertical ? 'py-2 px-4 border-r-2' : 'py-4 px-1 border-b-2',
                'font-medium text-sm transition-colors',
                activeTab === tab.id
                  ? isVertical
                    ? 'border-primary-500 text-primary-600 dark:text-primary-400'
                    : 'border-primary-500 text-primary-600 dark:text-primary-400'
                  : isVertical
                  ? 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-300'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-300',
                tab.disabled && 'opacity-50 cursor-not-allowed',
                'flex items-center space-x-2'
              )}
            >
              {tab.icon && <span>{tab.icon}</span>}
              <span>{tab.label}</span>
              {tab.badge && (
                <span className="ml-2 px-2 py-0.5 text-xs bg-primary-100 dark:bg-primary-900 text-primary-800 dark:text-primary-200 rounded-full">
                  {tab.badge}
                </span>
              )}
            </button>
          ))}
        </nav>
      </div>
      <div className={clsx(isVertical ? 'flex-1' : 'mt-4')}>
        {activeTabContent}
      </div>
    </div>
  );
});

Tabs.displayName = 'Tabs';

