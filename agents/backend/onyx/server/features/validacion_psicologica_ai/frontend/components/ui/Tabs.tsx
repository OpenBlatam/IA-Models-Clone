/**
 * Tabs component with keyboard navigation
 */

'use client';

import React, { useState, useRef, useEffect } from 'react';
import { cn } from '@/lib/utils/cn';

export interface TabItem {
  id: string;
  label: string;
  content: React.ReactNode;
  disabled?: boolean;
  icon?: React.ReactNode;
}

export interface TabsProps {
  items: TabItem[];
  defaultActiveId?: string;
  onTabChange?: (id: string) => void;
  className?: string;
  variant?: 'default' | 'pills' | 'underline';
  orientation?: 'horizontal' | 'vertical';
}

export const Tabs: React.FC<TabsProps> = ({
  items,
  defaultActiveId,
  onTabChange,
  className,
  variant = 'default',
  orientation = 'horizontal',
}) => {
  const [activeId, setActiveId] = useState(defaultActiveId || items[0]?.id || '');
  const tabRefs = useRef<Record<string, HTMLButtonElement>>({});

  useEffect(() => {
    if (defaultActiveId && items.some((item) => item.id === defaultActiveId)) {
      setActiveId(defaultActiveId);
    }
  }, [defaultActiveId, items]);

  const handleTabChange = (id: string) => {
    if (items.find((item) => item.id === id)?.disabled) {
      return;
    }
    setActiveId(id);
    if (onTabChange) {
      onTabChange(id);
    }
  };

  const handleKeyDown = (event: React.KeyboardEvent, id: string, index: number) => {
    if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
      event.preventDefault();
      const enabledItems = items.filter((item) => !item.disabled);
      const currentIndex = enabledItems.findIndex((item) => item.id === id);
      const direction = event.key === 'ArrowRight' ? 1 : -1;
      const nextIndex = (currentIndex + direction + enabledItems.length) % enabledItems.length;
      const nextId = enabledItems[nextIndex].id;
      handleTabChange(nextId);
      tabRefs.current[nextId]?.focus();
    } else if (event.key === 'Home') {
      event.preventDefault();
      const firstEnabled = items.find((item) => !item.disabled);
      if (firstEnabled) {
        handleTabChange(firstEnabled.id);
        tabRefs.current[firstEnabled.id]?.focus();
      }
    } else if (event.key === 'End') {
      event.preventDefault();
      const lastEnabled = [...items].reverse().find((item) => !item.disabled);
      if (lastEnabled) {
        handleTabChange(lastEnabled.id);
        tabRefs.current[lastEnabled.id]?.focus();
      }
    }
  };

  const activeTab = items.find((item) => item.id === activeId);

  const variantClasses = {
    default: 'border-b-2 border-transparent hover:border-primary',
    pills: 'rounded-full px-4 py-2',
    underline: 'border-b-2 border-transparent hover:border-primary',
  };

  return (
    <div
      className={cn(
        'flex',
        orientation === 'vertical' ? 'flex-row' : 'flex-col',
        className
      )}
    >
      <div
        role="tablist"
        aria-label="Tabs"
        className={cn(
          'flex',
          orientation === 'vertical' ? 'flex-col gap-2' : 'flex-row gap-1 border-b',
          variant === 'pills' && 'gap-2'
        )}
      >
        {items.map((item, index) => {
          const isActive = activeId === item.id;
          const isDisabled = item.disabled;

          return (
            <button
              key={item.id}
              ref={(el) => {
                if (el) {
                  tabRefs.current[item.id] = el;
                }
              }}
              role="tab"
              aria-selected={isActive}
              aria-controls={`tabpanel-${item.id}`}
              aria-disabled={isDisabled}
              tabIndex={isActive ? 0 : -1}
              disabled={isDisabled}
              onClick={() => handleTabChange(item.id)}
              onKeyDown={(e) => handleKeyDown(e, item.id, index)}
              className={cn(
                'flex items-center gap-2 px-4 py-2 text-sm font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2',
                isActive
                  ? variant === 'pills'
                    ? 'bg-primary text-primary-foreground'
                    : 'border-primary text-primary'
                  : 'text-muted-foreground hover:text-foreground',
                isDisabled && 'opacity-50 cursor-not-allowed',
                variantClasses[variant]
              )}
            >
              {item.icon && <span aria-hidden="true">{item.icon}</span>}
              {item.label}
            </button>
          );
        })}
      </div>

      <div
        role="tabpanel"
        id={`tabpanel-${activeId}`}
        aria-labelledby={`tab-${activeId}`}
        className={cn(
          'mt-4',
          orientation === 'vertical' && 'ml-4 flex-1'
        )}
      >
        {activeTab?.content}
      </div>
    </div>
  );
};
