'use client';

import { ReactNode } from 'react';
import * as TabsPrimitive from '@radix-ui/react-tabs';
import { cn } from '@/lib/utils';

interface TabsProps {
  defaultValue?: string;
  value?: string;
  onValueChange?: (value: string) => void;
  children: ReactNode;
  className?: string;
  orientation?: 'horizontal' | 'vertical';
}

interface TabsListProps {
  children: ReactNode;
  className?: string;
}

interface TabsTriggerProps {
  value: string;
  children: ReactNode;
  className?: string;
  disabled?: boolean;
}

interface TabsContentProps {
  value: string;
  children: ReactNode;
  className?: string;
}

export const Tabs = ({ defaultValue, value, onValueChange, children, className, orientation = 'horizontal' }: TabsProps) => {
  return (
    <TabsPrimitive.Root
      defaultValue={defaultValue}
      value={value}
      onValueChange={onValueChange}
      orientation={orientation}
      className={cn('w-full', className)}
    >
      {children}
    </TabsPrimitive.Root>
  );
};

export const TabsList = ({ children, className }: TabsListProps) => {
  return (
    <TabsPrimitive.List
      className={cn(
        'inline-flex items-center justify-center rounded-lg bg-gray-100 dark:bg-gray-800 p-1',
        className
      )}
    >
      {children}
    </TabsPrimitive.List>
  );
};

export const TabsTrigger = ({ value, children, className, disabled }: TabsTriggerProps) => {
  return (
    <TabsPrimitive.Trigger
      value={value}
      disabled={disabled}
      className={cn(
        'inline-flex items-center justify-center rounded-md px-3 py-1.5 text-sm font-medium transition-all',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500',
        'disabled:pointer-events-none disabled:opacity-50',
        'data-[state=active]:bg-white dark:data-[state=active]:bg-gray-900',
        'data-[state=active]:text-gray-900 dark:data-[state=active]:text-gray-100',
        'data-[state=inactive]:text-gray-600 dark:data-[state=inactive]:text-gray-400',
        className
      )}
    >
      {children}
    </TabsPrimitive.Trigger>
  );
};

export const TabsContent = ({ value, children, className }: TabsContentProps) => {
  return (
    <TabsPrimitive.Content
      value={value}
      className={cn(
        'mt-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500',
        className
      )}
    >
      {children}
    </TabsPrimitive.Content>
  );
};
