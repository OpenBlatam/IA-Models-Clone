'use client';

import { ReactNode } from 'react';
import * as Tabs from '@radix-ui/react-tabs';
import { cn } from '@/lib/utils';

interface TabsProps {
  defaultValue: string;
  children: ReactNode;
  className?: string;
}

interface TabsListProps {
  children: ReactNode;
  className?: string;
}

interface TabsTriggerProps {
  value: string;
  children: ReactNode;
  className?: string;
}

interface TabsContentProps {
  value: string;
  children: ReactNode;
  className?: string;
}

const TabsRoot = ({ defaultValue, children, className }: TabsProps) => {
  return (
    <Tabs.Root defaultValue={defaultValue} className={cn('w-full', className)}>
      {children}
    </Tabs.Root>
  );
};

const TabsList = ({ children, className }: TabsListProps) => {
  return (
    <Tabs.List
      className={cn(
        'inline-flex h-10 items-center justify-center rounded-lg bg-gray-100 p-1 text-gray-600',
        className
      )}
    >
      {children}
    </Tabs.List>
  );
};

const TabsTrigger = ({ value, children, className }: TabsTriggerProps) => {
  return (
    <Tabs.Trigger
      value={value}
      className={cn(
        'inline-flex items-center justify-center whitespace-nowrap rounded-md px-3 py-1.5 text-sm font-medium ring-offset-white transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 data-[state=active]:bg-white data-[state=active]:text-gray-900 data-[state=active]:shadow-sm',
        className
      )}
      tabIndex={0}
    >
      {children}
    </Tabs.Trigger>
  );
};

const TabsContent = ({ value, children, className }: TabsContentProps) => {
  return (
    <Tabs.Content
      value={value}
      className={cn(
        'mt-2 ring-offset-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2',
        className
      )}
    >
      {children}
    </Tabs.Content>
  );
};

export { TabsRoot as Tabs, TabsList, TabsTrigger, TabsContent };

