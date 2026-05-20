'use client';

import { ReactNode } from 'react';
import * as AccordionPrimitive from '@radix-ui/react-accordion';
import { ChevronDown } from 'lucide-react';
import { cn } from '@/lib/utils';

interface AccordionItem {
  value: string;
  trigger: ReactNode;
  content: ReactNode;
}

interface AccordionProps {
  items: AccordionItem[];
  type?: 'single' | 'multiple';
  defaultValue?: string | string[];
  value?: string | string[];
  onValueChange?: (value: string | string[]) => void;
  className?: string;
  collapsible?: boolean;
}

export const Accordion = ({
  items,
  type = 'single',
  defaultValue,
  value,
  onValueChange,
  className,
  collapsible = true,
}: AccordionProps) => {
  if (type === 'multiple') {
    return (
      <AccordionPrimitive.Root
        type="multiple"
        defaultValue={defaultValue as string[]}
        value={value as string[]}
        onValueChange={onValueChange as (value: string[]) => void}
        className={cn('w-full space-y-2', className)}
      >
        {items.map((item) => (
          <AccordionPrimitive.Item
            key={item.value}
            value={item.value}
            className="rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden"
          >
            <AccordionPrimitive.Header>
              <AccordionPrimitive.Trigger
                className={cn(
                  'flex w-full items-center justify-between px-4 py-3 text-left',
                  'hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors',
                  'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500'
                )}
              >
                {item.trigger}
                <ChevronDown className="h-4 w-4 text-gray-500 dark:text-gray-400 transition-transform data-[state=open]:rotate-180" />
              </AccordionPrimitive.Trigger>
            </AccordionPrimitive.Header>
            <AccordionPrimitive.Content className="overflow-hidden data-[state=closed]:animate-accordion-up data-[state=open]:animate-accordion-down">
              <div className="px-4 pb-4 pt-0">{item.content}</div>
            </AccordionPrimitive.Content>
          </AccordionPrimitive.Item>
        ))}
      </AccordionPrimitive.Root>
    );
  }

  return (
    <AccordionPrimitive.Root
      type="single"
      defaultValue={defaultValue as string}
      value={value as string}
      onValueChange={onValueChange as (value: string) => void}
      collapsible={collapsible}
      className={cn('w-full space-y-2', className)}
    >
      {items.map((item) => (
        <AccordionPrimitive.Item
          key={item.value}
          value={item.value}
          className="rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden"
        >
          <AccordionPrimitive.Header>
            <AccordionPrimitive.Trigger
              className={cn(
                'flex w-full items-center justify-between px-4 py-3 text-left',
                'hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors',
                'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500'
              )}
            >
              {item.trigger}
              <ChevronDown className="h-4 w-4 text-gray-500 dark:text-gray-400 transition-transform data-[state=open]:rotate-180" />
            </AccordionPrimitive.Trigger>
          </AccordionPrimitive.Header>
          <AccordionPrimitive.Content className="overflow-hidden data-[state=closed]:animate-accordion-up data-[state=open]:animate-accordion-down">
            <div className="px-4 pb-4 pt-0">{item.content}</div>
          </AccordionPrimitive.Content>
        </AccordionPrimitive.Item>
      ))}
    </AccordionPrimitive.Root>
  );
};
