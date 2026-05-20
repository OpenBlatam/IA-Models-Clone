'use client';

import * as AccordionPrimitive from '@radix-ui/react-accordion';
import { ChevronDown } from 'lucide-react';
import { cn } from '@/lib/utils';
import { ReactNode } from 'react';

interface AccordionProps {
  type?: 'single' | 'multiple';
  defaultValue?: string | string[];
  children: ReactNode;
  className?: string;
}

interface AccordionItemProps {
  value: string;
  children: ReactNode;
  className?: string;
}

interface AccordionTriggerProps {
  children: ReactNode;
  className?: string;
}

interface AccordionContentProps {
  children: ReactNode;
  className?: string;
}

const Accordion = ({ type = 'single', defaultValue, children, className }: AccordionProps) => {
  return (
    <AccordionPrimitive.Root
      type={type}
      defaultValue={defaultValue}
      collapsible
      className={cn('w-full', className)}
    >
      {children}
    </AccordionPrimitive.Root>
  );
};

const AccordionItem = ({ value, children, className }: AccordionItemProps) => {
  return (
    <AccordionPrimitive.Item value={value} className={cn('border-b', className)}>
      {children}
    </AccordionPrimitive.Item>
  );
};

const AccordionTrigger = ({ children, className }: AccordionTriggerProps) => {
  return (
    <AccordionPrimitive.Header className="flex">
      <AccordionPrimitive.Trigger
        className={cn(
          'flex flex-1 items-center justify-between py-4 font-medium transition-all hover:underline',
          '[&[data-state=open]>svg]:rotate-180',
          className
        )}
      >
        {children}
        <ChevronDown className="h-4 w-4 shrink-0 transition-transform duration-200" />
      </AccordionPrimitive.Trigger>
    </AccordionPrimitive.Header>
  );
};

const AccordionContent = ({ children, className }: AccordionContentProps) => {
  return (
    <AccordionPrimitive.Content
      className={cn(
        'overflow-hidden text-sm transition-all data-[state=closed]:animate-accordion-up data-[state=open]:animate-accordion-down',
        className
      )}
    >
      <div className="pb-4 pt-0">{children}</div>
    </AccordionPrimitive.Content>
  );
};

export { Accordion, AccordionItem, AccordionTrigger, AccordionContent };

