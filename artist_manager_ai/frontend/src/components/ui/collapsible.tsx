'use client';

import * as CollapsiblePrimitive from '@radix-ui/react-collapsible';
import { ChevronDown } from 'lucide-react';
import { cn } from '@/lib/utils';
import { ReactNode, useState } from 'react';

interface CollapsibleProps {
  children: ReactNode;
  defaultOpen?: boolean;
  className?: string;
}

interface CollapsibleTriggerProps {
  children: ReactNode;
  className?: string;
}

interface CollapsibleContentProps {
  children: ReactNode;
  className?: string;
}

const Collapsible = ({ children, defaultOpen = false, className }: CollapsibleProps) => {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <CollapsiblePrimitive.Root open={open} onOpenChange={setOpen} className={cn('w-full', className)}>
      {children}
    </CollapsiblePrimitive.Root>
  );
};

const CollapsibleTrigger = ({ children, className }: CollapsibleTriggerProps) => {
  return (
    <CollapsiblePrimitive.Trigger
      className={cn(
        'flex w-full items-center justify-between py-2 font-medium transition-all hover:underline',
        '[&[data-state=open]>svg]:rotate-180',
        className
      )}
    >
      {children}
      <ChevronDown className="h-4 w-4 shrink-0 transition-transform duration-200" />
    </CollapsiblePrimitive.Trigger>
  );
};

const CollapsibleContent = ({ children, className }: CollapsibleContentProps) => {
  return (
    <CollapsiblePrimitive.Content
      className={cn(
        'overflow-hidden text-sm transition-all data-[state=closed]:animate-collapsible-up data-[state=open]:animate-collapsible-down',
        className
      )}
    >
      <div className="pb-2 pt-0">{children}</div>
    </CollapsiblePrimitive.Content>
  );
};

export { Collapsible, CollapsibleTrigger, CollapsibleContent };

