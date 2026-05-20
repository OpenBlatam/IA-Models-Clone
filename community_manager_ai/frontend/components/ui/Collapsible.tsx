'use client';

import { ReactNode } from 'react';
import * as CollapsiblePrimitive from '@radix-ui/react-collapsible';
import { ChevronDown } from 'lucide-react';
import { cn } from '@/lib/utils';
import { motion, AnimatePresence } from 'framer-motion';

interface CollapsibleProps {
  trigger: ReactNode;
  children: ReactNode;
  defaultOpen?: boolean;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  className?: string;
}

export const Collapsible = ({
  trigger,
  children,
  defaultOpen = false,
  open,
  onOpenChange,
  className,
}: CollapsibleProps) => {
  return (
    <CollapsiblePrimitive.Root
      defaultOpen={defaultOpen}
      open={open}
      onOpenChange={onOpenChange}
      className={cn('w-full', className)}
    >
      <CollapsiblePrimitive.Trigger asChild className="w-full">
        <div className="flex items-center justify-between cursor-pointer">
          {trigger}
          <ChevronDown className="h-4 w-4 text-gray-500 dark:text-gray-400 transition-transform data-[state=open]:rotate-180" />
        </div>
      </CollapsiblePrimitive.Trigger>
      <CollapsiblePrimitive.Content>
        <AnimatePresence>
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="pt-4">{children}</div>
          </motion.div>
        </AnimatePresence>
      </CollapsiblePrimitive.Content>
    </CollapsiblePrimitive.Root>
  );
};



