'use client';

import { ReactNode } from 'react';
import * as PopoverPrimitive from '@radix-ui/react-popover';
import { cn } from '@/lib/utils';
import { X } from 'lucide-react';

interface PopoverProps {
  trigger: ReactNode;
  content: ReactNode;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  side?: 'top' | 'right' | 'bottom' | 'left';
  align?: 'start' | 'center' | 'end';
  showCloseButton?: boolean;
  className?: string;
}

export const Popover = ({
  trigger,
  content,
  open,
  onOpenChange,
  side = 'bottom',
  align = 'center',
  showCloseButton = false,
  className,
}: PopoverProps) => {
  return (
    <PopoverPrimitive.Root open={open} onOpenChange={onOpenChange}>
      <PopoverPrimitive.Trigger asChild>{trigger}</PopoverPrimitive.Trigger>
      <PopoverPrimitive.Portal>
        <PopoverPrimitive.Content
          side={side}
          align={align}
          sideOffset={5}
          className={cn(
            'z-50 w-72 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4 shadow-lg',
            'animate-in fade-in-0 zoom-in-95',
            className
          )}
        >
          {showCloseButton && (
            <PopoverPrimitive.Close className="absolute right-2 top-2 rounded p-1 text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700">
              <X className="h-4 w-4" />
            </PopoverPrimitive.Close>
          )}
          {content}
        </PopoverPrimitive.Content>
      </PopoverPrimitive.Portal>
    </PopoverPrimitive.Root>
  );
};



