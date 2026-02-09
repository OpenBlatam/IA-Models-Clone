'use client';

import * as DropdownMenuPrimitive from '@radix-ui/react-dropdown-menu';
import { ReactNode } from 'react';
import { cn } from '@/lib/utils';
import { ChevronRight } from 'lucide-react';

interface DropdownMenuProps {
  children: ReactNode;
  trigger: ReactNode;
  align?: 'start' | 'center' | 'end';
}

interface DropdownMenuItemProps {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  className?: string;
}

interface DropdownMenuSeparatorProps {
  className?: string;
}

const DropdownMenu = ({ children, trigger, align = 'end' }: DropdownMenuProps) => {
  return (
    <DropdownMenuPrimitive.Root>
      <DropdownMenuPrimitive.Trigger asChild>{trigger}</DropdownMenuPrimitive.Trigger>
      <DropdownMenuPrimitive.Portal>
        <DropdownMenuPrimitive.Content
          align={align}
          className={cn(
            'z-50 min-w-[8rem] overflow-hidden rounded-md border bg-white p-1 text-gray-950 shadow-md',
            'data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95'
          )}
          sideOffset={5}
        >
          {children}
        </DropdownMenuPrimitive.Content>
      </DropdownMenuPrimitive.Portal>
    </DropdownMenuPrimitive.Root>
  );
};

const DropdownMenuItem = ({ children, onClick, disabled, className }: DropdownMenuItemProps) => {
  return (
    <DropdownMenuPrimitive.Item
      onClick={onClick}
      disabled={disabled}
      className={cn(
        'relative flex cursor-pointer select-none items-center rounded-sm px-2 py-1.5 text-sm outline-none transition-colors',
        'focus:bg-gray-100 focus:text-gray-900',
        'data-[disabled]:pointer-events-none data-[disabled]:opacity-50',
        className
      )}
    >
      {children}
    </DropdownMenuPrimitive.Item>
  );
};

const DropdownMenuSeparator = ({ className }: DropdownMenuSeparatorProps) => {
  return (
    <DropdownMenuPrimitive.Separator
      className={cn('-mx-1 my-1 h-px bg-gray-200', className)}
    />
  );
};

const DropdownMenuLabel = ({ children, className }: { children: ReactNode; className?: string }) => {
  return (
    <DropdownMenuPrimitive.Label
      className={cn('px-2 py-1.5 text-sm font-semibold', className)}
    >
      {children}
    </DropdownMenuPrimitive.Label>
  );
};

export { DropdownMenu, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuLabel };

