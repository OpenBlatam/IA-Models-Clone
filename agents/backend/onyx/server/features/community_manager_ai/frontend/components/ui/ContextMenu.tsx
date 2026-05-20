'use client';

import { ReactNode } from 'react';
import * as ContextMenuPrimitive from '@radix-ui/react-context-menu';
import { cn } from '@/lib/utils';
import { Check, ChevronRight } from 'lucide-react';

interface ContextMenuItemType {
  label: string;
  icon?: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  checked?: boolean;
  separator?: boolean;
  submenu?: ContextMenuItemType[];
}

interface ContextMenuProps {
  children: ReactNode;
  items: ContextMenuItemType[];
  className?: string;
}

const ContextMenuItem = ({
  item,
  onSelect,
}: {
  item: ContextMenuItemType;
  onSelect: () => void;
}) => {
  if (item.separator) {
    return <ContextMenuPrimitive.Separator className="my-1 h-px bg-gray-200 dark:bg-gray-700" />;
  }

  if (item.submenu) {
    return (
      <ContextMenuPrimitive.Sub>
        <ContextMenuPrimitive.SubTrigger
          className={cn(
            'flex items-center gap-2 rounded px-2 py-1.5 text-sm outline-none',
            'focus:bg-gray-100 dark:focus:bg-gray-700',
            item.disabled && 'opacity-50 cursor-not-allowed'
          )}
          disabled={item.disabled}
        >
          {item.icon}
          <span>{item.label}</span>
          <ChevronRight className="ml-auto h-4 w-4" />
        </ContextMenuPrimitive.SubTrigger>
        <ContextMenuPrimitive.Portal>
          <ContextMenuPrimitive.SubContent
            className={cn(
              'z-50 min-w-[8rem] rounded-md border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-1 shadow-lg',
              'animate-in fade-in-0 zoom-in-95'
            )}
          >
            {item.submenu.map((subItem, index) => (
              <ContextMenuItem key={index} item={subItem} onSelect={subItem.onClick || (() => {})} />
            ))}
          </ContextMenuPrimitive.SubContent>
        </ContextMenuPrimitive.Portal>
      </ContextMenuPrimitive.Sub>
    );
  }

  return (
    <ContextMenuPrimitive.Item
      className={cn(
        'flex items-center gap-2 rounded px-2 py-1.5 text-sm outline-none cursor-pointer',
        'focus:bg-gray-100 dark:focus:bg-gray-700',
        item.disabled && 'opacity-50 cursor-not-allowed'
      )}
      disabled={item.disabled}
      onSelect={onSelect}
    >
      {item.icon}
      <span className="flex-1">{item.label}</span>
      {item.checked && <Check className="h-4 w-4" />}
    </ContextMenuPrimitive.Item>
  );
};

export const ContextMenu = ({ children, items, className }: ContextMenuProps) => {
  return (
    <ContextMenuPrimitive.Root>
      <ContextMenuPrimitive.Trigger asChild className={className}>
        {children}
      </ContextMenuPrimitive.Trigger>
      <ContextMenuPrimitive.Portal>
        <ContextMenuPrimitive.Content
          className={cn(
            'z-50 min-w-[8rem] rounded-md border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-1 shadow-lg',
            'animate-in fade-in-0 zoom-in-95'
          )}
        >
          {items.map((item, index) => (
            <ContextMenuItem
              key={index}
              item={item}
              onSelect={item.onClick || (() => {})}
            />
          ))}
        </ContextMenuPrimitive.Content>
      </ContextMenuPrimitive.Portal>
    </ContextMenuPrimitive.Root>
  );
};

