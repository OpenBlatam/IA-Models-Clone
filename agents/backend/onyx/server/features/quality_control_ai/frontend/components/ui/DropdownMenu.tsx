'use client';

import { memo, useState, useRef, type ReactNode } from 'react';
import { ChevronDown } from 'lucide-react';
import { useClickOutside } from '@/lib/hooks';
import { cn } from '@/lib/utils';
import { Button } from './Button';

interface DropdownMenuItem {
  label: string;
  onClick: () => void;
  icon?: ReactNode;
  disabled?: boolean;
  divider?: boolean;
}

interface DropdownMenuProps {
  trigger: ReactNode;
  items: DropdownMenuItem[];
  align?: 'left' | 'right';
  className?: string;
}

const DropdownMenu = memo(
  ({ trigger, items, align = 'left', className }: DropdownMenuProps): JSX.Element => {
    const [isOpen, setIsOpen] = useState(false);
    const menuRef = useClickOutside<HTMLDivElement>(() => setIsOpen(false));

    const handleItemClick = (item: DropdownMenuItem): void => {
      if (item.disabled) return;
      item.onClick();
      setIsOpen(false);
    };

    return (
      <div className={cn('relative inline-block', className)} ref={menuRef}>
        <button
          type="button"
          onClick={() => setIsOpen(!isOpen)}
          className="inline-flex items-center"
          aria-expanded={isOpen}
          aria-haspopup="true"
        >
          {trigger}
        </button>

        {isOpen && (
          <div
            className={cn(
              'absolute z-50 mt-2 min-w-[200px] rounded-lg bg-white shadow-lg border border-gray-200 py-1',
              align === 'right' ? 'right-0' : 'left-0'
            )}
            role="menu"
            aria-orientation="vertical"
          >
            {items.map((item, index) => (
              <div key={index}>
                {item.divider && index > 0 && (
                  <div className="my-1 border-t border-gray-200" role="separator" />
                )}
                <button
                  type="button"
                  onClick={() => handleItemClick(item)}
                  disabled={item.disabled}
                  className={cn(
                    'w-full px-4 py-2 text-left text-sm text-gray-700 hover:bg-gray-100 transition-colors flex items-center gap-2',
                    item.disabled && 'opacity-50 cursor-not-allowed'
                  )}
                  role="menuitem"
                >
                  {item.icon && <span className="flex-shrink-0">{item.icon}</span>}
                  <span>{item.label}</span>
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }
);

DropdownMenu.displayName = 'DropdownMenu';

export default DropdownMenu;

