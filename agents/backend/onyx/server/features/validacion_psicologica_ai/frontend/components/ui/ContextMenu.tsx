/**
 * Context menu component
 */

'use client';

import React, { useState, useEffect, useRef } from 'react';
import { cn } from '@/lib/utils/cn';
import { createPortal } from 'react-dom';

export interface ContextMenuItem {
  id: string;
  label: string;
  icon?: React.ReactNode;
  onClick: () => void;
  disabled?: boolean;
  separator?: boolean;
}

export interface ContextMenuProps {
  items: ContextMenuItem[];
  children: React.ReactElement;
  className?: string;
}

export const ContextMenu: React.FC<ContextMenuProps> = ({ items, children, className }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const menuRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
      document.addEventListener('keydown', handleEscape);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('keydown', handleEscape);
    };
  }, [isOpen]);

  const handleContextMenu = (event: React.MouseEvent) => {
    event.preventDefault();
    setPosition({ x: event.clientX, y: event.clientY });
    setIsOpen(true);
  };

  const handleItemClick = (item: ContextMenuItem) => {
    if (!item.disabled) {
      item.onClick();
      setIsOpen(false);
    }
  };

  const handleKeyDown = (event: React.KeyboardEvent, item: ContextMenuItem) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      handleItemClick(item);
    }
  };

  return (
    <>
      {React.cloneElement(children, {
        ref: triggerRef,
        onContextMenu: handleContextMenu,
      })}
      {isOpen &&
        createPortal(
          <div
            ref={menuRef}
            className={cn(
              'fixed z-50 min-w-[200px] bg-popover border rounded-md shadow-lg py-1',
              className
            )}
            style={{
              top: `${position.y}px`,
              left: `${position.x}px`,
            }}
            role="menu"
            aria-label="Menú contextual"
          >
            {items.map((item, index) => {
              if (item.separator) {
                return <div key={item.id} className="my-1 border-t border-border" />;
              }

              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => handleItemClick(item)}
                  onKeyDown={(e) => handleKeyDown(e, item)}
                  disabled={item.disabled}
                  className={cn(
                    'w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-accent transition-colors focus:outline-none focus:bg-accent',
                    item.disabled && 'opacity-50 cursor-not-allowed'
                  )}
                  role="menuitem"
                  tabIndex={0}
                >
                  {item.icon && <span aria-hidden="true">{item.icon}</span>}
                  {item.label}
                </button>
              );
            })}
          </div>,
          document.body
        )}
    </>
  );
};



