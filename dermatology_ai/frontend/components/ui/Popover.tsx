'use client';

import React, { useState, useRef, useEffect } from 'react';
import { clsx } from 'clsx';
import { Portal } from './Portal';

interface PopoverProps {
  trigger: React.ReactNode;
  content: React.ReactNode;
  placement?: 'top' | 'bottom' | 'left' | 'right';
  className?: string;
  offset?: number;
}

export const Popover: React.FC<PopoverProps> = ({
  trigger,
  content,
  placement = 'bottom',
  className,
  offset = 8,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [position, setPosition] = useState({ top: 0, left: 0 });
  const triggerRef = useRef<HTMLDivElement>(null);
  const popoverRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isOpen && triggerRef.current && popoverRef.current) {
      const triggerRect = triggerRef.current.getBoundingClientRect();
      const popoverRect = popoverRef.current.getBoundingClientRect();

      let top = 0;
      let left = 0;

      switch (placement) {
        case 'top':
          top = triggerRect.top - popoverRect.height - offset;
          left = triggerRect.left + triggerRect.width / 2 - popoverRect.width / 2;
          break;
        case 'bottom':
          top = triggerRect.bottom + offset;
          left = triggerRect.left + triggerRect.width / 2 - popoverRect.width / 2;
          break;
        case 'left':
          top = triggerRect.top + triggerRect.height / 2 - popoverRect.height / 2;
          left = triggerRect.left - popoverRect.width - offset;
          break;
        case 'right':
          top = triggerRect.top + triggerRect.height / 2 - popoverRect.height / 2;
          left = triggerRect.right + offset;
          break;
      }

      // Adjust if out of viewport
      if (top < 0) top = triggerRect.bottom + offset;
      if (left < 0) left = triggerRect.left;
      if (top + popoverRect.height > window.innerHeight) {
        top = window.innerHeight - popoverRect.height - 10;
      }
      if (left + popoverRect.width > window.innerWidth) {
        left = window.innerWidth - popoverRect.width - 10;
      }

      setPosition({ top, left });
    }
  }, [isOpen, placement, offset]);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (
        triggerRef.current &&
        !triggerRef.current.contains(e.target as Node) &&
        popoverRef.current &&
        !popoverRef.current.contains(e.target as Node)
      ) {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen]);

  return (
    <>
      <div ref={triggerRef} onClick={() => setIsOpen(!isOpen)}>
        {trigger}
      </div>
      {isOpen && (
        <Portal>
          <div
            ref={popoverRef}
            className={clsx(
              'absolute z-50 bg-white dark:bg-gray-800 rounded-lg shadow-xl border border-gray-200 dark:border-gray-700 p-4 min-w-[200px]',
              'animate-fade-in',
              className
            )}
            style={{ top: `${position.top}px`, left: `${position.left}px` }}
          >
            {content}
          </div>
        </Portal>
      )}
    </>
  );
};
