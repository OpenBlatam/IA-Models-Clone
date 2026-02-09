/**
 * Popover component
 */

'use client';

import React, { useState, useRef, useEffect } from 'react';
import { cn } from '@/lib/utils/cn';
import { createPortal } from 'react-dom';
import { X } from 'lucide-react';
import { Button } from './Button';

export interface PopoverProps {
  trigger: React.ReactElement;
  content: React.ReactNode;
  isOpen?: boolean;
  onOpenChange?: (open: boolean) => void;
  position?: 'top' | 'bottom' | 'left' | 'right';
  align?: 'start' | 'center' | 'end';
  className?: string;
}

export const Popover: React.FC<PopoverProps> = ({
  trigger,
  content,
  isOpen: controlledOpen,
  onOpenChange,
  position = 'bottom',
  align = 'center',
  className,
}) => {
  const [internalOpen, setInternalOpen] = useState(false);
  const [popoverPosition, setPopoverPosition] = useState({ top: 0, left: 0 });
  const triggerRef = useRef<HTMLElement>(null);
  const popoverRef = useRef<HTMLDivElement>(null);

  const isControlled = controlledOpen !== undefined;
  const isOpen = isControlled ? controlledOpen : internalOpen;

  const updatePosition = () => {
    if (!triggerRef.current || !popoverRef.current) {
      return;
    }

    const triggerRect = triggerRef.current.getBoundingClientRect();
    const popoverRect = popoverRef.current.getBoundingClientRect();
    const scrollY = window.scrollY;
    const scrollX = window.scrollX;

    let top = 0;
    let left = 0;

    switch (position) {
      case 'top':
        top = triggerRect.top + scrollY - popoverRect.height - 8;
        break;
      case 'bottom':
        top = triggerRect.bottom + scrollY + 8;
        break;
      case 'left':
        top = triggerRect.top + scrollY + triggerRect.height / 2 - popoverRect.height / 2;
        left = triggerRect.left + scrollX - popoverRect.width - 8;
        break;
      case 'right':
        top = triggerRect.top + scrollY + triggerRect.height / 2 - popoverRect.height / 2;
        left = triggerRect.right + scrollX + 8;
        break;
    }

    switch (align) {
      case 'start':
        if (position === 'top' || position === 'bottom') {
          left = triggerRect.left + scrollX;
        } else {
          // Already set for left/right
        }
        break;
      case 'center':
        if (position === 'top' || position === 'bottom') {
          left = triggerRect.left + scrollX + triggerRect.width / 2 - popoverRect.width / 2;
        }
        break;
      case 'end':
        if (position === 'top' || position === 'bottom') {
          left = triggerRect.right + scrollX - popoverRect.width;
        }
        break;
    }

    setPopoverPosition({ top, left });
  };

  useEffect(() => {
    if (isOpen) {
      updatePosition();
      const handleScroll = () => updatePosition();
      const handleResize = () => updatePosition();
      window.addEventListener('scroll', handleScroll, true);
      window.addEventListener('resize', handleResize);
      return () => {
        window.removeEventListener('scroll', handleScroll, true);
        window.removeEventListener('resize', handleResize);
      };
    }
  }, [isOpen]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        popoverRef.current &&
        !popoverRef.current.contains(event.target as Node) &&
        triggerRef.current &&
        !triggerRef.current.contains(event.target as Node)
      ) {
        if (isControlled && onOpenChange) {
          onOpenChange(false);
        } else {
          setInternalOpen(false);
        }
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => {
        document.removeEventListener('mousedown', handleClickOutside);
      };
    }
  }, [isOpen, isControlled, onOpenChange]);

  const handleToggle = () => {
    if (isControlled && onOpenChange) {
      onOpenChange(!isOpen);
    } else {
      setInternalOpen(!isOpen);
    }
  };

  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === 'Escape' && isOpen) {
      if (isControlled && onOpenChange) {
        onOpenChange(false);
      } else {
        setInternalOpen(false);
      }
    }
  };

  return (
    <>
      {React.cloneElement(trigger, {
        ref: triggerRef,
        onClick: handleToggle,
        'aria-expanded': isOpen,
        'aria-haspopup': 'true',
      })}
      {isOpen &&
        createPortal(
          <div
            ref={popoverRef}
            className={cn(
              'fixed z-50 bg-popover border rounded-lg shadow-lg p-4',
              className
            )}
            style={{
              top: `${popoverPosition.top}px`,
              left: `${popoverPosition.left}px`,
            }}
            role="dialog"
            aria-modal="false"
            onKeyDown={handleKeyDown}
            tabIndex={-1}
          >
            {content}
          </div>,
          document.body
        )}
    </>
  );
};



