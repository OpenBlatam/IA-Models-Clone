/**
 * Drawer component for side panels
 */

'use client';

import React, { useEffect } from 'react';
import { cn } from '@/lib/utils/cn';
import { X } from 'lucide-react';
import { Button } from './Button';
import { Backdrop } from './Backdrop';

export interface DrawerProps {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  children: React.ReactNode;
  position?: 'left' | 'right' | 'top' | 'bottom';
  size?: 'sm' | 'md' | 'lg' | 'xl' | 'full';
  className?: string;
}

const sizeClasses = {
  sm: 'max-w-sm',
  md: 'max-w-md',
  lg: 'max-w-lg',
  xl: 'max-w-xl',
  full: 'max-w-full',
};

const positionClasses = {
  left: 'left-0 top-0 bottom-0',
  right: 'right-0 top-0 bottom-0',
  top: 'top-0 left-0 right-0',
  bottom: 'bottom-0 left-0 right-0',
};

const positionAnimations = {
  left: 'translate-x-0',
  right: 'translate-x-0',
  top: 'translate-y-0',
  bottom: 'translate-y-0',
};

const positionAnimationsClosed = {
  left: '-translate-x-full',
  right: 'translate-x-full',
  top: '-translate-y-full',
  bottom: 'translate-y-full',
};

export const Drawer: React.FC<DrawerProps> = ({
  isOpen,
  onClose,
  title,
  children,
  position = 'right',
  size = 'md',
  className,
}) => {
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }

    return () => {
      document.body.style.overflow = '';
    };
  }, [isOpen]);

  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === 'Escape') {
      onClose();
    }
  };

  if (!isOpen) {
    return null;
  }

  return (
    <>
      <Backdrop isOpen={isOpen} onClick={onClose} />
      <div
        className={cn(
          'fixed z-50 bg-background shadow-lg transition-transform duration-300 ease-in-out',
          position === 'left' || position === 'right' ? 'h-full w-full' : 'h-auto w-full',
          position === 'left' || position === 'right' ? sizeClasses[size] : '',
          positionClasses[position],
          isOpen ? positionAnimations[position] : positionAnimationsClosed[position],
          className
        )}
        role="dialog"
        aria-modal="true"
        aria-labelledby={title ? 'drawer-title' : undefined}
        onKeyDown={handleKeyDown}
        tabIndex={-1}
      >
        <div className="flex flex-col h-full">
          {title && (
            <div className="flex items-center justify-between p-4 border-b">
              <h2 id="drawer-title" className="text-lg font-semibold">
                {title}
              </h2>
              <Button
                variant="ghost"
                size="sm"
                onClick={onClose}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    onClose();
                  }
                }}
                aria-label="Cerrar drawer"
                tabIndex={0}
              >
                <X className="h-4 w-4" aria-hidden="true" />
              </Button>
            </div>
          )}
          <div className="flex-1 overflow-y-auto p-4">{children}</div>
        </div>
      </div>
    </>
  );
};



