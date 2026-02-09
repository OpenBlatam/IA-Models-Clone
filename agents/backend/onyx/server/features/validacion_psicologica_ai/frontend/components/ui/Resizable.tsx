/**
 * Resizable component
 */

'use client';

import React, { useState, useRef, useEffect } from 'react';
import { cn } from '@/lib/utils/cn';
import { GripVertical } from 'lucide-react';

export interface ResizableProps {
  children: React.ReactNode;
  defaultWidth?: number;
  minWidth?: number;
  maxWidth?: number;
  direction?: 'horizontal' | 'vertical';
  className?: string;
}

export const Resizable: React.FC<ResizableProps> = ({
  children,
  defaultWidth = 300,
  minWidth = 200,
  maxWidth = 800,
  direction = 'horizontal',
  className,
}) => {
  const [width, setWidth] = useState(defaultWidth);
  const [isResizing, setIsResizing] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const startXRef = useRef<number>(0);
  const startWidthRef = useRef<number>(0);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing) {
        return;
      }

      const delta = direction === 'horizontal' ? e.clientX - startXRef.current : e.clientY - startXRef.current;
      const newWidth = startWidthRef.current + delta;
      const clampedWidth = Math.max(minWidth, Math.min(maxWidth, newWidth));
      setWidth(clampedWidth);
    };

    const handleMouseUp = () => {
      setIsResizing(false);
    };

    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isResizing, minWidth, maxWidth, direction]);

  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);
    startXRef.current = direction === 'horizontal' ? e.clientX : e.clientY;
    startWidthRef.current = width;
  };

  return (
    <div
      ref={containerRef}
      className={cn('relative flex', direction === 'vertical' && 'flex-col', className)}
    >
      <div
        style={{
          [direction === 'horizontal' ? 'width' : 'height']: `${width}px`,
        }}
        className="overflow-auto"
      >
        {children}
      </div>
      <div
        onMouseDown={handleMouseDown}
        className={cn(
          'flex items-center justify-center cursor-col-resize hover:bg-accent transition-colors',
          direction === 'vertical' && 'cursor-row-resize',
          isResizing && 'bg-accent'
        )}
        style={{
          [direction === 'horizontal' ? 'width' : 'height']: '8px',
        }}
        role="separator"
        aria-orientation={direction}
        aria-label="Redimensionar"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
            e.preventDefault();
            const delta = e.key === 'ArrowLeft' ? -10 : 10;
            setWidth((prev) => Math.max(minWidth, Math.min(maxWidth, prev + delta)));
          }
        }}
      >
        <GripVertical
          className={cn(
            'h-4 w-4 text-muted-foreground',
            direction === 'vertical' && 'rotate-90'
          )}
          aria-hidden="true"
        />
      </div>
    </div>
  );
};



