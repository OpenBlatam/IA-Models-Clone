'use client';

import { ReactNode, useState, useRef, useEffect } from 'react';
import { cn } from '@/utils/classNames';

interface ResizableProps {
  children: ReactNode;
  defaultWidth?: number;
  defaultHeight?: number;
  minWidth?: number;
  minHeight?: number;
  maxWidth?: number;
  maxHeight?: number;
  direction?: 'horizontal' | 'vertical' | 'both';
  className?: string;
}

export function Resizable({
  children,
  defaultWidth = 300,
  defaultHeight = 200,
  minWidth = 100,
  minHeight = 100,
  maxWidth,
  maxHeight,
  direction = 'both',
  className,
}: ResizableProps) {
  const [width, setWidth] = useState(defaultWidth);
  const [height, setHeight] = useState(defaultHeight);
  const [isResizing, setIsResizing] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing || !containerRef.current) return;

      const rect = containerRef.current.getBoundingClientRect();
      const newWidth = e.clientX - rect.left;
      const newHeight = e.clientY - rect.top;

      if (direction === 'horizontal' || direction === 'both') {
        const w = Math.max(
          minWidth,
          maxWidth ? Math.min(maxWidth, newWidth) : newWidth
        );
        setWidth(w);
      }

      if (direction === 'vertical' || direction === 'both') {
        const h = Math.max(
          minHeight,
          maxHeight ? Math.min(maxHeight, newHeight) : newHeight
        );
        setHeight(h);
      }
    };

    const handleMouseUp = () => {
      setIsResizing(false);
    };

    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      document.body.style.cursor =
        direction === 'horizontal'
          ? 'ew-resize'
          : direction === 'vertical'
          ? 'ns-resize'
          : 'nwse-resize';
      document.body.style.userSelect = 'none';
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
  }, [isResizing, direction, minWidth, minHeight, maxWidth, maxHeight]);

  return (
    <div
      ref={containerRef}
      className={cn('relative', className)}
      style={{ width, height }}
    >
      {children}
      {(direction === 'horizontal' || direction === 'both') && (
        <div
          onMouseDown={() => setIsResizing(true)}
          className="absolute right-0 top-0 bottom-0 w-1 cursor-ew-resize hover:bg-primary-500 bg-gray-300 dark:bg-gray-600"
        />
      )}
      {(direction === 'vertical' || direction === 'both') && (
        <div
          onMouseDown={() => setIsResizing(true)}
          className="absolute bottom-0 left-0 right-0 h-1 cursor-ns-resize hover:bg-primary-500 bg-gray-300 dark:bg-gray-600"
        />
      )}
      {direction === 'both' && (
        <div
          onMouseDown={() => setIsResizing(true)}
          className="absolute bottom-0 right-0 w-3 h-3 cursor-nwse-resize hover:bg-primary-500 bg-gray-300 dark:bg-gray-600"
        />
      )}
    </div>
  );
}

