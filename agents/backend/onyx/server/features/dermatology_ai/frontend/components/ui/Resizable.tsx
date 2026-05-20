'use client';

import React, { useState, useRef, useEffect } from 'react';
import { clsx } from 'clsx';

interface ResizableProps {
  children: React.ReactNode;
  direction?: 'horizontal' | 'vertical';
  minSize?: number;
  maxSize?: number;
  defaultSize?: number;
  className?: string;
}

export const Resizable: React.FC<ResizableProps> = ({
  children,
  direction = 'horizontal',
  minSize = 100,
  maxSize = 800,
  defaultSize = 300,
  className,
}) => {
  const [size, setSize] = useState(defaultSize);
  const [isResizing, setIsResizing] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing || !containerRef.current) return;

      const rect = containerRef.current.getBoundingClientRect();
      const newSize =
        direction === 'horizontal'
          ? e.clientX - rect.left
          : e.clientY - rect.top;

      if (newSize >= minSize && newSize <= maxSize) {
        setSize(newSize);
      }
    };

    const handleMouseUp = () => {
      setIsResizing(false);
    };

    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isResizing, direction, minSize, maxSize]);

  return (
    <div
      ref={containerRef}
      className={clsx('relative', className)}
      style={
        direction === 'horizontal'
          ? { width: `${size}px` }
          : { height: `${size}px` }
      }
    >
      {children}
      <div
        onMouseDown={() => setIsResizing(true)}
        className={clsx(
          'absolute bg-gray-300 dark:bg-gray-600 hover:bg-primary-500 dark:hover:bg-primary-400 transition-colors cursor-col-resize',
          direction === 'horizontal'
            ? 'right-0 top-0 bottom-0 w-1'
            : 'bottom-0 left-0 right-0 h-1 cursor-row-resize'
        )}
      />
    </div>
  );
};


