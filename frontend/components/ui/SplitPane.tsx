'use client';

import { ReactNode, useState, useRef, useEffect } from 'react';
import { cn } from '@/utils/classNames';

interface SplitPaneProps {
  left: ReactNode;
  right: ReactNode;
  defaultSize?: number; // percentage
  minSize?: number;
  maxSize?: number;
  direction?: 'horizontal' | 'vertical';
  className?: string;
}

export function SplitPane({
  left,
  right,
  defaultSize = 50,
  minSize = 10,
  maxSize = 90,
  direction = 'horizontal',
  className,
}: SplitPaneProps) {
  const [size, setSize] = useState(defaultSize);
  const [isDragging, setIsDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const isHorizontal = direction === 'horizontal';

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging || !containerRef.current) return;

      const rect = containerRef.current.getBoundingClientRect();
      const totalSize = isHorizontal ? rect.width : rect.height;
      const position = isHorizontal
        ? e.clientX - rect.left
        : e.clientY - rect.top;

      const newSize = Math.max(
        minSize,
        Math.min(maxSize, (position / totalSize) * 100)
      );
      setSize(newSize);
    };

    const handleMouseUp = () => {
      setIsDragging(false);
    };

    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = isHorizontal ? 'col-resize' : 'row-resize';
      document.body.style.userSelect = 'none';
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
  }, [isDragging, isHorizontal, minSize, maxSize]);

  return (
    <div
      ref={containerRef}
      className={cn(
        'flex',
        isHorizontal ? 'flex-row' : 'flex-col',
        'w-full h-full',
        className
      )}
    >
      <div
        className={cn('overflow-auto', isHorizontal ? '' : 'w-full')}
        style={{
          [isHorizontal ? 'width' : 'height']: `${size}%`,
        }}
      >
        {left}
      </div>
      <div
        onMouseDown={() => setIsDragging(true)}
        className={cn(
          'bg-gray-200 dark:bg-gray-700',
          'hover:bg-gray-300 dark:hover:bg-gray-600',
          'transition-colors',
          isHorizontal
            ? 'w-1 cursor-col-resize'
            : 'h-1 cursor-row-resize w-full'
        )}
      />
      <div
        className={cn('overflow-auto', isHorizontal ? '' : 'w-full')}
        style={{
          [isHorizontal ? 'width' : 'height']: `${100 - size}%`,
        }}
      >
        {right}
      </div>
    </div>
  );
}

