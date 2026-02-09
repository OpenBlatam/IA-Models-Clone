'use client';

import { useState, useEffect } from 'react';
import { GripVertical } from 'lucide-react';
import { cn } from '@/lib/utils/cn';

interface SplitViewProps {
  left: React.ReactNode;
  right: React.ReactNode;
  defaultSplit?: number; // 0-100 percentage
  minLeft?: number;
  minRight?: number;
  className?: string;
  resizable?: boolean;
}

export default function SplitView({
  left,
  right,
  defaultSplit = 50,
  minLeft = 20,
  minRight = 20,
  className,
  resizable = true,
}: SplitViewProps) {
  const [split, setSplit] = useState(defaultSplit);
  const [isDragging, setIsDragging] = useState(false);

  const handleMouseDown = () => {
    if (!resizable) return;
    setIsDragging(true);
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (!isDragging || !resizable) return;
    const container = e.currentTarget as HTMLElement;
    const rect = container.getBoundingClientRect();
    const newSplit = ((e.clientX - rect.left) / rect.width) * 100;
    const clampedSplit = Math.max(minLeft, Math.min(100 - minRight, newSplit));
    setSplit(clampedSplit);
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isDragging]);

  return (
    <div
      className={cn('flex h-full relative', className)}
      onMouseMove={handleMouseMove as any}
    >
      <div
        className="overflow-auto"
        style={{ width: `${split}%` }}
      >
        {left}
      </div>
      
      {resizable && (
        <div
          onMouseDown={handleMouseDown}
          className={cn(
            'w-1 bg-gray-200 hover:bg-tesla-blue cursor-col-resize flex items-center justify-center transition-colors group',
            isDragging && 'bg-tesla-blue'
          )}
        >
          <GripVertical className="w-4 h-4 text-gray-400 group-hover:text-tesla-blue transition-colors" />
        </div>
      )}
      
      <div
        className="overflow-auto"
        style={{ width: `${100 - split}%` }}
      >
        {right}
      </div>
    </div>
  );
}

