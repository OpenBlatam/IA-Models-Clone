/**
 * Scroll area component
 */

'use client';

import React from 'react';
import { cn } from '@/lib/utils/cn';

export interface ScrollAreaProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  className?: string;
}

export const ScrollArea: React.FC<ScrollAreaProps> = ({ children, className, ...props }) => {
  return (
    <div
      className={cn(
        'overflow-auto scrollbar-thin scrollbar-thumb-muted-foreground/20 scrollbar-track-transparent',
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
};



