'use client';

import React, { useState } from 'react';
import { Resizable } from './Resizable';
import { clsx } from 'clsx';

interface SplitPaneProps {
  left: React.ReactNode;
  right: React.ReactNode;
  defaultSize?: number;
  minSize?: number;
  maxSize?: number;
  direction?: 'horizontal' | 'vertical';
  className?: string;
}

export const SplitPane: React.FC<SplitPaneProps> = ({
  left,
  right,
  defaultSize = 300,
  minSize = 100,
  maxSize = 800,
  direction = 'horizontal',
  className,
}) => {
  return (
    <div
      className={clsx(
        'flex',
        direction === 'horizontal' ? 'flex-row' : 'flex-col',
        className
      )}
    >
      <Resizable
        direction={direction}
        defaultSize={defaultSize}
        minSize={minSize}
        maxSize={maxSize}
      >
        <div className="h-full overflow-auto">{left}</div>
      </Resizable>
      <div className="flex-1 overflow-auto">{right}</div>
    </div>
  );
};


