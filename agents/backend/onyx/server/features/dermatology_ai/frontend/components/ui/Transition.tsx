'use client';

import React, { ReactNode } from 'react';
import { clsx } from 'clsx';

interface TransitionProps {
  children: ReactNode;
  show: boolean;
  enter?: string;
  enterFrom?: string;
  enterTo?: string;
  leave?: string;
  leaveFrom?: string;
  leaveTo?: string;
  className?: string;
}

export const Transition: React.FC<TransitionProps> = ({
  children,
  show,
  enter = 'transition',
  enterFrom = 'opacity-0',
  enterTo = 'opacity-100',
  leave = 'transition',
  leaveFrom = 'opacity-100',
  leaveTo = 'opacity-0',
  className,
}) => {
  if (!show) return null;

  return (
    <div
      className={clsx(
        enter,
        show ? enterTo : enterFrom,
        className
      )}
    >
      {children}
    </div>
  );
};


