'use client';

import React, { useState, useEffect } from 'react';
import { clsx } from 'clsx';
import { useScrollPosition } from '@/hooks';

interface StickyHeaderProps {
  children: React.ReactNode;
  offset?: number;
  className?: string;
  scrolledClassName?: string;
}

export const StickyHeader: React.FC<StickyHeaderProps> = ({
  children,
  offset = 0,
  className,
  scrolledClassName,
}) => {
  const { y } = useScrollPosition();
  const [isScrolled, setIsScrolled] = useState(false);

  useEffect(() => {
    setIsScrolled(y > offset);
  }, [y, offset]);

  return (
    <header
      className={clsx(
        'sticky top-0 z-40 transition-all duration-300',
        isScrolled && 'shadow-md',
        isScrolled ? scrolledClassName : className
      )}
    >
      {children}
    </header>
  );
};


