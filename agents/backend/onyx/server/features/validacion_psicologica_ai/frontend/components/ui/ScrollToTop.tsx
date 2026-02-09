/**
 * Scroll to top button component
 */

'use client';

import React, { useState, useEffect } from 'react';
import { Button } from './Button';
import { ArrowUp } from 'lucide-react';
import { cn } from '@/lib/utils/cn';

export interface ScrollToTopProps {
  threshold?: number;
  className?: string;
}

export const ScrollToTop: React.FC<ScrollToTopProps> = ({
  threshold = 400,
  className,
}) => {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsVisible(window.scrollY > threshold);
    };

    window.addEventListener('scroll', handleScroll);
    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, [threshold]);

  const handleClick = () => {
    window.scrollTo({
      top: 0,
      behavior: 'smooth',
    });
  };

  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      handleClick();
    }
  };

  if (!isVisible) {
    return null;
  }

  return (
    <Button
      variant="primary"
      size="sm"
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      className={cn(
        'fixed bottom-8 right-8 z-50 rounded-full shadow-lg',
        className
      )}
      aria-label="Volver arriba"
      tabIndex={0}
    >
      <ArrowUp className="h-4 w-4" aria-hidden="true" />
    </Button>
  );
};



