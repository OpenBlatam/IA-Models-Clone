'use client';

import { memo, useEffect, useState } from 'react';
import { ArrowUp } from 'lucide-react';
import { useWindowScroll, useScrollToTop } from '@/lib/hooks';
import { Button } from './Button';
import { cn } from '@/lib/utils';

interface ScrollToTopProps {
  threshold?: number;
  className?: string;
  show?: boolean;
}

const ScrollToTop = memo(
  ({ threshold = 400, className, show }: ScrollToTopProps): JSX.Element => {
    const { y } = useWindowScroll();
    const scrollToTop = useScrollToTop();
    const [isVisible, setIsVisible] = useState(false);

    useEffect(() => {
      setIsVisible(y > threshold);
    }, [y, threshold]);

    if (show !== undefined && !show) {
      return null;
    }

    if (!isVisible) {
      return null;
    }

    return (
      <Button
        variant="primary"
        size="icon"
        onClick={scrollToTop}
        className={cn(
          'fixed bottom-8 right-8 z-50 rounded-full shadow-lg transition-all',
          className
        )}
        aria-label="Scroll to top"
      >
        <ArrowUp className="w-5 h-5" aria-hidden="true" />
      </Button>
    );
  }
);

ScrollToTop.displayName = 'ScrollToTop';

export default ScrollToTop;

