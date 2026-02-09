import { memo, useState, useEffect, useCallback } from 'react';
import { useIntersectionObserver } from '@/lib/hooks';
import Button from './Button';
import { ArrowUp } from 'lucide-react';
import { cn } from '@/lib/utils';

interface BackToTopProps {
  threshold?: number;
  className?: string;
  smooth?: boolean;
}

const BackToTop = memo(({
  threshold = 400,
  className = '',
  smooth = true,
}: BackToTopProps): JSX.Element => {
  const [isVisible, setIsVisible] = useState(false);
  const { elementRef } = useIntersectionObserver<HTMLDivElement>({
    threshold: 0,
  });

  useEffect(() => {
    const handleScroll = (): void => {
      setIsVisible(window.scrollY > threshold);
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, [threshold]);

  const scrollToTop = useCallback(() => {
    if (smooth) {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } else {
      window.scrollTo(0, 0);
    }
  }, [smooth]);

  if (!isVisible) {
    return <div ref={elementRef} className="hidden" />;
  }

  return (
    <Button
      onClick={scrollToTop}
      variant="secondary"
      className={cn(
        'fixed bottom-8 right-8 z-50',
        'rounded-full p-3 shadow-lg',
        'hover:scale-110 transition-transform',
        className
      )}
      aria-label="Back to top"
    >
      <ArrowUp className="w-5 h-5" />
    </Button>
  );
});

BackToTop.displayName = 'BackToTop';

export default BackToTop;



