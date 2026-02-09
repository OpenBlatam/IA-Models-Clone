import { memo, useEffect, useRef, useState } from 'react';
import { cn } from '@/lib/utils';

interface StickyProps {
  children: React.ReactNode;
  offset?: number;
  className?: string;
  zIndex?: number;
}

const Sticky = memo(({
  children,
  offset = 0,
  className = '',
  zIndex = 10,
}: StickyProps): JSX.Element => {
  const [isSticky, setIsSticky] = useState(false);
  const elementRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const element = elementRef.current;
    if (!element) {
      return;
    }

    const handleScroll = (): void => {
      const rect = element.getBoundingClientRect();
      setIsSticky(rect.top <= offset);
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    handleScroll();

    return () => window.removeEventListener('scroll', handleScroll);
  }, [offset]);

  return (
    <div
      ref={elementRef}
      className={cn(
        'transition-all duration-200',
        isSticky && 'fixed top-0 left-0 right-0',
        className
      )}
      style={isSticky ? { zIndex, top: `${offset}px` } : undefined}
    >
      {children}
    </div>
  );
});

Sticky.displayName = 'Sticky';

export default Sticky;



