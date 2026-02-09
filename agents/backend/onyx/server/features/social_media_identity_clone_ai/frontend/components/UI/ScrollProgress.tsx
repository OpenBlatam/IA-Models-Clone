import { memo, useState, useEffect } from 'react';
import { cn } from '@/lib/utils';

interface ScrollProgressProps {
  className?: string;
  color?: string;
  height?: number;
}

const ScrollProgress = memo(({
  className = '',
  color = 'bg-primary-600',
  height = 3,
}: ScrollProgressProps): JSX.Element => {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const handleScroll = (): void => {
      const windowHeight = window.innerHeight;
      const documentHeight = document.documentElement.scrollHeight;
      const scrollTop = window.scrollY || document.documentElement.scrollTop;
      const scrollableHeight = documentHeight - windowHeight;
      const scrollProgress = (scrollTop / scrollableHeight) * 100;
      setProgress(Math.min(100, Math.max(0, scrollProgress)));
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    handleScroll();

    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <div
      className={cn('fixed top-0 left-0 right-0 z-50', className)}
      style={{ height: `${height}px` }}
      role="progressbar"
      aria-valuenow={progress}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      <div
        className={cn('h-full transition-all duration-150', color)}
        style={{ width: `${progress}%` }}
      />
    </div>
  );
});

ScrollProgress.displayName = 'ScrollProgress';

export default ScrollProgress;



