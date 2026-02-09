import { memo, useState, useEffect, useRef } from 'react';
import { useIntersectionObserver } from '@/lib/hooks';
import { cn } from '@/lib/utils';

interface ReadingProgressProps {
  target?: string;
  className?: string;
  color?: string;
}

const ReadingProgress = memo(({
  target = 'article',
  className = '',
  color = 'bg-primary-600',
}: ReadingProgressProps): JSX.Element => {
  const [progress, setProgress] = useState(0);
  const targetRef = useRef<HTMLElement | null>(null);
  const { elementRef } = useIntersectionObserver<HTMLDivElement>({
    threshold: 0,
  });

  useEffect(() => {
    targetRef.current = document.querySelector(target);

    const handleScroll = (): void => {
      const element = targetRef.current;
      if (!element) {
        return;
      }

      const elementTop = element.offsetTop;
      const elementHeight = element.offsetHeight;
      const windowHeight = window.innerHeight;
      const scrollTop = window.scrollY;

      const elementBottom = elementTop + elementHeight;
      const viewportTop = scrollTop;
      const viewportBottom = scrollTop + windowHeight;

      if (viewportBottom < elementTop || viewportTop > elementBottom) {
        setProgress(0);
        return;
      }

      const visibleTop = Math.max(viewportTop, elementTop);
      const visibleBottom = Math.min(viewportBottom, elementBottom);
      const visibleHeight = visibleBottom - visibleTop;
      const progressValue = (visibleHeight / elementHeight) * 100;

      setProgress(Math.min(100, Math.max(0, progressValue)));
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    handleScroll();

    return () => window.removeEventListener('scroll', handleScroll);
  }, [target]);

  return (
    <div
      ref={elementRef}
      className={cn('fixed top-0 left-0 right-0 z-50 h-1', className)}
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

ReadingProgress.displayName = 'ReadingProgress';

export default ReadingProgress;



