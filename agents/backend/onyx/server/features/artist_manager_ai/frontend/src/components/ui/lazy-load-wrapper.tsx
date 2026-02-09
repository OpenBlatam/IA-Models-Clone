'use client';

import { ReactNode, Suspense } from 'react';
import { useIntersectionObserver } from '@/hooks/use-intersection-observer';
import { LoadingSpinner } from '@/components/ui/loading-spinner';
import { cn } from '@/lib/utils';

interface LazyLoadWrapperProps {
  children: ReactNode;
  fallback?: ReactNode;
  threshold?: number;
  className?: string;
  triggerOnce?: boolean;
}

const LazyLoadWrapper = ({
  children,
  fallback = <LoadingSpinner />,
  threshold = 0.1,
  className,
  triggerOnce = true,
}: LazyLoadWrapperProps) => {
  const { elementRef, hasIntersected } = useIntersectionObserver<HTMLDivElement>({
    threshold,
    triggerOnce,
  });

  return (
    <div ref={elementRef} className={cn(className)}>
      {hasIntersected ? (
        <Suspense fallback={fallback}>{children}</Suspense>
      ) : (
        fallback
      )}
    </div>
  );
};

export { LazyLoadWrapper };

