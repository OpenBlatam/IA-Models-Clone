import { memo, useEffect, useRef, useCallback } from 'react';
import { cn } from '@/lib/utils';
import LoadingSpinner from './LoadingSpinner';

interface InfiniteScrollProps {
  hasMore: boolean;
  isLoading: boolean;
  onLoadMore: () => void;
  threshold?: number;
  className?: string;
  loader?: React.ReactNode;
  endMessage?: React.ReactNode;
  children: React.ReactNode;
}

const InfiniteScroll = memo(({
  hasMore,
  isLoading,
  onLoadMore,
  threshold = 200,
  className = '',
  loader,
  endMessage,
  children,
}: InfiniteScrollProps): JSX.Element => {
  const observerRef = useRef<IntersectionObserver | null>(null);
  const sentinelRef = useRef<HTMLDivElement>(null);

  const handleIntersection = useCallback(
    (entries: IntersectionObserverEntry[]) => {
      const [entry] = entries;
      if (entry.isIntersecting && hasMore && !isLoading) {
        onLoadMore();
      }
    },
    [hasMore, isLoading, onLoadMore]
  );

  useEffect(() => {
    if (!sentinelRef.current) {
      return;
    }

    observerRef.current = new IntersectionObserver(handleIntersection, {
      rootMargin: `${threshold}px`,
    });

    observerRef.current.observe(sentinelRef.current);

    return () => {
      if (observerRef.current) {
        observerRef.current.disconnect();
      }
    };
  }, [handleIntersection, threshold]);

  return (
    <div className={cn('space-y-4', className)}>
      {children}
      <div ref={sentinelRef} className="h-4" />
      {isLoading && (loader || <LoadingSpinner />)}
      {!hasMore && endMessage && <div className="text-center text-gray-500">{endMessage}</div>}
    </div>
  );
});

InfiniteScroll.displayName = 'InfiniteScroll';

export default InfiniteScroll;



