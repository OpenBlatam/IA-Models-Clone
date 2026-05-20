/**
 * Memes Grid Skeleton Component
 * Loading skeleton for memes grid
 */

import { Card, CardContent } from '@/components/ui/Card';
import { Skeleton } from '@/components/ui/Skeleton';

interface MemesGridSkeletonProps {
  count?: number;
}

/**
 * Skeleton component for memes grid
 */
export const MemesGridSkeleton = ({ count = 8 }: MemesGridSkeletonProps) => {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
      {Array.from({ length: count }).map((_, i) => (
        <Card key={i} className="overflow-hidden">
          <Skeleton className="aspect-square w-full" />
          <CardContent className="p-4 space-y-3">
            <Skeleton variant="text" className="h-4 w-3/4" />
            <div className="flex gap-1">
              <Skeleton variant="text" className="h-5 w-16" />
              <Skeleton variant="text" className="h-5 w-16" />
            </div>
            <Skeleton variant="text" className="h-8 w-full" />
          </CardContent>
        </Card>
      ))}
    </div>
  );
};


