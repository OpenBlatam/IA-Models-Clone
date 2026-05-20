/**
 * Post Skeleton Component
 * Loading skeleton for post cards
 */

import { Card, CardContent } from '@/components/ui/Card';
import { Skeleton } from '@/components/ui/Skeleton';

/**
 * Skeleton component for post cards
 */
export const PostSkeleton = () => {
  return (
    <Card>
      <CardContent className="p-6">
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1 space-y-3">
            <div className="flex items-center gap-2">
              <Skeleton variant="text" className="h-5 w-20" />
              <Skeleton variant="text" className="h-4 w-32" />
            </div>
            <Skeleton variant="text" className="h-4 w-full" />
            <Skeleton variant="text" className="h-4 w-5/6" />
            <div className="flex gap-2">
              <Skeleton variant="text" className="h-6 w-16" />
              <Skeleton variant="text" className="h-6 w-16" />
              <Skeleton variant="text" className="h-6 w-16" />
            </div>
          </div>
          <div className="flex gap-2">
            <Skeleton variant="circular" className="h-8 w-8" />
            <Skeleton variant="circular" className="h-8 w-8" />
            <Skeleton variant="circular" className="h-8 w-8" />
          </div>
        </div>
      </CardContent>
    </Card>
  );
};


