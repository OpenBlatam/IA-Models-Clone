/**
 * Dashboard Skeleton Component
 * Loading skeleton for dashboard page
 */

import { Card, CardContent, CardHeader } from '@/components/ui/Card';
import { Skeleton } from '@/components/ui/Skeleton';

/**
 * Skeleton component for dashboard stats cards
 */
export const DashboardStatsSkeleton = () => {
  return (
    <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
      {Array.from({ length: 4 }).map((_, i) => (
        <Card key={i}>
          <CardContent className="p-6">
            <div className="space-y-3">
              <Skeleton variant="text" className="h-4 w-24" />
              <Skeleton variant="text" className="h-8 w-16" />
              <Skeleton variant="text" className="h-3 w-32" />
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
};

/**
 * Skeleton component for dashboard charts
 */
export const DashboardChartsSkeleton = () => {
  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
      {Array.from({ length: 2 }).map((_, i) => (
        <Card key={i}>
          <CardHeader>
            <Skeleton variant="text" className="h-6 w-32" />
          </CardHeader>
          <CardContent>
            <Skeleton className="h-[300px] w-full" />
          </CardContent>
        </Card>
      ))}
    </div>
  );
};


