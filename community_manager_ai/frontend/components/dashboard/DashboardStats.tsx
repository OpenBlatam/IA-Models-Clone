/**
 * Dashboard Stats Component
 * Server Component for displaying dashboard statistics
 */

import { Suspense } from 'react';
import { StatsCard } from '@/components/ui/StatsCard';
import { Skeleton } from '@/components/ui/Skeleton';
import { FileText, Calendar, Share2, Users } from 'lucide-react';
import type { DashboardOverview } from '@/types';

interface DashboardStatsProps {
  overview: DashboardOverview | undefined;
  isLoading: boolean;
  t: (key: string) => string;
}

/**
 * Loading skeleton for stats cards
 */
const StatsSkeleton = () => (
  <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
    {Array.from({ length: 4 }).map((_, i) => (
      <div key={i} className="rounded-lg border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-6">
        <Skeleton className="h-4 w-24 mb-2" />
        <Skeleton className="h-8 w-16 mb-2" />
        <Skeleton className="h-3 w-32" />
      </div>
    ))}
  </div>
);

/**
 * Dashboard statistics cards
 */
export const DashboardStats = ({ overview, isLoading, t }: DashboardStatsProps) => {
  if (isLoading) {
    return <StatsSkeleton />;
  }

  const stats = [
    {
      name: t('totalPosts'),
      value: overview?.total_posts || 0,
      icon: FileText,
      color: 'text-blue-600',
      trend: 'up' as const,
      change: {
        value: 12,
        label: 'vs mes anterior',
        positive: true,
      },
    },
    {
      name: t('scheduled'),
      value: overview?.scheduled_posts || 0,
      icon: Calendar,
      color: 'text-yellow-600',
      trend: 'neutral' as const,
    },
    {
      name: t('published'),
      value: overview?.published_posts || 0,
      icon: Share2,
      color: 'text-green-600',
      trend: 'up' as const,
      change: {
        value: 8,
        label: 'vs mes anterior',
        positive: true,
      },
    },
    {
      name: t('platforms'),
      value: overview?.connected_platforms || 0,
      icon: Users,
      color: 'text-purple-600',
      trend: 'neutral' as const,
    },
  ];

  return (
    <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
      {stats.map((stat, index) => (
        <StatsCard
          key={index}
          title={stat.name}
          value={stat.value}
          icon={stat.icon}
          trend={stat.trend}
          change={stat.change}
        />
      ))}
    </div>
  );
};


