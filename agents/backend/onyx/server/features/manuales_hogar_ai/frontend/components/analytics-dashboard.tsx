'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/lib/api/client';
import { useStatistics } from '@/lib/hooks/use-manuals';
import { useMultipleQueries } from '@/lib/hooks/use-multiple-queries';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { StatCard } from './ui/stat-card';
import { LoadingState } from './ui/loading-state';
import { ErrorState } from './ui/error-state';
import { EmptyState } from './ui/empty-state';
import { FileText, TrendingUp, Star, Users } from 'lucide-react';
import { ANALYTICS, MESSAGES } from '@/lib/constants';
import { ANALYTICS_PERIODS, ANALYTICS_INTERVALS } from '@/lib/constants/analytics';
import { getTopModel, getTopModelUsage, getSortedCategories, getCategoryCount } from '@/lib/utils/analytics';

export const AnalyticsDashboard = (): JSX.Element => {
  const [days, setDays] = useState(ANALYTICS.DEFAULT_DAYS);
  const [interval, setInterval] = useState<typeof ANALYTICS.INTERVALS[number]>('day');

  const { data: stats, isLoading: statsLoading, error: statsError } = useStatistics(days);
  const { data: comprehensiveStats, isLoading: comprehensiveLoading } = useQuery({
    queryKey: ['comprehensive-stats', days],
    queryFn: () => apiClient.getComprehensiveStats(days),
  });
  const { data: trends, isLoading: trendsLoading } = useQuery({
    queryKey: ['trends', days, interval],
    queryFn: () => apiClient.getTrends(days, interval),
  });

  const { shouldShowLoading, shouldShowError, error } = useMultipleQueries([
    { isLoading: statsLoading, error: statsError },
    { isLoading: comprehensiveLoading, error: null },
    { isLoading: trendsLoading, error: null },
  ]);

  if (shouldShowLoading) {
    return <LoadingState title={MESSAGES.ANALYTICS.LOADING} />;
  }

  if (shouldShowError) {
    return <ErrorState title="Error" message={MESSAGES.ANALYTICS.LOAD_ERROR} />;
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <Select value={String(days)} onValueChange={(value) => setDays(Number(value))}>
            <SelectTrigger className="w-32">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {ANALYTICS_PERIODS.map((period) => (
                <SelectItem key={period.value} value={String(period.value)}>
                  {period.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Select value={interval} onValueChange={(value) => setInterval(value as 'day' | 'week')}>
            <SelectTrigger className="w-32">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {ANALYTICS_INTERVALS.map((intervalOption) => (
                <SelectItem key={intervalOption.value} value={intervalOption.value}>
                  {intervalOption.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard
            title="Total Manuales"
            value={stats.total_manuals}
            description={`En los últimos ${days} días`}
            icon={FileText}
          />
          <StatCard
            title="Total Tokens"
            value={stats.total_tokens}
            description="Tokens utilizados"
            icon={TrendingUp}
          />
          <StatCard
            title="Top Modelo"
            value={getTopModel(stats)}
            description={`${getTopModelUsage(stats)} usos`}
            icon={Star}
          />
          <StatCard
            title="Categorías"
            value={getCategoryCount(stats)}
            description="Categorías activas"
            icon={Users}
          />
        </div>
      )}

      {stats && Object.keys(stats.category_stats).length > 0 ? (
        <Card>
          <CardHeader>
            <CardTitle>Estadísticas por Categoría</CardTitle>
            <CardDescription>
              Distribución de manuales por categoría
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {getSortedCategories(stats).map(([category, count]) => (
                <div key={category} className="flex items-center justify-between p-2 hover:bg-gray-50 rounded">
                  <span className="capitalize text-sm">{category}</span>
                  <span className="font-semibold">{count}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      ) : (
        <EmptyState
          title={MESSAGES.ANALYTICS.NO_CATEGORIES}
          description={MESSAGES.ANALYTICS.NO_CATEGORIES_DESC}
        />
      )}

      {comprehensiveStats?.stats && (
        <Card>
          <CardHeader>
            <CardTitle>Estadísticas Comprehensivas</CardTitle>
            <CardDescription>
              Análisis detallado del sistema
            </CardDescription>
          </CardHeader>
          <CardContent>
            <pre className="text-xs bg-gray-50 p-4 rounded-lg overflow-auto">
              {JSON.stringify(comprehensiveStats.stats, null, 2)}
            </pre>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

