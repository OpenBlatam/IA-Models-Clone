'use client';

import { useQuery } from 'react-query';
import { useMemo } from 'react';
import { apiClient } from '@/lib/api/client';
import type { DashboardResponse, MetricsResponse } from '@/types';
import PageLayout from '@/components/Layout/PageLayout';
import Card from '@/components/UI/Card';
import StatsCard from '@/components/UI/StatsCard';
import MetricItem from '@/components/UI/MetricItem';
import { formatNumber } from '@/lib/utils';
import { REFETCH_INTERVALS } from '@/lib/constants';

const DashboardPage = (): JSX.Element => {
  const { data: dashboard, isLoading: dashboardLoading } = useQuery<DashboardResponse>(
    'dashboard',
    () => apiClient.getDashboard(),
    { staleTime: 30000, cacheTime: 300000 }
  );

  const { data: metrics, isLoading: metricsLoading } = useQuery<MetricsResponse>(
    'metrics',
    () => apiClient.getMetrics(),
    { 
      refetchInterval: REFETCH_INTERVALS.METRICS,
      staleTime: 10000,
      cacheTime: 60000
    }
  );

  const isLoading = dashboardLoading || metricsLoading;
  const hasNoData = !dashboard && !metrics;

  if (isLoading) {
    return (
      <PageLayout>
        <div className="max-w-7xl mx-auto">
          <h1 className="text-3xl font-bold mb-8">Dashboard</h1>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[1, 2, 3].map((i) => (
              <Card key={i}>
                <div className="animate-pulse">
                  <div className="h-4 bg-gray-200 rounded w-24 mb-2"></div>
                  <div className="h-8 bg-gray-200 rounded w-16"></div>
                </div>
              </Card>
            ))}
          </div>
        </div>
      </PageLayout>
    );
  }

  if (hasNoData) {
    return (
      <PageLayout>
        <div className="max-w-7xl mx-auto">
          <h1 className="text-3xl font-bold mb-8">Dashboard</h1>
          <Card>
            <p className="text-center text-gray-600 py-8">No data available</p>
          </Card>
        </div>
      </PageLayout>
    );
  }

  return (
    <PageLayout>
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-8">Dashboard</h1>

        {dashboard && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <StatsCard
              label="Total Identities"
              value={dashboard.dashboard.overview.total_identities}
              icon="👤"
            />
            <StatsCard
              label="Total Content"
              value={dashboard.dashboard.overview.total_content}
              icon="📝"
            />
            <StatsCard
              label="Content Today"
              value={dashboard.dashboard.overview.content_today}
              icon="📅"
            />
          </div>
        )}

        {dashboard && (
          <Card title="Content by Platform" className="mb-8">
            <div className="grid grid-cols-3 gap-4">
              {Object.entries(dashboard.dashboard.content_by_platform).map(([platform, count]) => (
                <div key={platform} className="text-center">
                  <p className="text-sm text-gray-600 mb-1">{platform.toUpperCase()}</p>
                  <p className="text-3xl font-bold">{formatNumber(count)}</p>
                </div>
              ))}
            </div>
          </Card>
        )}

        {metrics && (
          <Card title="System Metrics" className="mb-8">
            <div className="space-y-4">
              <div>
                <p className="text-sm font-semibold mb-2">Counters</p>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {Object.entries(metrics.metrics.counters).map(([key, value]) => (
                    <MetricItem
                      key={key}
                      label={key.replace(/_/g, ' ')}
                      value={formatNumber(value)}
                    />
                  ))}
                </div>
              </div>
              {Object.keys(metrics.metrics.timers).length > 0 && (
                <div className="pt-4 border-t">
                  <p className="text-sm font-semibold mb-2">Timers (seconds)</p>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {Object.entries(metrics.metrics.timers).map(([key, value]) => (
                      <MetricItem
                        key={key}
                        label={key.replace(/_/g, ' ')}
                        value={`${value.toFixed(2)}s`}
                      />
                    ))}
                  </div>
                </div>
              )}
            </div>
          </Card>
        )}

        {dashboard && dashboard.dashboard.top_identities.length > 0 && (
          <Card title="Top Identities">
            <div className="space-y-4">
              {dashboard.dashboard.top_identities.map((identity) => (
                <div
                  key={identity.profile_id}
                  className="p-4 border rounded-lg hover:bg-gray-50 transition-colors"
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-semibold">{identity.username}</p>
                      {identity.display_name && (
                        <p className="text-sm text-gray-600">{identity.display_name}</p>
                      )}
                    </div>
                    <div className="text-right">
                      <p className="text-sm text-gray-600">
                        {identity.total_videos} videos, {identity.total_posts} posts
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        )}
      </div>
    </PageLayout>
  );
};

export default DashboardPage;

