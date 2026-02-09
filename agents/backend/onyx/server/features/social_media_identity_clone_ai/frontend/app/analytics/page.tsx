'use client';

import { useQuery } from 'react-query';
import { apiClient } from '@/lib/api/client';
import PageLayout from '@/components/Layout/PageLayout';
import Card from '@/components/UI/Card';
import LoadingSpinner from '@/components/UI/LoadingSpinner';
import Tabs from '@/components/UI/Tabs';
import StatsCard from '@/components/UI/StatsCard';
import { formatNumber } from '@/lib/utils';

const AnalyticsPage = (): JSX.Element => {
  const { data: stats, isLoading: statsLoading } = useQuery('analytics-stats', () =>
    apiClient.getAnalyticsStats()
  );

  const { data: trends, isLoading: trendsLoading } = useQuery('analytics-trends', () =>
    apiClient.getAnalyticsTrends()
  );

  const isLoading = statsLoading || trendsLoading;

  const tabs = [
    {
      id: 'stats',
      label: 'Statistics',
      content: (
        <div className="space-y-6">
          {stats && typeof stats === 'object' && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {Object.entries(stats as Record<string, unknown>).map(([key, value]) => (
                <StatsCard
                  key={key}
                  label={key.replace(/_/g, ' ')}
                  value={typeof value === 'number' ? formatNumber(value) : String(value)}
                />
              ))}
            </div>
          )}
        </div>
      ),
    },
    {
      id: 'trends',
      label: 'Trends',
      content: (
        <div className="space-y-6">
          {trends && typeof trends === 'object' && (
            <Card>
              <pre className="text-sm overflow-auto">
                {JSON.stringify(trends, null, 2)}
              </pre>
            </Card>
          )}
        </div>
      ),
    },
  ];

  if (isLoading) {
    return (
      <PageLayout>
        <LoadingSpinner />
      </PageLayout>
    );
  }

  return (
    <PageLayout>
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-8">Analytics</h1>
        <Tabs tabs={tabs} />
      </div>
    </PageLayout>
  );
};

export default AnalyticsPage;



