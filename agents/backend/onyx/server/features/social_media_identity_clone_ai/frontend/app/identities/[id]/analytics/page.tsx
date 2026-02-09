'use client';

import { useQuery } from 'react-query';
import { useParams } from 'next/navigation';
import { apiClient } from '@/lib/api/client';
import PageLayout from '@/components/Layout/PageLayout';
import Card from '@/components/UI/Card';
import LoadingSpinner from '@/components/UI/LoadingSpinner';
import StatsCard from '@/components/UI/StatsCard';
import { formatNumber } from '@/lib/utils';

const IdentityAnalyticsPage = (): JSX.Element => {
  const params = useParams();
  const identityId = params.id as string;

  const { data: analytics, isLoading } = useQuery(
    ['identity-analytics', identityId],
    () => apiClient.getAnalyticsForIdentity(identityId),
    { enabled: !!identityId, staleTime: 30000 }
  );

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
        <h1 className="text-3xl font-bold mb-8">Identity Analytics</h1>
        {analytics && typeof analytics === 'object' ? (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {Object.entries(analytics as Record<string, unknown>).map(([key, value]) => (
                <StatsCard
                  key={key}
                  label={key.replace(/_/g, ' ')}
                  value={typeof value === 'number' ? formatNumber(value) : String(value)}
                />
              ))}
            </div>
            <Card title="Detailed Analytics">
              <pre className="text-sm overflow-auto">
                {JSON.stringify(analytics, null, 2)}
              </pre>
            </Card>
          </div>
        ) : (
          <Card>
            <p className="text-center text-gray-600 py-8">No analytics data available</p>
          </Card>
        )}
      </div>
    </PageLayout>
  );
};

export default IdentityAnalyticsPage;



