'use client';

import { useQuery } from 'react-query';
import { apiClient } from '@/lib/api/client';
import PageLayout from '@/components/Layout/PageLayout';
import Card from '@/components/UI/Card';
import LoadingSpinner from '@/components/UI/LoadingSpinner';
import StatusBadge from '@/components/UI/StatusBadge';

const HealthPage = (): JSX.Element => {
  const { data: health, isLoading } = useQuery('health', () => apiClient.healthCheck(), {
    refetchInterval: 30000,
  });

  if (isLoading) {
    return (
      <PageLayout>
        <LoadingSpinner />
      </PageLayout>
    );
  }

  return (
    <PageLayout>
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold mb-8">System Health</h1>
        <Card>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold">Status</h2>
              <StatusBadge
                status={health && typeof health === 'object' && 'status' in health ? String(health.status) : 'Unknown'}
                variant={health && typeof health === 'object' && 'status' in health && health.status === 'healthy' ? 'success' : 'error'}
              />
            </div>
            {health && typeof health === 'object' && (
              <div className="space-y-2">
                {Object.entries(health).map(([key, value]) => (
                  <div key={key} className="flex items-center justify-between py-2 border-b border-gray-100">
                    <span className="text-sm text-gray-600 capitalize">{key.replace(/_/g, ' ')}</span>
                    <span className="text-sm font-semibold text-gray-900">
                      {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </Card>
      </div>
    </PageLayout>
  );
};

export default HealthPage;



