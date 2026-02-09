'use client';

import { useState } from 'react';
import { useQuery } from 'react-query';
import { apiClient } from '@/lib/api/client';
import type { Recommendation } from '@/types';
import PageLayout from '@/components/Layout/PageLayout';
import Card from '@/components/UI/Card';
import Input from '@/components/UI/Input';
import Button from '@/components/UI/Button';
import LoadingSpinner from '@/components/UI/LoadingSpinner';
import EmptyState from '@/components/UI/EmptyState';
import Tabs from '@/components/UI/Tabs';

const RecommendationsPage = (): JSX.Element => {
  const [identityId, setIdentityId] = useState('');

  const { data: systemRecommendations, isLoading: systemLoading } = useQuery<Recommendation[]>(
    'system-recommendations',
    () => apiClient.getSystemRecommendations()
  );

  const { data: identityRecommendations, isLoading: identityLoading } = useQuery<Recommendation[]>(
    ['identity-recommendations', identityId],
    () => apiClient.getRecommendationsForIdentity(identityId),
    { enabled: identityId.length > 0 }
  );

  const handleIdentityIdChange = (e: React.ChangeEvent<HTMLInputElement>): void => {
    setIdentityId(e.target.value);
  };

  const tabs = [
    {
      id: 'system',
      label: 'System Recommendations',
      content: (
        <div className="space-y-4">
          {systemLoading ? (
            <LoadingSpinner />
          ) : !systemRecommendations || systemRecommendations.length === 0 ? (
            <EmptyState title="No recommendations" description="System recommendations will appear here" />
          ) : (
            systemRecommendations.map((rec) => (
              <Card key={rec.recommendation_id}>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <h3 className="font-semibold">{rec.title}</h3>
                    <span className="px-2 py-1 bg-primary-100 text-primary-700 rounded text-xs">
                      Priority: {rec.priority}
                    </span>
                  </div>
                  <p className="text-gray-600">{rec.description}</p>
                  <span className="text-xs text-gray-500">{rec.type}</span>
                </div>
              </Card>
            ))
          )}
        </div>
      ),
    },
    {
      id: 'identity',
      label: 'Identity Recommendations',
      content: (
        <div className="space-y-4">
          <Card>
            <Input
              label="Identity ID"
              value={identityId}
              onChange={handleIdentityIdChange}
              placeholder="Enter identity ID to get recommendations"
            />
          </Card>
          {identityId && (
            <>
              {identityLoading ? (
                <LoadingSpinner />
              ) : !identityRecommendations || identityRecommendations.length === 0 ? (
                <EmptyState
                  title="No recommendations"
                  description={`No recommendations found for identity ${identityId}`}
                />
              ) : (
                identityRecommendations.map((rec) => (
                  <Card key={rec.recommendation_id}>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <h3 className="font-semibold">{rec.title}</h3>
                        <span className="px-2 py-1 bg-primary-100 text-primary-700 rounded text-xs">
                          Priority: {rec.priority}
                        </span>
                      </div>
                      <p className="text-gray-600">{rec.description}</p>
                      <span className="text-xs text-gray-500">{rec.type}</span>
                    </div>
                  </Card>
                ))
              )}
            </>
          )}
        </div>
      ),
    },
  ];

  return (
    <PageLayout>
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-8">Recommendations</h1>
        <Tabs tabs={tabs} />
      </div>
    </PageLayout>
  );
};

export default RecommendationsPage;



