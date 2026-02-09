'use client';

import { useQuery, useMutation, useQueryClient } from 'react-query';
import { useParams, useRouter } from 'next/navigation';
import { apiClient } from '@/lib/api/client';
import type { ABTest } from '@/types';
import PageLayout from '@/components/Layout/PageLayout';
import Card from '@/components/UI/Card';
import Button from '@/components/UI/Button';
import LoadingSpinner from '@/components/UI/LoadingSpinner';
import StatusBadge from '@/components/UI/StatusBadge';

const ABTestDetailPage = (): JSX.Element => {
  const params = useParams();
  const router = useRouter();
  const queryClient = useQueryClient();
  const testId = params.id as string;

  const { data: test, isLoading } = useQuery<ABTest>(
    ['ab-test', testId],
    () => apiClient.getABTest(testId),
    { enabled: !!testId }
  );

  const { data: winner, isLoading: winnerLoading } = useQuery(
    ['ab-test-winner', testId],
    () => apiClient.getABTestWinner(testId),
    { enabled: !!testId }
  );

  const startMutation = useMutation(() => apiClient.startABTest(testId), {
    onSuccess: () => {
      queryClient.invalidateQueries(['ab-test', testId]);
    },
  });

  const stopMutation = useMutation(() => apiClient.stopABTest(testId), {
    onSuccess: () => {
      queryClient.invalidateQueries(['ab-test', testId]);
    },
  });

  const handleBack = (): void => {
    router.push('/ab-testing');
  };

  const handleStart = (): void => {
    startMutation.mutate();
  };

  const handleStop = (): void => {
    stopMutation.mutate();
  };

  if (isLoading) {
    return (
      <PageLayout>
        <LoadingSpinner />
      </PageLayout>
    );
  }

  if (!test) {
    return (
      <PageLayout>
        <Card>
          <p className="text-center text-gray-600 py-8">A/B Test not found</p>
          <div className="text-center">
            <Button onClick={handleBack}>Back to A/B Tests</Button>
          </div>
        </Card>
      </PageLayout>
    );
  }

  return (
    <PageLayout>
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-3xl font-bold">A/B Test Details</h1>
          <Button variant="secondary" onClick={handleBack}>
            Back
          </Button>
        </div>

        <Card>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-2xl font-semibold">{test.name}</h2>
              <StatusBadge status={test.status} />
            </div>

            {test.description && (
              <div>
                <p className="text-sm text-gray-600 mb-1">Description</p>
                <p className="text-gray-900">{test.description}</p>
              </div>
            )}

            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-gray-600 mb-1">Test ID</p>
                <p className="text-sm font-mono">{test.test_id}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600 mb-1">Status</p>
                <StatusBadge status={test.status} />
              </div>
            </div>

            {test.variants && test.variants.length > 0 && (
              <div>
                <p className="text-sm text-gray-600 mb-2">Variants</p>
                <div className="space-y-2">
                  {test.variants.map((variant, idx) => (
                    <div key={idx} className="p-3 bg-gray-50 rounded">
                      <p className="font-semibold">Variant {idx + 1}</p>
                      {typeof variant === 'object' && (
                        <pre className="text-xs mt-1 overflow-auto">
                          {JSON.stringify(variant, null, 2)}
                        </pre>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {winner && (
              <div className="p-4 bg-green-50 rounded-lg">
                <p className="font-semibold text-green-800 mb-2">Winner</p>
                <pre className="text-sm overflow-auto">
                  {JSON.stringify(winner, null, 2)}
                </pre>
              </div>
            )}

            <div className="flex gap-2 pt-4 border-t">
              <Button
                variant="primary"
                onClick={handleStart}
                isLoading={startMutation.isLoading}
                disabled={test.status === 'running'}
              >
                Start Test
              </Button>
              <Button
                variant="secondary"
                onClick={handleStop}
                isLoading={stopMutation.isLoading}
                disabled={test.status !== 'running'}
              >
                Stop Test
              </Button>
            </div>
          </div>
        </Card>
      </div>
    </PageLayout>
  );
};

export default ABTestDetailPage;



