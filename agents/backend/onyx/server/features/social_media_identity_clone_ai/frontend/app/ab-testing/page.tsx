'use client';

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { useRouter } from 'next/navigation';
import { apiClient } from '@/lib/api/client';
import type { ABTest } from '@/types';
import PageLayout from '@/components/Layout/PageLayout';
import Card from '@/components/UI/Card';
import Button from '@/components/UI/Button';
import Input from '@/components/UI/Input';
import LoadingSpinner from '@/components/UI/LoadingSpinner';
import EmptyState from '@/components/UI/EmptyState';
import StatusBadge from '@/components/UI/StatusBadge';

const ABTestingPage = (): JSX.Element => {
  const router = useRouter();
  const queryClient = useQueryClient();
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    variants: [],
  });

  const { data: tests, isLoading } = useQuery<ABTest[]>('ab-tests', async () => {
    const tests: ABTest[] = [];
    return tests;
  });

  const createMutation = useMutation(() => apiClient.createABTest(formData), {
    onSuccess: () => {
      queryClient.invalidateQueries('ab-tests');
      setShowCreateForm(false);
      setFormData({ name: '', variants: [] });
    },
  });

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>): void => {
    e.preventDefault();
    createMutation.mutate();
  };

  const handleNameChange = (e: React.ChangeEvent<HTMLInputElement>): void => {
    setFormData({ ...formData, name: e.target.value });
  };

  const handleToggleForm = (): void => {
    setShowCreateForm(!showCreateForm);
  };

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
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-3xl font-bold">A/B Testing</h1>
          <Button onClick={handleToggleForm}>
            {showCreateForm ? 'Cancel' : 'Create Test'}
          </Button>
        </div>

        {showCreateForm && (
          <Card title="Create A/B Test" className="mb-8">
            <form onSubmit={handleSubmit}>
              <div className="space-y-4">
                <Input
                  label="Test Name"
                  value={formData.name}
                  onChange={handleNameChange}
                  required
                />
                <Button type="submit" isLoading={createMutation.isLoading} className="w-full">
                  Create Test
                </Button>
              </div>
            </form>
          </Card>
        )}

        {!tests || tests.length === 0 ? (
          <Card>
            <EmptyState
              title="No A/B tests"
              description="Create an A/B test to compare content performance"
              actionLabel="Create Test"
              onAction={handleToggleForm}
            />
          </Card>
        ) : (
          <div className="space-y-4">
            {tests.map((test) => (
              <Card key={test.test_id}>
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="font-semibold">{test.name}</h3>
                    <StatusBadge status={test.status} className="mt-2" />
                  </div>
                  <div className="flex gap-2">
                    <Button
                      variant="secondary"
                      onClick={() => router.push(`/ab-testing/${test.test_id}`)}
                      className="text-sm"
                    >
                      View Details
                    </Button>
                    <Button
                      variant="secondary"
                      onClick={() => apiClient.startABTest(test.test_id)}
                      className="text-sm"
                    >
                      Start
                    </Button>
                    <Button
                      variant="secondary"
                      onClick={() => apiClient.stopABTest(test.test_id)}
                      className="text-sm"
                    >
                      Stop
                    </Button>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        )}
      </div>
    </PageLayout>
  );
};

export default ABTestingPage;

