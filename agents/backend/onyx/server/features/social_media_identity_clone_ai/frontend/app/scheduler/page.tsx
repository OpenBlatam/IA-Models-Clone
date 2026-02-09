'use client';

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { apiClient } from '@/lib/api/client';
import PageLayout from '@/components/Layout/PageLayout';
import Card from '@/components/UI/Card';
import Button from '@/components/UI/Button';
import Input from '@/components/UI/Input';
import Select from '@/components/UI/Select';
import LoadingSpinner from '@/components/UI/LoadingSpinner';
import EmptyState from '@/components/UI/EmptyState';
import { formatDate } from '@/lib/utils';

interface Schedule {
  schedule_id: string;
  identity_id: string;
  schedule_type: string;
  config: Record<string, unknown>;
  created_at: string;
}

const SchedulerPage = (): JSX.Element => {
  const queryClient = useQueryClient();
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [formData, setFormData] = useState({
    identity_id: '',
    schedule_type: 'daily',
    config: {},
  });

  const { data: schedules, isLoading } = useQuery<Schedule[]>('schedules', () =>
    apiClient.getSchedules()
  );

  const createMutation = useMutation(
    () =>
      apiClient.createSchedule(formData.identity_id, formData.schedule_type, formData.config),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('schedules');
        setShowCreateForm(false);
        setFormData({ identity_id: '', schedule_type: 'daily', config: {} });
      },
    }
  );

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>): void => {
    e.preventDefault();
    createMutation.mutate();
  };

  const handleIdentityIdChange = (e: React.ChangeEvent<HTMLInputElement>): void => {
    setFormData({ ...formData, identity_id: e.target.value });
  };

  const handleScheduleTypeChange = (e: React.ChangeEvent<HTMLSelectElement>): void => {
    setFormData({ ...formData, schedule_type: e.target.value });
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
          <h1 className="text-3xl font-bold">Scheduler</h1>
          <Button onClick={handleToggleForm}>
            {showCreateForm ? 'Cancel' : 'Create Schedule'}
          </Button>
        </div>

        {showCreateForm && (
          <Card title="Create New Schedule" className="mb-8">
            <form onSubmit={handleSubmit}>
              <div className="space-y-4">
                <Input
                  label="Identity ID"
                  value={formData.identity_id}
                  onChange={handleIdentityIdChange}
                  required
                />
                <Select
                  label="Schedule Type"
                  value={formData.schedule_type}
                  onChange={handleScheduleTypeChange}
                  options={[
                    { value: 'daily', label: 'Daily' },
                    { value: 'weekly', label: 'Weekly' },
                    { value: 'monthly', label: 'Monthly' },
                    { value: 'custom', label: 'Custom' },
                  ]}
                />
                <Button type="submit" isLoading={createMutation.isLoading} className="w-full">
                  Create Schedule
                </Button>
              </div>
            </form>
          </Card>
        )}

        {!schedules || schedules.length === 0 ? (
          <Card>
            <EmptyState
              title="No schedules"
              description="Create a schedule to automate content generation"
              actionLabel="Create Schedule"
              onAction={handleToggleForm}
            />
          </Card>
        ) : (
          <div className="space-y-4">
            {schedules.map((schedule) => (
              <Card key={schedule.schedule_id}>
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="font-semibold">Schedule {schedule.schedule_id}</h3>
                    <p className="text-sm text-gray-600">Identity: {schedule.identity_id}</p>
                    <p className="text-sm text-gray-600">Type: {schedule.schedule_type}</p>
                    <p className="text-xs text-gray-500">{formatDate(schedule.created_at)}</p>
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

export default SchedulerPage;



