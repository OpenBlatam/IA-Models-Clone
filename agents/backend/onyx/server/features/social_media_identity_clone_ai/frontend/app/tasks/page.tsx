'use client';

import { useQuery } from 'react-query';
import { apiClient } from '@/lib/api/client';
import { TaskStatus, type Task } from '@/types';
import PageLayout from '@/components/Layout/PageLayout';
import Card from '@/components/UI/Card';
import LoadingSpinner from '@/components/UI/LoadingSpinner';
import EmptyState from '@/components/UI/EmptyState';
import { formatDate } from '@/lib/utils';
import { REFETCH_INTERVALS } from '@/lib/constants';
import StatusBadge from '@/components/UI/StatusBadge';

const TasksPage = (): JSX.Element => {
  const { data: tasks, isLoading } = useQuery<Task[]>('tasks', () => apiClient.getTasks(), {
    refetchInterval: REFETCH_INTERVALS.TASKS,
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
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-8">Tasks</h1>

        {!tasks || tasks.length === 0 ? (
          <Card>
            <EmptyState title="No tasks found" description="Tasks will appear here when created" />
          </Card>
        ) : (
          <div className="space-y-4">
            {tasks.map((task) => (
              <Card key={task.task_id}>
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-4 mb-2">
                      <StatusBadge status={task.status} />
                      <span className="text-sm text-gray-600 font-mono" aria-label="Task ID">
                        {task.task_id}
                      </span>
                    </div>
                    {task.error && (
                      <div
                        className="mt-2 p-2 bg-red-50 border border-red-200 rounded text-sm text-red-700"
                        role="alert"
                        aria-live="polite"
                      >
                        {task.error}
                      </div>
                    )}
                    <div className="flex gap-4 mt-2 text-xs text-gray-500">
                      <span>Created: {formatDate(task.created_at)}</span>
                      <span>Updated: {formatDate(task.updated_at)}</span>
                    </div>
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

export default TasksPage;

