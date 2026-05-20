import { useMemo } from 'react';
import { Task } from '../types/task';

interface UseKanbanGroupingProps {
  tasks: Task[];
  groupByRepo: boolean;
  groupByModel: boolean;
}

export function useKanbanGrouping({
  tasks,
  groupByRepo,
  groupByModel,
}: UseKanbanGroupingProps) {
  const groupedTasks = useMemo(() => {
    if (!groupByRepo && !groupByModel) {
      return tasks;
    }

    let sorted = [...tasks];

    if (groupByRepo) {
      sorted = sorted.sort((a, b) => a.repository.localeCompare(b.repository));
    } else if (groupByModel) {
      sorted = sorted.sort((a, b) => {
        const modelA = a.model || 'Sin modelo';
        const modelB = b.model || 'Sin modelo';
        return modelA.localeCompare(modelB);
      });
    }

    return sorted;
  }, [tasks, groupByRepo, groupByModel]);

  const getGroupHeader = (task: Task, previousTask: Task | null): string | null => {
    if (groupByRepo && (!previousTask || task.repository !== previousTask.repository)) {
      return task.repository;
    }
    if (groupByModel && (!previousTask || (task.model || 'Sin modelo') !== (previousTask.model || 'Sin modelo'))) {
      return task.model || 'Sin modelo';
    }
    return null;
  };

  return {
    groupedTasks,
    getGroupHeader,
  };
}

