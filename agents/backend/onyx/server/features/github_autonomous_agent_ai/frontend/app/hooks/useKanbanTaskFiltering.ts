import { useMemo } from 'react';
import { Task } from '../types/task';
import { TaskStatus } from '../types/task';
import { KANBAN_COLUMNS } from '../constants/task-constants';

interface UseKanbanTaskFilteringProps {
  filteredTasks: Task[];
  tagFilter: string;
  taskTags: Map<string, string[]>;
  priorityFilter: string;
  taskPriorities: Map<string, 'low' | 'medium' | 'high' | 'urgent'>;
  searchInStreaming: boolean;
  searchQuery: string;
  errorFilter: string;
}

/**
 * Hook to handle all task filtering logic in the correct order
 */
export function useKanbanTaskFiltering({
  filteredTasks,
  tagFilter,
  taskTags,
  priorityFilter,
  taskPriorities,
  searchInStreaming,
  searchQuery,
  errorFilter,
}: UseKanbanTaskFilteringProps) {
  // Filtrar por tag (debe ir antes de tasksWithPriorityFilter)
  const tasksWithTagFilter = useMemo(() => {
    if (tagFilter === 'all') return filteredTasks;
    return filteredTasks.filter(task => {
      const taskTagsList = taskTags.get(task.id) || [];
      return taskTagsList.includes(tagFilter);
    });
  }, [filteredTasks, tagFilter, taskTags]);

  // Filtrar por prioridad (debe ir antes de tasksWithStreamingSearch)
  const tasksWithPriorityFilter = useMemo(() => {
    if (priorityFilter === 'all') return tasksWithTagFilter;
    return tasksWithTagFilter.filter(task => {
      const priority = taskPriorities.get(task.id) || 'medium';
      return priority === priorityFilter;
    });
  }, [tasksWithTagFilter, priorityFilter, taskPriorities]);

  // Búsqueda en contenido de streaming (debe ir antes de tasksWithErrorFilter)
  const tasksWithStreamingSearch = useMemo(() => {
    if (!searchInStreaming || !searchQuery) return tasksWithPriorityFilter;
    const query = searchQuery.toLowerCase();
    return tasksWithPriorityFilter.filter(task => {
      const streamingContent = typeof task.streamingContent === 'string' 
        ? task.streamingContent 
        : task.streamingContent 
          ? JSON.stringify(task.streamingContent) 
          : '';
      return streamingContent.toLowerCase().includes(query);
    });
  }, [tasksWithPriorityFilter, searchInStreaming, searchQuery]);

  // Filtrar por error (debe ir después de tasksWithStreamingSearch)
  const tasksWithErrorFilter = useMemo(() => {
    if (errorFilter === 'all') return tasksWithStreamingSearch;
    if (errorFilter === 'with-errors') {
      return tasksWithStreamingSearch.filter(t => t.error && t.error.length > 0);
    }
    if (errorFilter === 'no-errors') {
      return tasksWithStreamingSearch.filter(t => !t.error || t.error.length === 0);
    }
    return tasksWithStreamingSearch.filter(t => t.error && t.error.toLowerCase().includes(errorFilter.toLowerCase()));
  }, [tasksWithStreamingSearch, errorFilter]);

  // Memoizar las tareas por columna para evitar recálculos
  const tasksByColumn = useMemo(() => {
    const result: Record<TaskStatus, Task[]> = {} as Record<TaskStatus, Task[]>;
    KANBAN_COLUMNS.forEach((column) => {
      result[column.id] = tasksWithStreamingSearch.filter((task) => task.status === column.id);
    });
    return result;
  }, [tasksWithStreamingSearch]);

  return {
    tasksWithTagFilter,
    tasksWithPriorityFilter,
    tasksWithStreamingSearch,
    tasksWithErrorFilter,
    tasksByColumn,
  };
}

