import { useMemo } from 'react';
import { Task } from '../types/task';
import { KANBAN_COLUMNS } from '../constants/task-constants';

interface UseKanbanSearchProps {
  searchQuery: string;
  tasks: Task[];
  repositories: string[];
}

export function useKanbanSearch({
  searchQuery,
  tasks,
  repositories,
}: UseKanbanSearchProps) {
  const searchSuggestions = useMemo(() => {
    if (!searchQuery || searchQuery.length < 2) return [];

    const query = searchQuery.toLowerCase();
    const suggestions: string[] = [];

    // Sugerencias de repositorios
    repositories.forEach((repo) => {
      if (repo.toLowerCase().includes(query) && !query.startsWith('repo:')) {
        suggestions.push(`repo:${repo}`);
      }
    });

    // Sugerencias de estados
    KANBAN_COLUMNS.forEach((col) => {
      if (col.label.toLowerCase().includes(query) && !query.startsWith('status:')) {
        suggestions.push(`status:${col.id}`);
      }
    });

    // Sugerencias de modelos
    const models = Array.from(new Set(tasks.map((t) => t.model).filter(Boolean)));
    models.forEach((model) => {
      if (model && model.toLowerCase().includes(query) && !query.startsWith('model:')) {
        suggestions.push(`model:${model}`);
      }
    });

    return suggestions.slice(0, 5);
  }, [searchQuery, repositories, tasks]);

  return {
    searchSuggestions,
  };
}

