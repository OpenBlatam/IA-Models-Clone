import { useMemo, useCallback } from 'react';
import { Task, TaskStatus } from '../types/task';

interface UseKanbanFiltersProps {
  tasks: Task[];
  searchQuery: string;
  selectedRepository: string;
  statusFilter: TaskStatus | 'all';
  dateFilter: 'all' | 'today' | 'week' | 'month' | 'custom';
  sortBy: 'date' | 'status' | 'repo' | 'model';
  sortOrder: 'asc' | 'desc';
}

export function useKanbanFilters({
  tasks,
  searchQuery,
  selectedRepository,
  statusFilter,
  dateFilter,
  sortBy,
  sortOrder,
}: UseKanbanFiltersProps) {
  const filteredTasks = useMemo(() => {
    let filtered = tasks;
    
    // Filtro por repositorio
    if (selectedRepository !== 'all') {
      filtered = filtered.filter(task => task.repository === selectedRepository);
    }
    
    // Filtro por estado
    if (statusFilter !== 'all') {
      filtered = filtered.filter(task => task.status === statusFilter);
    }
    
    // Filtro por fecha
    if (dateFilter !== 'all') {
      const now = new Date();
      const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
      const weekAgo = new Date(today);
      weekAgo.setDate(weekAgo.getDate() - 7);
      const monthAgo = new Date(today);
      monthAgo.setMonth(monthAgo.getMonth() - 1);
      
      filtered = filtered.filter(task => {
        const taskDate = new Date(task.createdAt);
        switch (dateFilter) {
          case 'today':
            return taskDate >= today;
          case 'week':
            return taskDate >= weekAgo;
          case 'month':
            return taskDate >= monthAgo;
          case 'custom':
            if (typeof window !== 'undefined') {
              const dateFrom = localStorage.getItem('kanban-date-from');
              const dateTo = localStorage.getItem('kanban-date-to');
              if (dateFrom) {
                const from = new Date(dateFrom);
                from.setHours(0, 0, 0, 0);
                if (taskDate < from) return false;
              }
              if (dateTo) {
                const to = new Date(dateTo);
                to.setHours(23, 59, 59, 999);
                if (taskDate > to) return false;
              }
              return true;
            }
            return true;
          default:
            return true;
        }
      });
    }
    
    // Filtro por búsqueda avanzada
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      
      // Búsqueda avanzada con operadores
      if (query.startsWith('repo:')) {
        const repoQuery = query.replace('repo:', '').trim();
        filtered = filtered.filter(task => task.repository.toLowerCase().includes(repoQuery));
      } else if (query.startsWith('status:')) {
        const statusQuery = query.replace('status:', '').trim();
        filtered = filtered.filter(task => task.status.toLowerCase() === statusQuery);
      } else if (query.startsWith('error:')) {
        const errorQuery = query.replace('error:', '').trim();
        filtered = filtered.filter(task => task.error && task.error.toLowerCase().includes(errorQuery));
      } else if (query.startsWith('model:')) {
        const modelQuery = query.replace('model:', '').trim();
        filtered = filtered.filter(task => task.model && task.model.toLowerCase().includes(modelQuery));
      } else {
        // Búsqueda general
        filtered = filtered.filter((task) => 
          task.instruction.toLowerCase().includes(query) ||
          task.repository.toLowerCase().includes(query) ||
          (task.error && task.error.toLowerCase().includes(query)) ||
          (task.model && task.model.toLowerCase().includes(query)) ||
          (task.streamingContent && typeof task.streamingContent === 'string' && 
           task.streamingContent.toLowerCase().includes(query))
        );
      }
    }
    
    // Ordenar
    const sorted = [...filtered].sort((a, b) => {
      let comparison = 0;
      
      switch (sortBy) {
        case 'date':
          comparison = new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime();
          break;
        case 'status':
          comparison = a.status.localeCompare(b.status);
          break;
        case 'repo':
          comparison = a.repository.localeCompare(b.repository);
          break;
        case 'model':
          const modelA = a.model || '';
          const modelB = b.model || '';
          comparison = modelA.localeCompare(modelB);
          break;
      }
      
      return sortOrder === 'asc' ? comparison : -comparison;
    });
    
    return sorted;
  }, [tasks, searchQuery, selectedRepository, statusFilter, dateFilter, sortBy, sortOrder]);
  
  return { filteredTasks };
}

