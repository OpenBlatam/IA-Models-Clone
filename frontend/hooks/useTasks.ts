'use client';

import { useState, useEffect, useCallback } from 'react';
import { apiClient } from '@/lib/api-client';
import { showToast } from '@/lib/toast';
import type { TaskListItem } from '@/types/api';

interface UseTasksOptions {
  autoRefresh?: boolean;
  refreshInterval?: number;
  initialPage?: number;
  itemsPerPage?: number;
}

export function useTasks(options: UseTasksOptions = {}) {
  const {
    autoRefresh = false,
    refreshInterval = 30000,
    initialPage = 1,
    itemsPerPage = 20,
  } = options;

  const [tasks, setTasks] = useState<TaskListItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [currentPage, setCurrentPage] = useState(initialPage);
  const [totalPages, setTotalPages] = useState(1);
  const [totalItems, setTotalItems] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const loadTasks = useCallback(async (showRefresh = false, page = currentPage) => {
    if (showRefresh) {
      setIsRefreshing(true);
    } else {
      setIsLoading(true);
    }
    setError(null);

    try {
      const response = await apiClient.listTasks({
        page,
        limit: itemsPerPage,
      });
      
      setTasks(response.tasks);
      setTotalPages(response.pagination?.total_pages || 1);
      setTotalItems(response.pagination?.total_items || response.tasks.length);
    } catch (err: any) {
      setError(err.message || 'Error al cargar tareas');
      showToast(err.message || 'Error al cargar tareas', 'error');
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
    }
  }, [currentPage, itemsPerPage]);

  useEffect(() => {
    loadTasks(false, currentPage);
  }, [currentPage]);

  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      loadTasks(true, currentPage);
    }, refreshInterval);

    return () => clearInterval(interval);
  }, [autoRefresh, refreshInterval, currentPage, loadTasks]);

  const refresh = useCallback(() => {
    loadTasks(true, currentPage);
  }, [loadTasks, currentPage]);

  const deleteTask = useCallback(async (taskId: string) => {
    try {
      await apiClient.deleteTask(taskId);
      showToast('Tarea eliminada exitosamente', 'success');
      loadTasks(false, currentPage);
    } catch (err: any) {
      showToast(err.message || 'Error al eliminar tarea', 'error');
    }
  }, [loadTasks, currentPage]);

  const cancelTask = useCallback(async (taskId: string) => {
    try {
      await apiClient.cancelTask(taskId);
      showToast('Tarea cancelada', 'success');
      loadTasks(false, currentPage);
    } catch (err: any) {
      showToast(err.message || 'Error al cancelar tarea', 'error');
    }
  }, [loadTasks, currentPage]);

  return {
    tasks,
    isLoading,
    isRefreshing,
    error,
    currentPage,
    totalPages,
    totalItems,
    setCurrentPage,
    loadTasks,
    refresh,
    deleteTask,
    cancelTask,
  };
}

