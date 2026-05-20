import { useMemo, useCallback } from 'react';
import { Task, TaskStatus } from '../types/task';

export function useKanbanStats(tasks: Task[]) {
  const getTasksByStatus = useCallback((status: TaskStatus) => {
    return tasks.filter((task) => task.status === status);
  }, [tasks]);
  
  const stats = useMemo(() => {
    const now = Date.now();
    const today = new Date(now).setHours(0, 0, 0, 0);
    const weekAgo = now - 7 * 24 * 60 * 60 * 1000;
    const monthAgo = now - 30 * 24 * 60 * 60 * 1000;
    
    const statsData = {
      total: tasks.length,
      pending: getTasksByStatus('pending').length,
      processing: getTasksByStatus('processing').length,
      running: getTasksByStatus('running').length,
      completed: getTasksByStatus('completed').length,
      failed: getTasksByStatus('failed').length,
      stopped: getTasksByStatus('stopped').length,
      withCommits: tasks.filter(t => t.executionResult?.commitSha).length,
      withErrors: tasks.filter(t => t.error).length,
      createdToday: tasks.filter(t => new Date(t.createdAt).getTime() >= today).length,
      createdThisWeek: tasks.filter(t => new Date(t.createdAt).getTime() >= weekAgo).length,
      createdThisMonth: tasks.filter(t => new Date(t.createdAt).getTime() >= monthAgo).length,
      avgProcessingTime: (() => {
        const completedTasks = tasks.filter(t => t.status === 'completed' && t.processingStartedAt);
        if (completedTasks.length === 0) return 0;
        const totalTime = completedTasks.reduce((acc, t) => {
          if (t.processingStartedAt) {
            const start = new Date(t.processingStartedAt).getTime();
            const end = new Date(t.createdAt).getTime() + (24 * 60 * 60 * 1000); // Estimado
            return acc + (end - start);
          }
          return acc;
        }, 0);
        return Math.round(totalTime / completedTasks.length / 1000 / 60); // minutos
      })(),
      successRate: (() => {
        const totalProcessed = tasks.filter(t => t.status === 'completed' || t.status === 'failed').length;
        if (totalProcessed === 0) return 0;
        return Math.round((getTasksByStatus('completed').length / totalProcessed) * 100);
      })(),
    };
    return statsData;
  }, [tasks, getTasksByStatus]);
  
  return { stats, getTasksByStatus };
}

