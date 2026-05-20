import { useState, useEffect } from 'react';
import { Task } from '../types/task';

export interface TaskHistoryEntry {
  timestamp: Date;
  changes: Partial<Task>;
}

export function useKanbanTaskHistory(tasks: Task[]) {
  const [taskHistory, setTaskHistory] = useState<Map<string, TaskHistoryEntry[]>>(new Map());

  // Trackear cambios en tareas para historial
  useEffect(() => {
    tasks.forEach(task => {
      const existingHistory = taskHistory.get(task.id) || [];
      const lastChange = existingHistory[existingHistory.length - 1];
      
      // Detectar cambios significativos
      if (!lastChange || 
          lastChange.changes.status !== task.status ||
          lastChange.changes.error !== task.error) {
        setTaskHistory(prev => {
          const newHistory = new Map(prev);
          const taskHistory = newHistory.get(task.id) || [];
          newHistory.set(task.id, [
            ...taskHistory,
            {
              timestamp: new Date(),
              changes: {
                status: task.status,
                error: task.error,
                streamingContent: task.streamingContent,
              }
            }
          ].slice(-50)); // Mantener solo últimos 50 cambios
          return newHistory;
        });
      }
    });
  }, [tasks, taskHistory]);

  return {
    taskHistory,
  };
}

