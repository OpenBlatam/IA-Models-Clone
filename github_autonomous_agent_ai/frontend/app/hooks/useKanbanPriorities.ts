import { useState, useEffect } from 'react';
import { Task } from '../types/task';

export function useKanbanPriorities(tasks: Task[]) {
  const [taskPriorities, setTaskPriorities] = useState<Map<string, 'low' | 'medium' | 'high' | 'urgent'>>(new Map());
  const [priorityFilter, setPriorityFilter] = useState<string>('all');

  // Detectar prioridades automáticamente
  useEffect(() => {
    const newPriorities = new Map<string, 'low' | 'medium' | 'high' | 'urgent'>();
    tasks.forEach(task => {
      const text = `${task.instruction} ${task.error || ''}`.toLowerCase();
      let priority: 'low' | 'medium' | 'high' | 'urgent' = 'medium';
      
      if (text.includes('urgent') || text.includes('crítico') || text.includes('critical')) {
        priority = 'urgent';
      } else if (text.includes('important') || text.includes('importante') || task.status === 'failed') {
        priority = 'high';
      } else if (text.includes('low') || text.includes('bajo') || task.status === 'completed') {
        priority = 'low';
      }
      
      newPriorities.set(task.id, priority);
    });
    setTaskPriorities(newPriorities);
  }, [tasks]);

  return {
    taskPriorities,
    setTaskPriorities,
    priorityFilter,
    setPriorityFilter,
  };
}

