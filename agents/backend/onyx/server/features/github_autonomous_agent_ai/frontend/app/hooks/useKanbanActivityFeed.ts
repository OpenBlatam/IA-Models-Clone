import { useState, useEffect, useMemo } from 'react';
import { Task } from '../types/task';

export interface ActivityFeedItem {
  id: string;
  type: string;
  message: string;
  timestamp: Date;
  taskId?: string;
}

export function useKanbanActivityFeed(tasks: Task[]) {
  const [activityFeed, setActivityFeed] = useState<ActivityFeedItem[]>([]);
  const [showActivityFeed, setShowActivityFeed] = useState(false);

  // Feed de actividad mejorado
  useEffect(() => {
    const newActivities: ActivityFeedItem[] = [];
    
    // Actividad: Tareas creadas
    tasks.filter(t => {
      const created = new Date(t.createdAt);
      return (Date.now() - created.getTime()) < 3600000; // Última hora
    }).forEach(task => {
      newActivities.push({
        id: `created-${task.id}`,
        type: 'created',
        message: `Nueva tarea creada: ${task.repository}`,
        timestamp: new Date(task.createdAt),
        taskId: task.id
      });
    });
    
    // Actividad: Tareas completadas
    tasks.filter(t => 
      t.status === 'completed' && 
      new Date(t.updatedAt || t.createdAt).getTime() > Date.now() - 3600000
    ).forEach(task => {
      newActivities.push({
        id: `completed-${task.id}`,
        type: 'completed',
        message: `Tarea completada: ${task.repository}`,
        timestamp: new Date(task.updatedAt || t.createdAt),
        taskId: task.id
      });
    });
    
    // Actividad: Tareas fallidas
    tasks.filter(t => 
      t.status === 'failed' && 
      new Date(t.updatedAt || t.createdAt).getTime() > Date.now() - 3600000
    ).forEach(task => {
      newActivities.push({
        id: `failed-${task.id}`,
        type: 'failed',
        message: `Tarea fallida: ${task.repository}`,
        timestamp: new Date(task.updatedAt || t.createdAt),
        taskId: task.id
      });
    });
    
    setActivityFeed(newActivities.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime()).slice(0, 50));
  }, [tasks]);

  return {
    activityFeed,
    showActivityFeed,
    setShowActivityFeed,
  };
}

