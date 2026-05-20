import { useState, useEffect } from 'react';
import { Task } from '../types/task';

export interface ActivityItem {
  task: Task;
  action: string;
  timestamp: Date;
}

export function useRecentActivity(tasks: Task[]) {
  const [recentActivity, setRecentActivity] = useState<ActivityItem[]>([]);

  useEffect(() => {
    const newActivity: ActivityItem[] = [];

    // Tareas recién creadas (últimas 10)
    const recentTasks = [...tasks]
      .sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime())
      .slice(0, 10);

    recentTasks.forEach((task) => {
      const createdTime = new Date(task.createdAt);
      const now = new Date();
      const diffMinutes = Math.floor((now.getTime() - createdTime.getTime()) / 1000 / 60);

      if (diffMinutes < 60) {
        newActivity.push({
          task,
          action: 'creada',
          timestamp: createdTime,
        });
      }
    });

    // Tareas completadas recientemente
    const completedTasks = tasks
      .filter((t) => t.status === 'completed')
      .sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime())
      .slice(0, 5);

    completedTasks.forEach((task) => {
      newActivity.push({
        task,
        action: 'completada',
        timestamp: new Date(task.createdAt),
      });
    });

    setRecentActivity(
      newActivity.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime()).slice(0, 20)
    );
  }, [tasks]);

  const addActivity = (activity: ActivityItem) => {
    setRecentActivity((prev) => [activity, ...prev].slice(0, 20));
  };

  return {
    recentActivity,
    addActivity,
  };
}

