import { useState, useEffect } from 'react';
import { Task } from '../types/task';

export interface Milestone {
  id: string;
  name: string;
  date: Date;
  tasks: string[];
}

export function useKanbanMilestones(tasks: Task[]) {
  const [milestones, setMilestones] = useState<Milestone[]>([]);
  const [showMilestones, setShowMilestones] = useState(false);

  // Generar milestones automáticamente
  useEffect(() => {
    const newMilestones: Milestone[] = [];
    
    // Milestone: Tareas completadas hoy
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const completedToday = tasks.filter(t => {
      if (t.status !== 'completed') return false;
      const completedDate = new Date(t.updatedAt || t.createdAt);
      completedDate.setHours(0, 0, 0, 0);
      return completedDate.getTime() === today.getTime();
    });
    
    if (completedToday.length > 0) {
      newMilestones.push({
        id: 'today',
        name: `Completadas hoy (${completedToday.length})`,
        date: today,
        tasks: completedToday.map(t => t.id)
      });
    }
    
    // Milestone: 10 tareas completadas
    const completed = tasks.filter(t => t.status === 'completed');
    if (completed.length >= 10 && completed.length % 10 === 0) {
      newMilestones.push({
        id: `milestone-${completed.length}`,
        name: `${completed.length} tareas completadas`,
        date: new Date(),
        tasks: completed.slice(-10).map(t => t.id)
      });
    }
    
    setMilestones(newMilestones);
  }, [tasks]);

  return {
    milestones,
    showMilestones,
    setShowMilestones,
  };
}

