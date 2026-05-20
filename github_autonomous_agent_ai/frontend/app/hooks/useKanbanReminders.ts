import { useState, useEffect } from 'react';
import { Task } from '../types/task';
import { useNotifications } from './useNotifications';

export interface Reminder {
  id: string;
  taskId: string;
  message: string;
  time: Date;
}

export function useKanbanReminders(tasks: Task[]) {
  const [reminders, setReminders] = useState<Reminder[]>([]);
  const [showReminders, setShowReminders] = useState(false);
  const { addNotification } = useNotifications({ maxNotifications: 100 });

  // Verificar recordatorios
  useEffect(() => {
    const checkReminders = () => {
      const now = new Date();
      reminders.forEach(reminder => {
        const reminderTime = new Date(reminder.time);
        if (reminderTime <= now && reminderTime > new Date(now.getTime() - 60000)) {
          const task = tasks.find(t => t.id === reminder.taskId);
          addNotification(
            'warning',
            'Recordatorio',
            reminder.message
          );
          // Remover recordatorio después de notificar
          setReminders(prev => prev.filter(r => r.id !== reminder.id));
        }
      });
    };
    
    const interval = setInterval(checkReminders, 30000); // Verificar cada 30 segundos
    return () => clearInterval(interval);
  }, [reminders, tasks, addNotification]);

  const addReminder = (taskId: string, message: string, time: Date) => {
    setReminders(prev => [...prev, {
      id: `reminder-${Date.now()}`,
      taskId,
      message,
      time,
    }]);
  };

  const deleteReminder = (reminderId: string) => {
    setReminders(prev => prev.filter(r => r.id !== reminderId));
  };

  return {
    reminders,
    showReminders,
    setShowReminders,
    addReminder,
    deleteReminder,
  };
}

