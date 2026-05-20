import { useCallback } from 'react';
import { Task } from '../types/task';
import { useNotifications } from './useNotifications';

interface UseKanbanWebSocketHandlersProps {
  soundEnabled: boolean;
  onActivityUpdate: (activity: { task: Task; action: string; timestamp: Date }) => void;
}

/**
 * Hook to handle WebSocket event callbacks for task updates
 */
export function useKanbanWebSocketHandlers({
  soundEnabled,
  onActivityUpdate,
}: UseKanbanWebSocketHandlersProps) {
  const { addNotification } = useNotifications({ maxNotifications: 100 });

  const playNotificationSound = useCallback(() => {
    if (soundEnabled && typeof window !== 'undefined' && 'Audio' in window) {
      try {
        const audio = new Audio('data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBSuBzvLZiTYIG2m98OSfTgwOUKfk8LZjHAY4kdfyzHksBSR3x/DdkEAKFF606euoVRQKRp/g8r5sIQUrgc7y2Yk2CBtpvfDkn04MDlCn5PC2YxwGOJHX8sx5LAUkd8fw3ZBAC');
        audio.volume = 0.3;
        audio.play().catch(() => {}); // Ignorar errores de autoplay
      } catch (e) {
        // Ignorar errores de audio
      }
    }
  }, [soundEnabled]);

  const handleTaskUpdate = useCallback((task: Task) => {
    console.log('📡 Evento WebSocket: tarea actualizada', task.id);
    onActivityUpdate({
      task,
      action: 'actualizada',
      timestamp: new Date()
    });
    
    // Notificación solo para cambios importantes
    if (task.status === 'completed' || task.status === 'failed') {
      addNotification(
        task.status === 'completed' ? 'success' : 'error',
        `Tarea ${task.status === 'completed' ? 'completada' : 'fallida'}`,
        `${task.repository}: ${task.instruction.substring(0, 50)}...`
      );
    }
  }, [onActivityUpdate, addNotification]);

  const handleTaskCreated = useCallback((task: Task) => {
    console.log('📡 Evento WebSocket: tarea creada', task.id);
    onActivityUpdate({
      task,
      action: 'creada',
      timestamp: new Date()
    });
    
    addNotification(
      'info',
      'Nueva tarea creada',
      `${task.repository}: ${task.instruction.substring(0, 50)}...`
    );
    
    playNotificationSound();
  }, [onActivityUpdate, addNotification, playNotificationSound]);

  const handleTaskCompleted = useCallback((task: Task) => {
    console.log('📡 Evento WebSocket: tarea completada', task.id);
    onActivityUpdate({
      task,
      action: 'completada',
      timestamp: new Date()
    });
    
    addNotification(
      'success',
      'Tarea completada',
      `${task.repository}: ${task.instruction.substring(0, 50)}...`
    );
  }, [onActivityUpdate, addNotification]);

  const handleTaskFailed = useCallback((task: Task) => {
    console.log('📡 Evento WebSocket: tarea fallida', task.id);
    onActivityUpdate({
      task,
      action: 'fallida',
      timestamp: new Date()
    });
    
    addNotification(
      'error',
      'Tarea fallida',
      `${task.repository}: ${task.error || task.instruction.substring(0, 50)}...`
    );
  }, [onActivityUpdate, addNotification]);

  return {
    handleTaskUpdate,
    handleTaskCreated,
    handleTaskCompleted,
    handleTaskFailed,
  };
}

