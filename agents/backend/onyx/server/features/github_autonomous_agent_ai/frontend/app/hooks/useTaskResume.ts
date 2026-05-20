import { useEffect, useState } from 'react';
import { Task } from '../types/task';
import { useTaskStore } from '../store/task-store';
import { useTaskProcessing } from './useTaskProcessing';
import { canResumeTask } from '../utils/task-helpers';

/**
 * Hook para manejar la reanudación automática de tareas al cargar la página
 */
export function useTaskResume() {
  const { tasks, isLoading, updateTask } = useTaskStore();
  const { resumeTasks } = useTaskProcessing({ updateTask });
  const [hasResumedTasks, setHasResumedTasks] = useState(false);

  useEffect(() => {
    if (!isLoading && tasks.length > 0 && !hasResumedTasks) {
      // Reanudar tareas pendientes Y las que están procesando pero sin controller activo
      // Esto asegura que las tareas continúen procesando incluso después de recargar
      const tasksToResume = tasks.filter(task => {
        if (!task.repoInfo) return false;
        
        // NO reanudar tareas explícitamente detenidas por el usuario
        if (task.status === 'stopped') return false;
        
        // Reanudar pendientes
        if (task.status === 'pending' && canResumeTask(task)) return true;
        
        // IMPORTANTE: Reanudar las que están procesando para asegurar que el backend las continúe
        // Esto es crítico para que no se pierda el progreso al recargar
        if (task.status === 'processing' || task.status === 'running') {
          console.log(`🔄 Tarea en procesamiento detectada al recargar: ${task.id}, reanudando en backend...`);
          return true;
        }
        
        // También reanudar si tiene plan aprobado pero sigue en processing
        if (task.status === 'processing' && task.pendingApproval?.approved) {
          console.log(`🔄 Tarea con plan aprobado detectada: ${task.id}, asegurando continuidad...`);
          return true;
        }
        
        return false;
      });
      
      if (tasksToResume.length > 0) {
        console.log(`🔄 Reanudando ${tasksToResume.length} tarea(s) después de recargar la página...`);
        setTimeout(() => {
          resumeTasks(tasksToResume);
          setHasResumedTasks(true);
        }, 1000); // Reducido de 1500 a 1000ms para reanudar más rápido
      } else {
        setHasResumedTasks(true);
      }
    }
  }, [isLoading, tasks, hasResumedTasks, resumeTasks]);

  return { hasResumedTasks };
}

