/**
 * Hook para optimistic updates - actualizar UI inmediatamente antes de confirmación del backend.
 */

import { useCallback, useRef } from 'react';
import { useTaskStore } from '../store/task-store';
import { Task } from '../types/task';

interface OptimisticUpdate {
  taskId: string;
  updates: Partial<Task>;
  timestamp: number;
}

/**
 * Hook para manejar optimistic updates.
 */
export function useOptimisticUpdates() {
  const { updateTask } = useTaskStore();
  const pendingUpdatesRef = useRef<Map<string, OptimisticUpdate>>(new Map());

  /**
   * Aplicar actualización optimista.
   */
  const applyOptimisticUpdate = useCallback((
    taskId: string,
    updates: Partial<Task>
  ) => {
    // Guardar update pendiente
    pendingUpdatesRef.current.set(taskId, {
      taskId,
      updates,
      timestamp: Date.now()
    });

    // Aplicar inmediatamente
    updateTask(taskId, updates);
  }, [updateTask]);

  /**
   * Confirmar actualización optimista (cuando el backend confirma).
   */
  const confirmUpdate = useCallback((taskId: string) => {
    pendingUpdatesRef.current.delete(taskId);
  }, []);

  /**
   * Revertir actualización optimista (cuando el backend falla).
   */
  const revertUpdate = useCallback((
    taskId: string,
    originalTask: Task
  ) => {
    const pending = pendingUpdatesRef.current.get(taskId);
    if (pending) {
      // Revertir a estado original
      updateTask(taskId, originalTask);
      pendingUpdatesRef.current.delete(taskId);
    }
  }, [updateTask]);

  /**
   * Obtener updates pendientes.
   */
  const getPendingUpdates = useCallback(() => {
    return Array.from(pendingUpdatesRef.current.values());
  }, []);

  /**
   * Limpiar updates antiguos (más de 5 minutos).
   */
  const cleanupOldUpdates = useCallback(() => {
    const fiveMinutesAgo = Date.now() - 5 * 60 * 1000;
    for (const [taskId, update] of pendingUpdatesRef.current.entries()) {
      if (update.timestamp < fiveMinutesAgo) {
        pendingUpdatesRef.current.delete(taskId);
      }
    }
  }, []);

  return {
    applyOptimisticUpdate,
    confirmUpdate,
    revertUpdate,
    getPendingUpdates,
    cleanupOldUpdates
  };
}



