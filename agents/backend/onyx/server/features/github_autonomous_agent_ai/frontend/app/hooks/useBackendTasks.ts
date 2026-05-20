/**
 * Hook para sincronizar tareas con el backend.
 */

import { useEffect, useState, useCallback } from 'react';
import { useTaskStore } from '../store/task-store';
import { getAPIClient } from '../lib/api-client';
import { Task } from '../types/task';
import { toast } from 'sonner';

interface UseBackendTasksOptions {
  enabled?: boolean;
  syncInterval?: number; // Intervalo de sincronización en ms
  autoSync?: boolean; // Sincronizar automáticamente
}

/**
 * Hook para sincronizar tareas con el backend.
 */
export function useBackendTasks(options: UseBackendTasksOptions = {}) {
  const {
    enabled = true,
    syncInterval = 5000,
    autoSync = true
  } = options;

  const { tasks, setTasks, updateTask, addTask } = useTaskStore();
  const [isSyncing, setIsSyncing] = useState(false);
  const [lastSync, setLastSync] = useState<Date | null>(null);
  const [error, setError] = useState<string | null>(null);

  const syncTasks = useCallback(async () => {
    if (!enabled || isSyncing) return;

    try {
      setIsSyncing(true);
      setError(null);

      const client = getAPIClient();
      const backendTasks = await client.listTasks();

      // Convertir tareas del backend al formato del frontend
      const convertedTasks: Task[] = backendTasks.map((bt: any) => ({
        id: bt.id,
        repository: `${bt.repository_owner}/${bt.repository_name}`,
        instruction: bt.instruction,
        status: mapBackendStatus(bt.status),
        createdAt: bt.created_at,
        processingStartedAt: bt.started_at,
        result: bt.result ? {
          content: typeof bt.result === 'string' ? bt.result : JSON.stringify(bt.result),
          plan: bt.result.plan,
          code: bt.result.code
        } : undefined,
        error: bt.error,
        streamingContent: undefined
      }));

      // Sincronizar: actualizar existentes y agregar nuevas
      const existingIds = new Set(tasks.map(t => t.id));
      const newTasks = convertedTasks.filter(t => !existingIds.has(t.id));
      
      // Actualizar tareas existentes
      convertedTasks.forEach(backendTask => {
        const localTask = tasks.find(t => t.id === backendTask.id);
        if (localTask && hasChanges(localTask, backendTask)) {
          updateTask(backendTask.id, backendTask);
        }
      });

      // Agregar nuevas tareas
      newTasks.forEach(task => {
        addTask(task);
      });

      setLastSync(new Date());
    } catch (err: any) {
      const errorMsg = err.message || 'Error sincronizando tareas';
      setError(errorMsg);
      console.error('Error sincronizando tareas:', err);
      toast.error('Error de sincronización', {
        description: errorMsg
      });
    } finally {
      setIsSyncing(false);
    }
  }, [enabled, isSyncing, tasks, setTasks, updateTask, addTask]);

  const createTaskInBackend = useCallback(async (
    repository: string,
    instruction: string
  ): Promise<Task | null> => {
    try {
      const [owner, repo] = repository.split('/');
      const client = getAPIClient();
      
      const backendTask = await client.createTask({
        repository_owner: owner,
        repository_name: repo,
        instruction
      });

      // Convertir a formato frontend
      const task: Task = {
        id: backendTask.id,
        repository: `${backendTask.repository_owner}/${backendTask.repository_name}`,
        instruction: backendTask.instruction,
        status: mapBackendStatus(backendTask.status),
        createdAt: backendTask.created_at,
        processingStartedAt: backendTask.started_at,
        result: backendTask.result ? {
          content: typeof backendTask.result === 'string' 
            ? backendTask.result 
            : JSON.stringify(backendTask.result),
          plan: backendTask.result.plan,
          code: backendTask.result.code
        } : undefined,
        error: backendTask.error
      };

      // Agregar al store local
      addTask(task);
      
      return task;
    } catch (err: any) {
      console.error('Error creando tarea en backend:', err);
      toast.error('Error creando tarea', {
        description: err.message || 'No se pudo crear la tarea'
      });
      return null;
    }
  }, [addTask]);

  // Sincronización automática
  useEffect(() => {
    if (!enabled || !autoSync) return;

    // Sincronizar inmediatamente
    syncTasks();

    // Configurar intervalo
    const interval = setInterval(syncTasks, syncInterval);
    return () => clearInterval(interval);
  }, [enabled, autoSync, syncInterval, syncTasks]);

  return {
    syncTasks,
    createTaskInBackend,
    isSyncing,
    lastSync,
    error
  };
}

/**
 * Mapear status del backend al formato del frontend.
 */
function mapBackendStatus(status: string): Task['status'] {
  const statusMap: Record<string, Task['status']> = {
    'pending': 'pending',
    'running': 'processing',
    'completed': 'completed',
    'failed': 'failed',
    'cancelled': 'stopped'
  };
  return statusMap[status] || 'pending';
}

/**
 * Verificar si hay cambios entre tarea local y del backend.
 */
function hasChanges(local: Task, backend: Task): boolean {
  return (
    local.status !== backend.status ||
    local.error !== backend.error ||
    JSON.stringify(local.result) !== JSON.stringify(backend.result)
  );
}



