import { useState, useEffect, useCallback } from 'react';
import { getAPIClient, Task, AgentStatus, CreateTaskRequest } from '../lib/api-client';
import { toast } from 'sonner';

export function useTasks() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchTasks = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const apiClient = getAPIClient();
      const data = await apiClient.getTasks();
      setTasks(data);
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || err.message || 'Error al cargar tareas';
      setError(errorMessage);
      toast.error(errorMessage);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchTasks();
  }, [fetchTasks]);

  const createTask = useCallback(async (task: CreateTaskRequest) => {
    try {
      const apiClient = getAPIClient();
      const newTask = await apiClient.createTask(task);
      setTasks((prev) => [...prev, newTask]);
      toast.success('Tarea creada exitosamente');
      return newTask;
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || err.message || 'Error al crear tarea';
      toast.error(errorMessage);
      throw err;
    }
  }, []);

  const updateTask = useCallback(async (id: string, task: Partial<Task>) => {
    try {
      const apiClient = getAPIClient();
      const updatedTask = await apiClient.updateTask(id, task);
      setTasks((prev) => prev.map((t) => (t.id === id ? updatedTask : t)));
      toast.success('Tarea actualizada exitosamente');
      return updatedTask;
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || err.message || 'Error al actualizar tarea';
      toast.error(errorMessage);
      throw err;
    }
  }, []);

  const deleteTask = useCallback(async (id: string) => {
    try {
      const apiClient = getAPIClient();
      await apiClient.deleteTask(id);
      setTasks((prev) => prev.filter((t) => t.id !== id));
      toast.success('Tarea eliminada exitosamente');
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || err.message || 'Error al eliminar tarea';
      toast.error(errorMessage);
      throw err;
    }
  }, []);

  return {
    tasks,
    loading,
    error,
    fetchTasks,
    createTask,
    updateTask,
    deleteTask,
  };
}

export function useAgents() {
  const [agents, setAgents] = useState<AgentStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchAgents = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const apiClient = getAPIClient();
      const data = await apiClient.getAgents();
      setAgents(data);
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || err.message || 'Error al cargar agentes';
      setError(errorMessage);
      toast.error(errorMessage);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAgents();
    // Auto-refresh cada 5 segundos
    const interval = setInterval(fetchAgents, 5000);
    return () => clearInterval(interval);
  }, [fetchAgents]);

  return {
    agents,
    loading,
    error,
    fetchAgents,
  };
}


