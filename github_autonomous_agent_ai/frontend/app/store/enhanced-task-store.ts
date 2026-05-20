/**
 * Enhanced Task Store - Store mejorado con sincronización con backend, undo/redo, y más funcionalidades.
 */

import { create } from 'zustand';
import { Task } from '../types/task';
import { loadTasksFromStorage, saveTasksToStorage, generateTaskId } from '../utils/task-storage';
import { getAPIClient } from '../lib/api-client';

interface TaskStore {
  tasks: Task[];
  isLoading: boolean;
  
  // Actions
  setTasks: (tasks: Task[]) => void;
  addTask: (task: Omit<Task, 'id' | 'createdAt'>) => Task;
  updateTask: (taskId: string, updates: Partial<Task>) => void;
  deleteTask: (taskId: string) => void;
  deleteTasks: (taskIds: string[]) => void;
  deleteAllTasks: () => void;
  loadTasks: () => void;
  
  // Computed
  getTaskById: (taskId: string) => Task | undefined;
  getTasksByStatus: (status: Task['status']) => Task[];
}

interface EnhancedTaskStore extends TaskStore {
  // Sincronización
  isSyncing: boolean;
  lastSync: Date | null;
  syncError: string | null;
  
  // Backend sync
  syncWithBackend: () => Promise<void>;
  createTaskInBackend: (repository: string, instruction: string) => Promise<Task | null>;
  
  // Undo/Redo
  history: Task[][];
  historyIndex: number;
  canUndo: boolean;
  canRedo: boolean;
  undo: () => void;
  redo: () => void;
  clearHistory: () => void;
  
  // Filtros y búsqueda
  searchQuery: string;
  filterStatus: Task['status'] | 'all';
  filterRepository: string | 'all';
  sortBy: 'createdAt' | 'status' | 'repository' | 'updatedAt';
  sortOrder: 'asc' | 'desc';
  setSearchQuery: (query: string) => void;
  setFilterStatus: (status: Task['status'] | 'all') => void;
  setFilterRepository: (repository: string | 'all') => void;
  setSortBy: (field: 'createdAt' | 'status' | 'repository' | 'updatedAt') => void;
  setSortOrder: (order: 'asc' | 'desc') => void;
  getFilteredTasks: () => Task[];
  
  // Estadísticas
  getStats: () => {
    total: number;
    byStatus: Record<Task['status'], number>;
    byRepository: Record<string, number>;
    completedToday: number;
    failedToday: number;
    averageProcessingTime: number;
  };
  
  // Bulk operations
  bulkUpdateStatus: (taskIds: string[], status: Task['status']) => void;
  bulkDelete: (taskIds: string[]) => void;
  duplicateTask: (taskId: string) => Task | null;
  
  // Persistencia mejorada
  exportTasks: () => string;
  importTasks: (json: string) => boolean;
}

export const useEnhancedTaskStore = create<EnhancedTaskStore>((set, get) => ({
  // Estado base
  tasks: [],
  isLoading: true,
  isSyncing: false,
  lastSync: null,
  syncError: null,
  history: [],
  historyIndex: -1,
  searchQuery: '',
  filterStatus: 'all',
  filterRepository: 'all',
  sortBy: 'createdAt',
  sortOrder: 'desc',

  // Computed
  get canUndo() {
    return get().historyIndex > 0;
  },
  
  get canRedo() {
    const state = get();
    return state.historyIndex < state.history.length - 1;
  },

  // Actions base
  setTasks: (tasks) => {
    // Guardar en historial antes de cambiar
    const state = get();
    const newHistory = state.history.slice(0, state.historyIndex + 1);
    newHistory.push([...state.tasks]);
    set({ 
      tasks, 
      history: newHistory,
      historyIndex: newHistory.length - 1
    });
    saveTasksToStorage(tasks);
  },

  addTask: (task) => {
    const newTask: Task = {
      ...task,
      id: generateTaskId(),
      createdAt: new Date().toISOString(),
    };
    set((state) => {
      const newTasks = [...state.tasks, newTask];
      // Guardar en historial
      const newHistory = state.history.slice(0, state.historyIndex + 1);
      newHistory.push([...state.tasks]);
      saveTasksToStorage(newTasks);
      return { 
        tasks: newTasks,
        history: newHistory,
        historyIndex: newHistory.length - 1
      };
    });
    return newTask;
  },

  updateTask: (taskId, updates) => {
    set((state) => {
      const newTasks = state.tasks.map((task) =>
        task.id === taskId ? { ...task, ...updates } : task
      );
      // Guardar en historial
      const newHistory = state.history.slice(0, state.historyIndex + 1);
      newHistory.push([...state.tasks]);
      saveTasksToStorage(newTasks);
      return { 
        tasks: newTasks,
        history: newHistory,
        historyIndex: newHistory.length - 1
      };
    });
  },

  deleteTask: (taskId) => {
    set((state) => {
      const newTasks = state.tasks.filter((task) => task.id !== taskId);
      // Guardar en historial
      const newHistory = state.history.slice(0, state.historyIndex + 1);
      newHistory.push([...state.tasks]);
      saveTasksToStorage(newTasks);
      return { 
        tasks: newTasks,
        history: newHistory,
        historyIndex: newHistory.length - 1
      };
    });
  },

  deleteTasks: (taskIds) => {
    set((state) => {
      const newTasks = state.tasks.filter((task) => !taskIds.includes(task.id));
      // Guardar en historial
      const newHistory = state.history.slice(0, state.historyIndex + 1);
      newHistory.push([...state.tasks]);
      saveTasksToStorage(newTasks);
      return { 
        tasks: newTasks,
        history: newHistory,
        historyIndex: newHistory.length - 1
      };
    });
  },

  deleteAllTasks: () => {
    set((state) => {
      const newHistory = state.history.slice(0, state.historyIndex + 1);
      newHistory.push([...state.tasks]);
      saveTasksToStorage([]);
      return { 
        tasks: [],
        history: newHistory,
        historyIndex: newHistory.length - 1
      };
    });
  },

  loadTasks: () => {
    set({ isLoading: true });
    try {
      const loadedTasks = loadTasksFromStorage();
      set({ 
        tasks: loadedTasks, 
        isLoading: false,
        history: [[...loadedTasks]],
        historyIndex: 0
      });
    } catch (error) {
      console.error('Error loading tasks:', error);
      set({ tasks: [], isLoading: false });
    }
  },

  getTaskById: (taskId) => {
    return get().tasks.find((task) => task.id === taskId);
  },

  getTasksByStatus: (status) => {
    return get().tasks.filter((task) => task.status === status);
  },

  // Undo/Redo
  undo: () => {
    const state = get();
    if (state.historyIndex > 0) {
      const previousState = state.history[state.historyIndex - 1];
      set({ 
        tasks: [...previousState],
        historyIndex: state.historyIndex - 1
      });
      saveTasksToStorage(previousState);
    }
  },

  redo: () => {
    const state = get();
    if (state.historyIndex < state.history.length - 1) {
      const nextState = state.history[state.historyIndex + 1];
      set({ 
        tasks: [...nextState],
        historyIndex: state.historyIndex + 1
      });
      saveTasksToStorage(nextState);
    }
  },

  clearHistory: () => {
    const state = get();
    set({ 
      history: [[...state.tasks]],
      historyIndex: 0
    });
  },

  // Sincronización con backend
  syncWithBackend: async () => {
    const state = get();
    if (state.isSyncing) return;

    set({ isSyncing: true, syncError: null });

    try {
      const client = getAPIClient();
      const backendTasks = await client.listTasks();

      // Convertir y sincronizar
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
        error: bt.error
      }));

      // Actualizar tareas existentes y agregar nuevas
      const existingIds = new Set(state.tasks.map(t => t.id));
      const newTasks = convertedTasks.filter(t => !existingIds.has(t.id));
      
      convertedTasks.forEach(backendTask => {
        const localTask = state.tasks.find(t => t.id === backendTask.id);
        if (localTask && hasChanges(localTask, backendTask)) {
          get().updateTask(backendTask.id, backendTask);
        }
      });

      newTasks.forEach(task => {
        get().addTask(task);
      });

      set({ lastSync: new Date(), isSyncing: false });
    } catch (error: any) {
      set({ 
        syncError: error.message || 'Error sincronizando',
        isSyncing: false 
      });
      throw error;
    }
  },

  createTaskInBackend: async (repository, instruction) => {
    try {
      const [owner, repo] = repository.split('/');
      const client = getAPIClient();
      
      const backendTask = await client.createTask({
        repository_owner: owner,
        repository_name: repo,
        instruction
      });

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

      get().addTask(task);
      return task;
    } catch (error: any) {
      console.error('Error creando tarea en backend:', error);
      throw error;
    }
  },

  // Filtros
  setSearchQuery: (query) => {
    set({ searchQuery: query });
  },

  setFilterStatus: (status) => {
    set({ filterStatus: status });
  },

  setFilterRepository: (repository) => {
    set({ filterRepository: repository });
  },

  setSortBy: (field) => {
    set({ sortBy: field });
  },

  setSortOrder: (order) => {
    set({ sortOrder: order });
  },

  getFilteredTasks: () => {
    const state = get();
    let filtered = [...state.tasks];

    // Filtrar por status
    if (state.filterStatus !== 'all') {
      filtered = filtered.filter(t => t.status === state.filterStatus);
    }

    // Filtrar por repositorio
    if (state.filterRepository !== 'all') {
      filtered = filtered.filter(t => t.repository === state.filterRepository);
    }

    // Filtrar por búsqueda
    if (state.searchQuery) {
      const query = state.searchQuery.toLowerCase();
      filtered = filtered.filter(t =>
        t.instruction.toLowerCase().includes(query) ||
        t.repository.toLowerCase().includes(query) ||
        t.id.toLowerCase().includes(query) ||
        (t.error && t.error.toLowerCase().includes(query))
      );
    }

    // Ordenar
    filtered.sort((a, b) => {
      let aVal: any, bVal: any;
      
      switch (state.sortBy) {
        case 'createdAt':
          aVal = new Date(a.createdAt).getTime();
          bVal = new Date(b.createdAt).getTime();
          break;
        case 'status':
          aVal = a.status;
          bVal = b.status;
          break;
        case 'repository':
          aVal = a.repository;
          bVal = b.repository;
          break;
        case 'updatedAt':
          aVal = a.processingStartedAt ? new Date(a.processingStartedAt).getTime() : 0;
          bVal = b.processingStartedAt ? new Date(b.processingStartedAt).getTime() : 0;
          break;
        default:
          return 0;
      }
      
      if (aVal < bVal) return state.sortOrder === 'asc' ? -1 : 1;
      if (aVal > bVal) return state.sortOrder === 'asc' ? 1 : -1;
      return 0;
    });

    return filtered;
  },

  // Estadísticas
  getStats: () => {
    const tasks = get().tasks;
    const today = new Date().toDateString();
    
    const byStatus = tasks.reduce((acc, task) => {
      acc[task.status] = (acc[task.status] || 0) + 1;
      return acc;
    }, {} as Record<Task['status'], number>);

    const byRepository = tasks.reduce((acc, task) => {
      acc[task.repository] = (acc[task.repository] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    const completedToday = tasks.filter(t => 
      t.status === 'completed' && 
      t.createdAt && 
      new Date(t.createdAt).toDateString() === today
    ).length;

    const failedToday = tasks.filter(t => 
      t.status === 'failed' && 
      t.createdAt && 
      new Date(t.createdAt).toDateString() === today
    ).length;

    // Calcular tiempo promedio de procesamiento
    const completedTasks = tasks.filter(t => 
      t.status === 'completed' && 
      t.processingStartedAt && 
      t.createdAt
    );
    
    const averageProcessingTime = completedTasks.length > 0
      ? completedTasks.reduce((sum, task) => {
          const start = new Date(task.processingStartedAt!).getTime();
          const created = new Date(task.createdAt).getTime();
          return sum + (start - created);
        }, 0) / completedTasks.length / 1000 // Convertir a segundos
      : 0;

    return {
      total: tasks.length,
      byStatus: {
        pending: byStatus.pending || 0,
        processing: byStatus.processing || 0,
        running: byStatus.running || 0,
        completed: byStatus.completed || 0,
        failed: byStatus.failed || 0,
        stopped: byStatus.stopped || 0
      },
      byRepository,
      completedToday,
      failedToday,
      averageProcessingTime
    };
  },

  // Bulk operations
  bulkUpdateStatus: (taskIds, status) => {
    set((state) => {
      const newTasks = state.tasks.map(task =>
        taskIds.includes(task.id) ? { ...task, status } : task
      );
      const newHistory = state.history.slice(0, state.historyIndex + 1);
      newHistory.push([...state.tasks]);
      saveTasksToStorage(newTasks);
      return {
        tasks: newTasks,
        history: newHistory,
        historyIndex: newHistory.length - 1
      };
    });
  },

  bulkDelete: (taskIds) => {
    get().deleteTasks(taskIds);
  },

  duplicateTask: (taskId) => {
    const task = get().getTaskById(taskId);
    if (!task) return null;
    
    const duplicated: Omit<Task, 'id' | 'createdAt'> = {
      repository: task.repository,
      instruction: task.instruction,
      status: 'pending',
      repoInfo: task.repoInfo
    };
    
    return get().addTask(duplicated);
  },

  // Export/Import
  exportTasks: () => {
    const tasks = get().tasks;
    return JSON.stringify(tasks, null, 2);
  },

  importTasks: (json) => {
    try {
      const tasks = JSON.parse(json) as Task[];
      if (Array.isArray(tasks)) {
        get().setTasks(tasks);
        return true;
      }
      return false;
    } catch (error) {
      console.error('Error importing tasks:', error);
      return false;
    }
  }
}));

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

function hasChanges(local: Task, backend: Task): boolean {
  return (
    local.status !== backend.status ||
    local.error !== backend.error ||
    JSON.stringify(local.result) !== JSON.stringify(backend.result)
  );
}
