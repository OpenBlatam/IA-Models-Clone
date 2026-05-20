import { create } from 'zustand';
import { Task } from '../types/task';
import { tasksAPI } from '../lib/tasks-api';

interface TaskStore {
  tasks: Task[];
  isLoading: boolean;
  isSyncing: boolean;
  lastSync: Date | null;
  
  // Actions
  setTasks: (tasks: Task[]) => void;
  addTask: (task: Omit<Task, 'id' | 'createdAt'>) => Promise<Task>;
  updateTask: (taskId: string, updates: Partial<Task>) => Promise<void>;
  deleteTask: (taskId: string) => Promise<void>;
  deleteTasks: (taskIds: string[]) => Promise<void>;
  deleteAllTasks: () => Promise<void>;
  loadTasks: (force?: boolean) => Promise<void>;
  syncTasks: () => Promise<void>;
  
  // Computed
  getTaskById: (taskId: string) => Task | undefined;
  getTasksByStatus: (status: Task['status']) => Task[];
}

export const useTaskStore = create<TaskStore>((set, get) => ({
  tasks: [],
  isLoading: true,
  isSyncing: false,
  lastSync: null,

  setTasks: (tasks) => {
    set({ tasks });
  },

  addTask: async (task) => {
    console.log('📦 addTask llamado con:', task);
    
    try {
      const newTask = await tasksAPI.createTask(task);
      console.log('📦 Nueva tarea creada en backend:', newTask);
      
      // Agregar la tarea al estado local inmediatamente
      set((state) => {
        // Verificar que no esté duplicada
        const exists = state.tasks.some(t => t.id === newTask.id);
        if (exists) {
          console.log('⚠️ Tarea ya existe en el estado, actualizando...');
          const updatedTasks = state.tasks.map(t => t.id === newTask.id ? newTask : t);
          return { tasks: updatedTasks };
        }
        const newTasks = [...state.tasks, newTask];
        console.log(`✅ Tarea agregada al estado local. Total: ${newTasks.length}`);
        return { tasks: newTasks };
      });
      
      console.log('✅ Tarea agregada exitosamente');
      return newTask;
    } catch (error) {
      console.error('Error creating task:', error);
      throw error;
    }
  },

  updateTask: async (taskId, updates) => {
    try {
      // Proteger tareas en procesamiento: no permitir cambiar de 'processing' a 'stopped' 
      // a menos que sea explícitamente solicitado (el usuario hizo clic en pausa)
      const currentState = get();
      const currentTask = currentState.tasks.find(t => t.id === taskId);
      
      if (currentTask) {
        const isLocalProcessing = currentTask.status === 'processing' || currentTask.status === 'running';
        const isUpdatingToStopped = updates.status === 'stopped';
        
        // Si la tarea está procesando y se intenta cambiar a 'stopped',
        // solo permitirlo si viene de una acción explícita del usuario
        // (verificamos que no sea una actualización automática del backend)
        if (isLocalProcessing && isUpdatingToStopped) {
          // Verificar si es una actualización explícita (tiene error o viene de stopTask)
          const isExplicitStop = updates.error?.includes('detenido por el usuario') || 
                                 updates.error?.includes('Procesamiento detenido');
          
          if (!isExplicitStop) {
            console.log(`🛡️ Protegiendo tarea ${taskId} en procesamiento - ignorando cambio a 'stopped' que no es explícito`);
            // No aplicar el cambio de estado, pero sí otros campos
            const { status, ...otherUpdates } = updates;
            if (Object.keys(otherUpdates).length > 0) {
              await tasksAPI.updateTask(taskId, otherUpdates);
              set((state) => {
                const newTasks = state.tasks.map((task) =>
                  task.id === taskId ? { ...task, ...otherUpdates } : task
                );
                return { tasks: newTasks };
              });
            }
            return;
          }
        }
      }
      
      await tasksAPI.updateTask(taskId, updates);
      
      set((state) => {
        const newTasks = state.tasks.map((task) =>
          task.id === taskId ? { ...task, ...updates } : task
        );
        return { tasks: newTasks };
      });
    } catch (error) {
      console.error('Error updating task:', error);
      // Actualizar localmente aunque falle el backend
      set((state) => {
        const newTasks = state.tasks.map((task) =>
          task.id === taskId ? { ...task, ...updates } : task
        );
        return { tasks: newTasks };
      });
    }
  },

  deleteTask: async (taskId) => {
    try {
      await tasksAPI.deleteTask(taskId);
      set((state) => {
        const newTasks = state.tasks.filter((task) => task.id !== taskId);
        return { tasks: newTasks };
      });
    } catch (error) {
      console.error('Error deleting task:', error);
      throw error;
    }
  },

  deleteTasks: async (taskIds) => {
    try {
      await Promise.all(taskIds.map(id => tasksAPI.deleteTask(id)));
      set((state) => {
        const newTasks = state.tasks.filter((task) => !taskIds.includes(task.id));
        return { tasks: newTasks };
      });
    } catch (error) {
      console.error('Error deleting tasks:', error);
      throw error;
    }
  },

  deleteAllTasks: async () => {
    try {
      const tasks = get().tasks;
      await Promise.all(tasks.map(task => tasksAPI.deleteTask(task.id)));
      set({ tasks: [] });
    } catch (error) {
      console.error('Error deleting all tasks:', error);
      throw error;
    }
  },

  loadTasks: async (force: boolean = false) => {
    // Solo cambiar isLoading si no está cargando ya (evitar reinicios)
    const currentState = get();
    if (currentState.isLoading && !force) {
      return;
    }
    
    set({ isLoading: true });
    try {
      const loadedTasks = await tasksAPI.getAllTasks();
      console.log(`📥 Tareas cargadas del backend: ${loadedTasks.length}`);
      
      // Fusionar con tareas locales para evitar perder tareas recién creadas
      set((state) => {
        const existingTaskIds = new Set(state.tasks.map(t => t.id));
        const newTasks = loadedTasks.filter(t => !existingTaskIds.has(t.id));
        const mergedTasks = [...state.tasks, ...newTasks];
        
        // Actualizar tareas existentes con datos del backend
        // IMPORTANTE: Preservar eventos locales y estados importantes
        // No sobrescribir tareas en 'processing' o 'running' con estado 'stopped' del backend
        // a menos que el usuario haya hecho clic en pausa explícitamente
        // También preservar planes aprobados y estados importantes
        const updatedTasks = mergedTasks.map(localTask => {
          const backendTask = loadedTasks.find(bt => bt.id === localTask.id);
          if (!backendTask) {
            return localTask;
          }
          
          // Proteger tareas en procesamiento activo
          const isLocalProcessing = localTask.status === 'processing' || localTask.status === 'running';
          const isBackendStopped = backendTask.status === 'stopped';
          
          // Si la tarea local está procesando y el backend dice que está detenida,
          // mantener el estado local de procesamiento (no sobrescribir con 'stopped')
          if (isLocalProcessing && isBackendStopped) {
            console.log(`🛡️ Protegiendo tarea ${localTask.id} en procesamiento - no sobrescribiendo con estado 'stopped' del backend`);
            // Mantener el estado local pero actualizar otros campos del backend
            return {
              ...backendTask,
              status: localTask.status, // Preservar estado de procesamiento
              processingStartedAt: localTask.processingStartedAt || backendTask.processingStartedAt,
              // Preservar plan aprobado si existe localmente
              pendingApproval: localTask.pendingApproval || backendTask.pendingApproval,
              // Preservar streamingContent local si es más reciente o tiene más contenido
              streamingContent: (localTask.streamingContent && 
                (typeof localTask.streamingContent === 'string' ? localTask.streamingContent.length : 0) > 
                (typeof backendTask.streamingContent === 'string' ? backendTask.streamingContent.length : 0))
                ? localTask.streamingContent : backendTask.streamingContent,
            };
          }
          
          // Preservar plan aprobado si existe en el backend pero no en local (o viceversa)
          const preservedPendingApproval = localTask.pendingApproval || backendTask.pendingApproval;
          
          // Preservar streamingContent: usar el que tenga más contenido o el más reciente
          const localContentLength = typeof localTask.streamingContent === 'string' 
            ? localTask.streamingContent.length 
            : (localTask.streamingContent ? JSON.stringify(localTask.streamingContent).length : 0);
          const backendContentLength = typeof backendTask.streamingContent === 'string' 
            ? backendTask.streamingContent.length 
            : (backendTask.streamingContent ? JSON.stringify(backendTask.streamingContent).length : 0);
          
          const preservedStreamingContent = localContentLength > backendContentLength 
            ? localTask.streamingContent 
            : backendTask.streamingContent;
          
          // Para otros casos, usar la tarea del backend pero preservar información importante
          return {
            ...backendTask,
            // Preservar plan aprobado si existe (priorizar local si existe, sino backend)
            pendingApproval: preservedPendingApproval,
            // Preservar streamingContent con más contenido
            streamingContent: preservedStreamingContent,
            // Preservar processingStartedAt si existe localmente
            processingStartedAt: localTask.processingStartedAt || backendTask.processingStartedAt,
            // Preservar pendingCommitApproval si existe localmente
            pendingCommitApproval: localTask.pendingCommitApproval || backendTask.pendingCommitApproval,
          };
        });
        
        // Solo actualizar si hay cambios reales para evitar re-renders innecesarios
        const hasChanges = updatedTasks.length !== state.tasks.length ||
          updatedTasks.some((task, index) => {
            const existing = state.tasks[index];
            return !existing || JSON.stringify(task) !== JSON.stringify(existing);
          });
        
        if (hasChanges) {
          console.log(`✅ Tareas fusionadas: ${updatedTasks.length} (${state.tasks.length} locales + ${newTasks.length} nuevas del backend)`);
          return { tasks: updatedTasks, isLoading: false, lastSync: new Date() };
        } else {
          console.log(`ℹ️ No hay cambios, manteniendo estado actual`);
          return { isLoading: false };
        }
      });
    } catch (error: any) {
      console.error('Error loading tasks:', error);
      // Intentar cargar desde localStorage como último recurso
      if (typeof window !== 'undefined') {
        try {
          const { loadTasksFromStorage } = await import('../utils/task-storage');
          const localTasks = loadTasksFromStorage();
          set((state) => {
            // Fusionar con estado actual
            const existingTaskIds = new Set(state.tasks.map(t => t.id));
            const newTasks = localTasks.filter(t => !existingTaskIds.has(t.id));
            const mergedTasks = [...state.tasks, ...newTasks];
            return { tasks: mergedTasks, isLoading: false };
          });
          console.log(`📦 Tareas cargadas desde localStorage: ${localTasks.length}`);
        } catch (storageError) {
          console.error('Error loading from localStorage:', storageError);
          set({ isLoading: false });
        }
      } else {
        set({ isLoading: false });
      }
    }
  },

  syncTasks: async () => {
    if (get().isSyncing) return;
    
    set({ isSyncing: true });
    try {
      const currentState = get();
      const loadedTasks = await tasksAPI.getAllTasks();
      
      // Proteger tareas en procesamiento activo durante la sincronización
      const protectedTasks = loadedTasks.map(backendTask => {
        const localTask = currentState.tasks.find(lt => lt.id === backendTask.id);
        if (!localTask) {
          return backendTask;
        }
        
        // Proteger tareas en procesamiento activo
        const isLocalProcessing = localTask.status === 'processing' || localTask.status === 'running';
        const isBackendStopped = backendTask.status === 'stopped';
        
        // Si la tarea local está procesando y el backend dice que está detenida,
        // mantener el estado local de procesamiento (no sobrescribir con 'stopped')
        if (isLocalProcessing && isBackendStopped) {
          console.log(`🛡️ Protegiendo tarea ${localTask.id} en procesamiento durante sync - no sobrescribiendo con estado 'stopped' del backend`);
          // Mantener el estado local pero actualizar otros campos del backend
          return {
            ...backendTask,
            status: localTask.status, // Preservar estado de procesamiento
            processingStartedAt: localTask.processingStartedAt || backendTask.processingStartedAt,
          };
        }
        
        // Para otros casos, usar la tarea del backend
        return backendTask;
      });
      
      set({ 
        tasks: protectedTasks, 
        isSyncing: false, 
        lastSync: new Date() 
      });
    } catch (error) {
      console.error('Error syncing tasks:', error);
      set({ isSyncing: false });
    }
  },

  getTaskById: (taskId) => {
    return get().tasks.find((task) => task.id === taskId);
  },

  getTasksByStatus: (status) => {
    return get().tasks.filter((task) => task.status === status);
  },
}));

