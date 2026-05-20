import { Task, TASK_STORAGE_KEY } from '../types/task';

/**
 * Normalizar el plan de una tarea para asegurar que files_to_create y files_to_modify sean arrays
 */
function normalizeTaskPlan(plan: any): any {
  if (!plan) return plan;
  
  const normalized = { ...plan };
  
  // Normalizar steps
  if (normalized.steps) {
    if (Array.isArray(normalized.steps)) {
      normalized.steps = normalized.steps.map((step: any) => 
        typeof step === 'string' ? step : JSON.stringify(step)
      );
    } else if (typeof normalized.steps === 'object') {
      normalized.steps = Object.values(normalized.steps).map((step: any) => 
        typeof step === 'string' ? step : JSON.stringify(step)
      );
    }
  }
  
  // Normalizar files_to_create
  if (normalized.files_to_create) {
    if (Array.isArray(normalized.files_to_create)) {
      normalized.files_to_create = normalized.files_to_create.map((file: any) => 
        typeof file === 'string' ? file : JSON.stringify(file)
      );
    } else if (typeof normalized.files_to_create === 'object' && normalized.files_to_create !== null) {
      // Si es un objeto, convertir las keys a array
      normalized.files_to_create = Object.keys(normalized.files_to_create);
    } else {
      normalized.files_to_create = [];
    }
  } else {
    normalized.files_to_create = [];
  }
  
  // Normalizar files_to_modify
  if (normalized.files_to_modify) {
    if (Array.isArray(normalized.files_to_modify)) {
      normalized.files_to_modify = normalized.files_to_modify.map((file: any) => 
        typeof file === 'string' ? file : JSON.stringify(file)
      );
    } else if (typeof normalized.files_to_modify === 'object' && normalized.files_to_modify !== null) {
      // Si es un objeto, convertir las keys a array
      normalized.files_to_modify = Object.keys(normalized.files_to_modify);
    } else {
      normalized.files_to_modify = [];
    }
  } else {
    normalized.files_to_modify = [];
  }
  
  return normalized;
}

/**
 * Normalizar una tarea completa
 */
function normalizeTask(task: any): Task {
  const normalized = { ...task };
  
  // Normalizar result.content
  if (normalized.result?.content && typeof normalized.result.content !== 'string') {
    normalized.result.content = JSON.stringify(normalized.result.content, null, 2);
  }
  
  // Normalizar result.code
  if (normalized.result?.code && typeof normalized.result.code !== 'string') {
    normalized.result.code = JSON.stringify(normalized.result.code, null, 2);
  }
  
  // Normalizar result.plan
  if (normalized.result?.plan) {
    normalized.result.plan = normalizeTaskPlan(normalized.result.plan);
  }
  
  // Normalizar streamingContent
  if (normalized.streamingContent && typeof normalized.streamingContent !== 'string') {
    normalized.streamingContent = JSON.stringify(normalized.streamingContent, null, 2);
  }
  
  return normalized as Task;
}

/**
 * Cargar tareas desde localStorage
 */
export function loadTasksFromStorage(): Task[] {
  if (typeof window === 'undefined') return [];
  
  try {
    const savedTasks = localStorage.getItem(TASK_STORAGE_KEY);
    if (savedTasks) {
      const tasks = JSON.parse(savedTasks) as Task[];
      // Normalizar todas las tareas al cargarlas
      return tasks.map(normalizeTask);
    }
  } catch (error) {
    console.error('Error loading tasks from storage:', error);
  }
  
  return [];
}

/**
 * Guardar tareas en localStorage
 */
export function saveTasksToStorage(tasks: Task[]): void {
  if (typeof window === 'undefined') return;
  
  try {
    if (tasks.length > 0) {
      localStorage.setItem(TASK_STORAGE_KEY, JSON.stringify(tasks));
    } else {
      localStorage.removeItem(TASK_STORAGE_KEY);
    }
  } catch (error) {
    console.error('Error saving tasks to storage:', error);
  }
}

/**
 * Generar un ID único para una nueva tarea
 */
export function generateTaskId(): string {
  return `task-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

