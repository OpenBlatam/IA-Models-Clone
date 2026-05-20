import { Task, TaskStatus } from '../types/task';
import { 
  TASK_STATUS_COLORS, 
  TASK_STATUS_LABELS, 
  TASK_STATUS_LABELS_PLURAL 
} from '../constants/task-constants';

/**
 * Obtener tareas filtradas por estado
 */
export function getTasksByStatus(tasks: Task[], status: TaskStatus): Task[] {
  return tasks.filter(task => task.status === status);
}

/**
 * Obtener color de badge según el estado
 * @deprecated Usar TASK_STATUS_COLORS directamente o el componente StatusBadge
 */
export function getStatusBadgeColor(status: TaskStatus): string {
  return TASK_STATUS_COLORS[status] || 'bg-gray-100 text-gray-800';
}

/**
 * Obtener etiqueta en español según el estado
 * @deprecated Usar TASK_STATUS_LABELS directamente
 */
export function getStatusLabel(status: TaskStatus): string {
  return TASK_STATUS_LABELS[status] || status;
}

/**
 * Obtener etiqueta plural para columnas de Kanban
 * @deprecated Usar TASK_STATUS_LABELS_PLURAL directamente
 */
export function getStatusLabelPlural(status: TaskStatus): string {
  return TASK_STATUS_LABELS_PLURAL[status] || status;
}

/**
 * Obtener color de borde según el estado
 */
export function getStatusColor(status: TaskStatus): string {
  const colors: Record<TaskStatus, string> = {
    pending: 'border-gray-200',
    processing: 'border-yellow-200',
    running: 'border-blue-200',
    completed: 'border-green-200',
    failed: 'border-red-200',
    stopped: 'border-orange-200',
    pending_approval: 'border-yellow-300',
  };
  return colors[status] || 'border-gray-200';
}

/**
 * Calcular tiempo transcurrido desde que comenzó el procesamiento
 */
export function calculateElapsedTime(task: Task): string {
  if (!task.processingStartedAt || task.status !== 'processing') {
    return '';
  }
  
  const startTime = new Date(task.processingStartedAt).getTime();
  const now = Date.now();
  const diff = now - startTime;

  const hours = Math.floor(diff / (1000 * 60 * 60));
  const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
  const seconds = Math.floor((diff % (1000 * 60)) / 1000);

  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
}

/**
 * Verificar si una tarea puede ser reanudada
 * Permite reanudar tareas pendientes o detenidas
 */
export function canResumeTask(task: Task): boolean {
  return (
    (task.status === 'pending' || task.status === 'stopped') &&
    !!task.repoInfo
  );
}

/**
 * Verificar si una tarea está en un estado activo (solo durante generación del plan)
 * IMPORTANTE: El botón de parar solo debe aparecer durante 'processing' (generación del plan)
 * NO durante 'pending_approval' o 'pending_commit_approval'
 */
export function isTaskActive(task: Task): boolean {
  // Solo mostrar botón de parar durante la generación del plan
  return task.status === 'processing' || task.status === 'running';
}

