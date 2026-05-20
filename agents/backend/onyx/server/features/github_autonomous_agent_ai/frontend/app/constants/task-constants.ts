import { TaskStatus } from '../types/task';

export const TASK_STATUSES: TaskStatus[] = [
  'pending',
  'processing',
  'running',
  'completed',
  'failed',
  'stopped',
  'pending_approval',
  'pending_commit_approval',
];

export const TASK_STATUS_LABELS: Record<TaskStatus, string> = {
  pending: 'Pendiente',
  processing: '🔄 Procesando...',
  running: 'En Ejecución',
  completed: 'Completada',
  failed: 'Fallida',
  stopped: '⏹️ Detenida',
  pending_approval: '⏳ Pendiente Aprobación Plan',
  pending_commit_approval: '⏳ Pendiente Aprobación Commit',
};

export const TASK_STATUS_LABELS_PLURAL: Record<TaskStatus, string> = {
  pending: 'Pendientes',
  processing: 'Procesando',
  running: 'En Ejecución',
  completed: 'Completadas',
  failed: 'Fallidas',
  stopped: 'Detenidas',
  pending_approval: 'Pendientes Aprobación Plan',
  pending_commit_approval: 'Pendientes Aprobación Commit',
};

export const TASK_STATUS_COLORS: Record<TaskStatus, string> = {
  pending: 'bg-gray-100 text-gray-800',
  processing: 'bg-yellow-100 text-yellow-800 animate-pulse',
  running: 'bg-blue-100 text-blue-800',
  completed: 'bg-green-100 text-green-800',
  failed: 'bg-red-100 text-red-800',
  stopped: 'bg-orange-100 text-orange-800',
  pending_approval: 'bg-yellow-100 text-yellow-800',
  pending_commit_approval: 'bg-purple-100 text-purple-800',
};

export const TASK_STATUS_BORDER_COLORS: Record<TaskStatus, string> = {
  pending: 'border-gray-200',
  processing: 'border-yellow-200',
  running: 'border-blue-200',
  completed: 'border-green-200',
  failed: 'border-red-200',
  stopped: 'border-orange-200',
  pending_approval: 'border-yellow-300',
  pending_commit_approval: 'border-purple-300',
};

export const KANBAN_COLUMNS: Array<{ id: TaskStatus; label: string }> = [
  { id: 'pending', label: TASK_STATUS_LABELS_PLURAL.pending },
  { id: 'processing', label: TASK_STATUS_LABELS_PLURAL.processing },
  { id: 'running', label: TASK_STATUS_LABELS_PLURAL.running },
  { id: 'completed', label: TASK_STATUS_LABELS_PLURAL.completed },
  { id: 'failed', label: TASK_STATUS_LABELS_PLURAL.failed },
  { id: 'stopped', label: TASK_STATUS_LABELS_PLURAL.stopped },
];



