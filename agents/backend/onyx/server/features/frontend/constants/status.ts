export const TASK_STATUS = {
  QUEUED: 'queued',
  PROCESSING: 'processing',
  COMPLETED: 'completed',
  FAILED: 'failed',
} as const;

export type TaskStatusType = typeof TASK_STATUS[keyof typeof TASK_STATUS];

export const STATUS_BADGES: Record<TaskStatusType, { label: string; className: string }> = {
  [TASK_STATUS.COMPLETED]: {
    label: 'Completada',
    className: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
  },
  [TASK_STATUS.PROCESSING]: {
    label: 'Procesando',
    className: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
  },
  [TASK_STATUS.FAILED]: {
    label: 'Fallida',
    className: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
  },
  [TASK_STATUS.QUEUED]: {
    label: 'En Cola',
    className: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
  },
};

export const STATUS_ICONS: Record<TaskStatusType, string> = {
  [TASK_STATUS.COMPLETED]: '✓',
  [TASK_STATUS.PROCESSING]: '⟳',
  [TASK_STATUS.FAILED]: '✗',
  [TASK_STATUS.QUEUED]: '○',
};

