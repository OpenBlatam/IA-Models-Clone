import React from 'react';
import { cn } from '../../utils/cn';

export type TaskStatus = 'pending' | 'in_progress' | 'completed' | 'failed';

interface StatusBadgeProps {
  status: TaskStatus;
  className?: string;
}

const TASK_STATUS_COLORS: Record<TaskStatus, string> = {
  pending: 'bg-yellow-100 text-yellow-800',
  in_progress: 'bg-blue-100 text-blue-800',
  completed: 'bg-green-100 text-green-800',
  failed: 'bg-red-100 text-red-800',
};

const TASK_STATUS_LABELS: Record<TaskStatus, string> = {
  pending: 'Pendiente',
  in_progress: 'En Progreso',
  completed: 'Completado',
  failed: 'Fallido',
};

export const StatusBadge: React.FC<StatusBadgeProps> = ({
  status,
  className,
}) => {
  return (
    <span
      className={cn(
        'px-3 py-1.5 rounded-md text-xs font-medium whitespace-nowrap',
        TASK_STATUS_COLORS[status],
        className
      )}
    >
      {TASK_STATUS_LABELS[status]}
    </span>
  );
};


