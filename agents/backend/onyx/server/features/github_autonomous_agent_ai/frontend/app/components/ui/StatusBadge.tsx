'use client';

import { TaskStatus } from '../../types/task';
import { TASK_STATUS_COLORS, TASK_STATUS_LABELS } from '../../constants/task-constants';
import { cn } from '../../utils/cn';

interface StatusBadgeProps {
  status: TaskStatus;
  className?: string;
}

export function StatusBadge({ status, className }: StatusBadgeProps) {
  return (
    <span className={cn(
      "px-3 py-1.5 rounded-md text-xs font-medium whitespace-nowrap",
      TASK_STATUS_COLORS[status],
      className
    )}>
      {TASK_STATUS_LABELS[status]}
    </span>
  );
}



