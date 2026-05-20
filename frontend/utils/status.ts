import { STATUS_BADGES, STATUS_ICONS, type TaskStatusType } from '@/constants/status';

export function getStatusBadge(status: string) {
  const badge = STATUS_BADGES[status as TaskStatusType] || STATUS_BADGES.queued;
  return badge;
}

export function getStatusIcon(status: string) {
  const icon = STATUS_ICONS[status as TaskStatusType] || STATUS_ICONS.queued;
  return icon;
}

export function getStatusColor(status: string): string {
  const colors: Record<string, string> = {
    completed: 'green',
    processing: 'yellow',
    failed: 'red',
    queued: 'blue',
  };
  return colors[status] || 'gray';
}

