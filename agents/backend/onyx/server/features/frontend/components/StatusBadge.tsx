'use client';

import { getStatusBadge } from '@/utils/status';

interface StatusBadgeProps {
  status: string;
  className?: string;
}

export default function StatusBadge({ status, className = '' }: StatusBadgeProps) {
  const badge = getStatusBadge(status);
  return (
    <span className={`px-2 py-1 rounded text-xs font-medium ${badge.className} ${className}`}>
      {badge.label}
    </span>
  );
}

