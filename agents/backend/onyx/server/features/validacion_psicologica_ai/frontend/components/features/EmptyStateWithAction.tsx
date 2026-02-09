/**
 * Empty state component with action button
 */

'use client';

import React from 'react';
import { Button } from '@/components/ui';
import { EmptyState } from '@/components/ui';
import { Plus } from 'lucide-react';

export interface EmptyStateWithActionProps {
  title: string;
  description: string;
  actionLabel?: string;
  onAction?: () => void;
  icon?: React.ReactNode;
}

export const EmptyStateWithAction: React.FC<EmptyStateWithActionProps> = ({
  title,
  description,
  actionLabel = 'Crear nuevo',
  onAction,
  icon,
}) => {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-center">
      {icon && <div className="mb-4 text-muted-foreground">{icon}</div>}
      <h3 className="text-lg font-semibold mb-2">{title}</h3>
      <p className="text-muted-foreground mb-6 max-w-md">{description}</p>
      {onAction && (
        <Button onClick={onAction} onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            onAction();
          }
        }} aria-label={actionLabel} tabIndex={0}>
          <Plus className="h-4 w-4 mr-2" aria-hidden="true" />
          {actionLabel}
        </Button>
      )}
    </div>
  );
};



