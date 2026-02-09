/**
 * Toggle between list and table view
 */

'use client';

import React from 'react';
import { Button } from '@/components/ui';
import { List, Table } from 'lucide-react';
import { cn } from '@/lib/utils/cn';

export type ViewMode = 'list' | 'table';

export interface ViewToggleProps {
  viewMode: ViewMode;
  onViewChange: (mode: ViewMode) => void;
  className?: string;
}

export const ViewToggle: React.FC<ViewToggleProps> = ({
  viewMode,
  onViewChange,
  className,
}) => {
  const handleListClick = () => {
    onViewChange('list');
  };

  const handleTableClick = () => {
    onViewChange('table');
  };

  const handleKeyDown = (event: React.KeyboardEvent, mode: ViewMode) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      onViewChange(mode);
    }
  };

  return (
    <div className={cn('flex items-center gap-2 border rounded-lg p-1', className)} role="tablist" aria-label="Cambiar vista">
      <Button
        variant={viewMode === 'list' ? 'primary' : 'ghost'}
        size="sm"
        onClick={handleListClick}
        onKeyDown={(e) => handleKeyDown(e, 'list')}
        aria-pressed={viewMode === 'list'}
        aria-label="Vista de lista"
        className="flex-1"
        tabIndex={0}
      >
        <List className="h-4 w-4 mr-2" aria-hidden="true" />
        Lista
      </Button>
      <Button
        variant={viewMode === 'table' ? 'primary' : 'ghost'}
        size="sm"
        onClick={handleTableClick}
        onKeyDown={(e) => handleKeyDown(e, 'table')}
        aria-pressed={viewMode === 'table'}
        aria-label="Vista de tabla"
        className="flex-1"
        tabIndex={0}
      >
        <Table className="h-4 w-4 mr-2" aria-hidden="true" />
        Tabla
      </Button>
    </div>
  );
};




