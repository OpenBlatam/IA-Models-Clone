/**
 * View switcher component for different view modes
 */

'use client';

import React from 'react';
import { Button } from '@/components/ui';
import { List, Table, LayoutGrid, Calendar } from 'lucide-react';
import { cn } from '@/lib/utils/cn';

export type ViewType = 'list' | 'table' | 'grid' | 'timeline';

export interface ViewSwitcherProps {
  currentView: ViewType;
  onViewChange: (view: ViewType) => void;
  availableViews?: ViewType[];
  className?: string;
}

const VIEW_OPTIONS: Array<{ value: ViewType; label: string; icon: React.ComponentType<{ className?: string }> }> = [
  { value: 'list', label: 'Lista', icon: List },
  { value: 'table', label: 'Tabla', icon: Table },
  { value: 'grid', label: 'Grid', icon: LayoutGrid },
  { value: 'timeline', label: 'Timeline', icon: Calendar },
];

export const ViewSwitcher: React.FC<ViewSwitcherProps> = ({
  currentView,
  onViewChange,
  availableViews = ['list', 'table', 'grid', 'timeline'],
  className,
}) => {
  const handleViewChange = (view: ViewType) => {
    onViewChange(view);
  };

  const handleKeyDown = (event: React.KeyboardEvent, view: ViewType) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      handleViewChange(view);
    }
  };

  const filteredViews = VIEW_OPTIONS.filter((view) => availableViews.includes(view.value));

  return (
    <div
      className={cn('flex items-center gap-1 border rounded-lg p-1', className)}
      role="tablist"
      aria-label="Cambiar vista"
    >
      {filteredViews.map((view) => {
        const Icon = view.icon;
        const isActive = currentView === view.value;

        return (
          <Button
            key={view.value}
            variant={isActive ? 'primary' : 'ghost'}
            size="sm"
            onClick={() => handleViewChange(view.value)}
            onKeyDown={(e) => handleKeyDown(e, view.value)}
            aria-pressed={isActive}
            aria-label={`Vista ${view.label}`}
            className="flex-1"
            tabIndex={0}
          >
            <Icon className="h-4 w-4 mr-2" aria-hidden="true" />
            {view.label}
          </Button>
        );
      })}
    </div>
  );
};
