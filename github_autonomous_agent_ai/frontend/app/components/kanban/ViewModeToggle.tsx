'use client';

import { cn } from '../../utils/cn';

interface ViewModeToggleProps {
  viewMode: 'kanban' | 'list' | 'timeline' | 'calendar';
  onViewModeChange: (mode: 'kanban' | 'list' | 'timeline' | 'calendar') => void;
}

export function ViewModeToggle({ viewMode, onViewModeChange }: ViewModeToggleProps) {
  return (
    <div className="flex items-center gap-1 bg-gray-100 rounded-lg p-1 border border-gray-300">
      <button
        onClick={() => onViewModeChange('kanban')}
        className={cn(
          'px-3 py-1.5 rounded-md text-sm font-medium transition-colors flex items-center gap-1.5',
          viewMode === 'kanban'
            ? 'bg-white text-gray-900 shadow-sm'
            : 'text-gray-600 hover:text-gray-900'
        )}
        title="Vista Kanban"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2"
          />
        </svg>
        Kanban
      </button>
      <button
        onClick={() => onViewModeChange('list')}
        className={cn(
          'px-3 py-1.5 rounded-md text-sm font-medium transition-colors flex items-center gap-1.5',
          viewMode === 'list'
            ? 'bg-white text-gray-900 shadow-sm'
            : 'text-gray-600 hover:text-gray-900'
        )}
        title="Vista Lista"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M4 6h16M4 12h16M4 18h16"
          />
        </svg>
        Lista
      </button>
      <button
        onClick={() => onViewModeChange('timeline')}
        className={cn(
          'px-3 py-1.5 rounded-md text-sm font-medium transition-colors flex items-center gap-1.5',
          viewMode === 'timeline'
            ? 'bg-white text-gray-900 shadow-sm'
            : 'text-gray-600 hover:text-gray-900'
        )}
        title="Vista Timeline"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
        Timeline
      </button>
      <button
        onClick={() => onViewModeChange('calendar')}
        className={cn(
          'px-3 py-1.5 rounded-md text-sm font-medium transition-colors flex items-center gap-1.5',
          viewMode === 'calendar'
            ? 'bg-white text-gray-900 shadow-sm'
            : 'text-gray-600 hover:text-gray-900'
        )}
        title="Vista Calendario"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"
          />
        </svg>
        Calendario
      </button>
    </div>
  );
}

