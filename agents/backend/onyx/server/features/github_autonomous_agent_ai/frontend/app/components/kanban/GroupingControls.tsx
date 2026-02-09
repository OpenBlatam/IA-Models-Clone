'use client';

import { cn } from '../../utils/cn';

interface GroupingControlsProps {
  groupByRepo: boolean;
  groupByModel: boolean;
  onToggleRepo: () => void;
  onToggleModel: () => void;
}

export function GroupingControls({
  groupByRepo,
  groupByModel,
  onToggleRepo,
  onToggleModel,
}: GroupingControlsProps) {
  return (
    <div className="flex items-center gap-1">
      <button
        onClick={onToggleRepo}
        className={cn(
          'px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2',
          groupByRepo
            ? 'bg-blue-100 text-blue-700 border border-blue-300'
            : 'bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300'
        )}
        title="Agrupar tareas por repositorio"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"
          />
        </svg>
        Repo
      </button>
      <button
        onClick={onToggleModel}
        className={cn(
          'px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2',
          groupByModel
            ? 'bg-purple-100 text-purple-700 border border-purple-300'
            : 'bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300'
        )}
        title="Agrupar tareas por modelo"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
          />
        </svg>
        Modelo
      </button>
    </div>
  );
}

