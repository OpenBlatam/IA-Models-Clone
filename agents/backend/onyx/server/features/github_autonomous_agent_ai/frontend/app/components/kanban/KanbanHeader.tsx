'use client';

import Link from 'next/link';
import { Task } from '../../types/task';
import { cn } from '../../utils/cn';

interface KanbanHeaderProps {
  tasks: Task[];
  filteredTasks: Task[];
  searchQuery: string;
  selectedRepository: string;
  onClearFilters?: () => void;
}

export function KanbanHeader({
  tasks,
  filteredTasks,
  searchQuery,
  selectedRepository,
  onClearFilters,
}: KanbanHeaderProps) {
  return (
    <header className="bg-white border-b border-gray-200 sticky top-0 z-50 shadow-sm">
      <div className="max-w-full mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link
              href="/"
              className="flex items-center gap-2.5 text-base md:text-lg text-black hover:opacity-80 transition-opacity"
            >
              <div className="w-6 h-6 md:w-7 md:h-7 flex items-center justify-center flex-shrink-0">
                <svg viewBox="0 0 24 24" className="w-full h-full">
                  <defs>
                    <linearGradient id="gradient-header" x1="0%" y1="0%" x2="0%" y2="100%">
                      <stop offset="0%" stopColor="#8800ff" />
                      <stop offset="16.66%" stopColor="#0000ff" />
                      <stop offset="33.33%" stopColor="#0088ff" />
                      <stop offset="50%" stopColor="#00ff00" />
                      <stop offset="66.66%" stopColor="#ffdd00" />
                      <stop offset="83.33%" stopColor="#ff8800" />
                      <stop offset="100%" stopColor="#ff0000" />
                    </linearGradient>
                  </defs>
                  <path
                    d="M7 20L12 4L17 20H14.5L12 12.5L9.5 20H7Z"
                    fill="url(#gradient-header)"
                  />
                </svg>
              </div>
              <span className="font-normal">
                <span className="font-light">GitHub</span>{' '}
                <span className="font-normal">Autonomous Agent AI</span>
              </span>
            </Link>
            <h1 className="text-xl font-bold">Vista Kanban</h1>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-sm text-gray-600">
              Total: <span className="font-semibold">{tasks.length}</span> tareas
              {(searchQuery || selectedRepository !== 'all') && (
                <span className="ml-2 text-blue-600">
                  ({filteredTasks.length}{' '}
                  {filteredTasks.length === 1 ? 'encontrada' : 'encontradas'})
                </span>
              )}
            </span>
            {(searchQuery || selectedRepository !== 'all') && onClearFilters && (
              <button
                onClick={onClearFilters}
                className="px-3 py-1.5 text-sm text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-lg transition-colors"
              >
                Limpiar filtros
              </button>
            )}
            <Link
              href="/agent-control"
              className="px-4 py-2 bg-black text-white rounded-lg hover:bg-gray-800 transition-colors text-sm"
            >
              Agent Control
            </Link>
          </div>
        </div>
      </div>
    </header>
  );
}

