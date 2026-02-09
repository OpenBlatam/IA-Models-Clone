'use client';

import { format } from 'date-fns';
import { es } from 'date-fns/locale';
import { Task, TaskStatus } from '../../types/task';
import { cn } from '../../utils/cn';

interface ListViewProps {
  tasks: Task[];
  allTasksCount: number;
  onSelectTask: (task: Task) => void;
  onDeleteTask: (taskId: string) => void;
  sortBy: 'date' | 'status' | 'repo' | 'model';
  setSortBy: (sort: 'date' | 'status' | 'repo' | 'model') => void;
  sortOrder: 'asc' | 'desc';
  setSortOrder: (order: 'asc' | 'desc') => void;
}

export function ListView({
  tasks,
  allTasksCount,
  onSelectTask,
  onDeleteTask,
  sortBy,
  setSortBy,
  sortOrder,
  setSortOrder,
}: ListViewProps) {
  return (
    <div className="max-w-full">
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
        <table className="w-full">
          <thead className="bg-gray-50 border-b border-gray-200">
            <tr>
              <th 
                className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                onClick={() => {
                  if (sortBy === 'repo') {
                    setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
                  } else {
                    setSortBy('repo');
                    setSortOrder('asc');
                  }
                }}
              >
                <div className="flex items-center gap-1">
                  Repositorio
                  {sortBy === 'repo' && (
                    sortOrder === 'asc' ? '↑' : '↓'
                  )}
                </div>
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Instrucción</th>
              <th 
                className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                onClick={() => {
                  if (sortBy === 'status') {
                    setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
                  } else {
                    setSortBy('status');
                    setSortOrder('asc');
                  }
                }}
              >
                <div className="flex items-center gap-1">
                  Estado
                  {sortBy === 'status' && (
                    sortOrder === 'asc' ? '↑' : '↓'
                  )}
                </div>
              </th>
              <th 
                className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                onClick={() => {
                  if (sortBy === 'model') {
                    setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
                  } else {
                    setSortBy('model');
                    setSortOrder('asc');
                  }
                }}
              >
                <div className="flex items-center gap-1">
                  Modelo
                  {sortBy === 'model' && (
                    sortOrder === 'asc' ? '↑' : '↓'
                  )}
                </div>
              </th>
              <th 
                className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                onClick={() => {
                  if (sortBy === 'date') {
                    setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
                  } else {
                    setSortBy('date');
                    setSortOrder('desc');
                  }
                }}
              >
                <div className="flex items-center gap-1">
                  Creada
                  {sortBy === 'date' && (
                    sortOrder === 'asc' ? '↑' : '↓'
                  )}
                </div>
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Acciones</th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {tasks.length === 0 ? (
              <tr>
                <td colSpan={6} className="px-4 py-12 text-center text-gray-500">
                  No hay tareas que coincidan con los filtros
                </td>
              </tr>
            ) : (
              tasks.map((task) => {
                const isProcessing = task.status === 'processing' || task.status === 'running';
                const isCompleted = task.status === 'completed';
                const isFailed = task.status === 'failed';
                
                return (
                  <tr
                    key={task.id}
                    className="hover:bg-gray-50 cursor-pointer transition-colors"
                    onClick={() => onSelectTask(task)}
                  >
                    <td className="px-4 py-3 whitespace-nowrap">
                      <div className="text-sm font-medium text-gray-900">{task.repository}</div>
                    </td>
                    <td className="px-4 py-3">
                      <div className="text-sm text-gray-900 line-clamp-2 max-w-md">{task.instruction}</div>
                      {task.error && (
                        <div className="text-xs text-red-600 mt-1 line-clamp-1">⚠️ {task.error}</div>
                      )}
                      {task.executionResult?.success && (
                        <div className="text-xs text-green-600 mt-1 flex items-center gap-1">
                          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                          </svg>
                          Commit exitoso
                        </div>
                      )}
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap">
                      <div className="flex items-center gap-2">
                        {isProcessing && (
                          <svg className="w-4 h-4 text-blue-500 animate-spin" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                          </svg>
                        )}
                        {isCompleted && !task.error && (
                          <svg className="w-4 h-4 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                          </svg>
                        )}
                        {isFailed && (
                          <svg className="w-4 h-4 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                          </svg>
                        )}
                        <span className={cn(
                          "px-2 py-1 rounded-full text-xs font-medium",
                          task.status === 'completed' ? 'bg-green-100 text-green-800' :
                          task.status === 'processing' || task.status === 'running' ? 'bg-blue-100 text-blue-800' :
                          task.status === 'failed' ? 'bg-red-100 text-red-800' :
                          task.status === 'stopped' ? 'bg-orange-100 text-orange-800' :
                          'bg-gray-100 text-gray-800'
                        )}>
                          {task.status}
                        </span>
                      </div>
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap">
                      {task.model ? (
                        <span className="text-xs text-gray-600 font-mono">{task.model}</span>
                      ) : (
                        <span className="text-xs text-gray-400">-</span>
                      )}
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500">
                      {format(new Date(task.createdAt), 'dd MMM yyyy', { locale: es })}
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap text-sm">
                      <div className="flex items-center gap-2">
                        <a
                          href={`/tasks/${task.id}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          onClick={(e) => e.stopPropagation()}
                          className="text-blue-600 hover:text-blue-800"
                          title="Abrir en nueva pestaña"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                          </svg>
                        </a>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            if (confirm('¿Eliminar esta tarea?')) {
                              onDeleteTask(task.id);
                            }
                          }}
                          className="text-red-600 hover:text-red-800"
                          title="Eliminar"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                          </svg>
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
      {tasks.length > 0 && (
        <div className="mt-4 text-sm text-gray-600 text-center">
          Mostrando {tasks.length} de {allTasksCount} tareas
        </div>
      )}
    </div>
  );
}

