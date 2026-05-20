'use client';

import { format } from 'date-fns';
import { es } from 'date-fns/locale';
import { Task } from '../../types/task';
import { cn } from '../../utils/cn';

interface TimelineViewProps {
  tasks: Task[];
  onSelectTask: (task: Task) => void;
  onQuickView: (task: Task | null) => void;
}

export function TimelineView({ tasks, onSelectTask, onQuickView }: TimelineViewProps) {
  const sortedTasks = [...tasks].sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());

  return (
    <div className="max-w-4xl mx-auto">
      <div className="relative">
        {/* Línea vertical del timeline */}
        <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-gray-300"></div>
        
        <div className="space-y-6">
          {sortedTasks.map((task) => {
            const isProcessing = task.status === 'processing' || task.status === 'running';
            const isCompleted = task.status === 'completed';
            const isFailed = task.status === 'failed';
            
            return (
              <div key={task.id} className="relative flex items-start gap-4">
                {/* Punto en el timeline */}
                <div className={cn(
                  "relative z-10 w-8 h-8 rounded-full border-2 flex items-center justify-center flex-shrink-0",
                  isCompleted ? "bg-green-500 border-green-600" :
                  isFailed ? "bg-red-500 border-red-600" :
                  isProcessing ? "bg-blue-500 border-blue-600 animate-pulse" :
                  "bg-gray-400 border-gray-500"
                )}>
                  {isCompleted && (
                    <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  )}
                  {isFailed && (
                    <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  )}
                </div>
                
                {/* Contenido de la tarjeta */}
                <div 
                  className={cn(
                    "flex-1 bg-white border rounded-lg p-4 shadow-sm hover:shadow-md transition-all cursor-pointer",
                    isCompleted ? "border-green-200" :
                    isFailed ? "border-red-200" :
                    isProcessing ? "border-blue-200" :
                    "border-gray-200"
                  )}
                  onClick={() => onSelectTask(task)}
                  onMouseEnter={() => onQuickView(task)}
                  onMouseLeave={() => onQuickView(null)}
                >
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-xs font-medium text-gray-600">{task.repository}</span>
                        <span className={cn(
                          "px-2 py-0.5 rounded-full text-xs font-medium",
                          isCompleted ? "bg-green-100 text-green-800" :
                          isFailed ? "bg-red-100 text-red-800" :
                          isProcessing ? "bg-blue-100 text-blue-800" :
                          "bg-gray-100 text-gray-800"
                        )}>
                          {task.status}
                        </span>
                      </div>
                      <p className="text-sm font-medium text-gray-900 line-clamp-2">{task.instruction}</p>
                    </div>
                    <div className="text-xs text-gray-400 ml-4">
                      {format(new Date(task.createdAt), 'dd MMM, HH:mm', { locale: es })}
                    </div>
                  </div>
                  
                  {task.executionResult?.success && (
                    <div className="mt-2 text-xs text-green-600 flex items-center gap-1">
                      <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                      Commit exitoso
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

