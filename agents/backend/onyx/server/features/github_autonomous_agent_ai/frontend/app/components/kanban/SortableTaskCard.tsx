'use client';

import { motion } from 'framer-motion';
import { format } from 'date-fns';
import { es } from 'date-fns/locale';
import { useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { Task } from '../../types/task';
import { calculateElapsedTime } from '../../utils/task-helpers';
import { cn } from '../../utils/cn';

interface SortableTaskCardProps {
  task: Task;
  onDelete?: (taskId: string) => void;
  onSelect?: (task: Task) => void;
  cardSize?: 'compact' | 'normal' | 'expanded';
  onQuickView?: (task: Task | null) => void;
}

export function SortableTaskCard({ 
  task, 
  onDelete, 
  onSelect, 
  cardSize = 'normal', 
  onQuickView 
}: SortableTaskCardProps) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id: task.id });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : 1,
  };

  // Determinar indicadores de estado
  const isProcessing = task.status === 'processing' || task.status === 'running';
  const isCompleted = task.status === 'completed';
  const isFailed = task.status === 'failed';
  const hasError = !!task.error;

  const handleDelete = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (onDelete && confirm('¿Estás seguro de que quieres eliminar esta tarea?')) {
      onDelete(task.id);
    }
  };

  return (
    <motion.div
      ref={setNodeRef}
      style={style}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.9 }}
      transition={{ duration: 0.2 }}
      {...attributes}
      {...listeners}
    >
      <div
        className={cn(
          "block bg-white border rounded-lg hover:shadow-md transition-all cursor-pointer group relative",
          cardSize === 'compact' ? "p-2" : cardSize === 'expanded' ? "p-4" : "p-3",
          isFailed || hasError ? "border-red-300 hover:border-red-400" :
          isCompleted ? "border-green-200 hover:border-green-300" :
          isProcessing ? "border-blue-200 hover:border-blue-300" :
          "border-gray-200 hover:border-blue-400"
        )}
        onClick={(e) => {
          // Permitir drag sin abrir detalles
          if (!isDragging && onSelect) {
            e.preventDefault();
            onSelect(task);
          }
        }}
        onMouseEnter={() => onQuickView?.(task)}
        onMouseLeave={() => onQuickView?.(null)}
        role="button"
        tabIndex={0}
        aria-label={`Tarea: ${task.instruction.substring(0, 50)}... Estado: ${task.status}`}
        onKeyDown={(e) => {
          if ((e.key === 'Enter' || e.key === ' ') && onSelect) {
            e.preventDefault();
            onSelect(task);
          }
        }}
      >
        <div className="space-y-2">
          {/* Header con repo y estado */}
          <div className="flex items-start justify-between gap-2">
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1">
                <p className="text-xs font-medium text-gray-600 truncate">
                  {task.repository}
                </p>
                {/* Indicadores de estado */}
                <div className="flex items-center gap-1 flex-shrink-0">
                  {isProcessing && (
                    <div className="relative">
                      <svg className="w-4 h-4 text-blue-500 animate-spin" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                    </div>
                  )}
                  {isCompleted && !hasError && (
                    <svg className="w-4 h-4 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  )}
                  {(isFailed || hasError) && (
                    <svg className="w-4 h-4 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  )}
                </div>
              </div>
              <p className={cn(
                "text-gray-800 font-medium",
                cardSize === 'compact' ? "text-xs line-clamp-1" :
                cardSize === 'expanded' ? "text-base line-clamp-3" :
                "text-sm line-clamp-2"
              )}>
                {task.instruction}
              </p>
            </div>
            {/* Acciones rápidas */}
            <div className="flex items-center gap-1 flex-shrink-0" onClick={(e) => e.stopPropagation()}>
              {onDelete && (
                <button
                  onClick={handleDelete}
                  className="p-1 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded transition-colors"
                  title="Eliminar tarea"
                  aria-label="Eliminar tarea"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                </button>
              )}
            </div>
          </div>

          {/* Timer para tareas en procesamiento */}
          {isProcessing && task.processingStartedAt && cardSize !== 'compact' && (
            <div className="flex items-center gap-1.5 text-xs text-blue-600 font-mono bg-blue-50 px-2 py-1 rounded">
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span>{calculateElapsedTime(task)}</span>
            </div>
          )}

          {/* Contenido de streaming/plan mejorado */}
          {task.streamingContent && cardSize !== 'compact' && (
            <div className="text-xs text-gray-600 bg-gray-50 px-2 py-1.5 rounded border border-gray-200">
              {(() => {
                let content = '';
                if (typeof task.streamingContent === 'string') {
                  content = task.streamingContent;
                } else {
                  content = JSON.stringify(task.streamingContent, null, 2);
                }
                
                // Intentar parsear como JSON para mostrar mejor
                try {
                  const parsed = typeof task.streamingContent === 'string' 
                    ? JSON.parse(task.streamingContent) 
                    : task.streamingContent;
                  
                  if (parsed.plan) {
                    return (
                      <div className="space-y-1">
                        <div className="font-semibold text-gray-700">Plan:</div>
                        {parsed.plan.steps && Array.isArray(parsed.plan.steps) && (
                          <ul className="list-disc list-inside space-y-0.5 text-gray-600">
                            {parsed.plan.steps.slice(0, 3).map((step: string, idx: number) => (
                              <li key={idx} className="line-clamp-1">{step}</li>
                            ))}
                            {parsed.plan.steps.length > 3 && (
                              <li className="text-gray-400 italic">+{parsed.plan.steps.length - 3} pasos más</li>
                            )}
                          </ul>
                        )}
                      </div>
                    );
                  }
                } catch {
                  // No es JSON válido, mostrar como texto
                }
                
                return (
                  <div className="line-clamp-3 italic">
                    {content.substring(0, 150)}
                    {content.length > 150 && '...'}
                  </div>
                );
              })()}
            </div>
          )}

          {/* Error message */}
          {hasError && (
            <div className="text-xs text-red-600 line-clamp-2 bg-red-50 px-2 py-1 rounded flex items-start gap-1">
              <svg className="w-3 h-3 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span>{task.error}</span>
            </div>
          )}

          {/* Resultados de ejecución/commits mejorado */}
          {task.executionResult && (
            <div className={cn(
              "text-xs rounded-lg overflow-hidden",
              task.executionResult.success 
                ? "bg-gradient-to-r from-green-50 to-emerald-50 text-green-800 border border-green-300 shadow-sm" 
                : "bg-red-50 text-red-700 border border-red-200"
            )}>
              {task.executionResult.success ? (
                <div className="px-2.5 py-2">
                  <div className="flex items-start gap-2.5">
                    <div className={cn(
                      "flex-shrink-0 rounded-full p-1",
                      cardSize === 'compact' ? "p-0.5" : "p-1",
                      "bg-green-500"
                    )}>
                      <svg className={cn(
                        "text-white",
                        cardSize === 'compact' ? "w-2.5 h-2.5" : "w-3 h-3"
                      )} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                      </svg>
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-semibold text-green-900">Commit exitoso</span>
                        {task.executionResult.branch && (
                          <span className="text-xs bg-green-200 text-green-800 px-1.5 py-0.5 rounded font-mono">
                            {task.executionResult.branch}
                          </span>
                        )}
                      </div>
                      {task.executionResult.commitSha && (
                        <div className="flex items-center gap-2 flex-wrap">
                          <div className="flex items-center gap-1 bg-white px-2 py-0.5 rounded border border-green-200">
                            <svg className="w-3 h-3 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                            </svg>
                            <span className="font-mono text-xs text-gray-700 font-semibold">
                              {task.executionResult.commitSha.substring(0, 7)}
                            </span>
                          </div>
                          {task.executionResult.commitUrl && (
                            <a
                              href={task.executionResult.commitUrl}
                              target="_blank"
                              rel="noopener noreferrer"
                              onClick={(e) => e.stopPropagation()}
                              className="flex items-center gap-1 text-blue-600 hover:text-blue-800 hover:bg-blue-50 px-2 py-0.5 rounded transition-colors font-medium"
                              title="Ver commit en GitHub"
                            >
                              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                              </svg>
                              <span>Ver en GitHub</span>
                            </a>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="px-2.5 py-2 flex items-start gap-2">
                  <svg className="w-3 h-3 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                  <span className="line-clamp-2 flex-1">{task.executionResult.error || 'Error en ejecución'}</span>
                </div>
              )}
            </div>
          )}
          
          {/* Pending commit approval */}
          {task.pendingCommitApproval && (
            <div className="text-xs bg-purple-50 text-purple-700 px-2 py-1.5 rounded border border-purple-200 flex items-center gap-2">
              <svg className="w-3 h-3 animate-pulse" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span>Pendiente aprobación de commit</span>
            </div>
          )}

          {/* Footer con fecha */}
          {cardSize !== 'compact' && (
            <div className="flex items-center justify-between pt-1 border-t border-gray-100">
              <div className={cn(
                "text-gray-400",
                cardSize === 'expanded' ? "text-sm" : "text-xs"
              )}>
                {format(new Date(task.createdAt), cardSize === 'expanded' ? 'dd MMM yyyy, HH:mm:ss' : 'dd MMM yyyy, HH:mm', { locale: es })}
              </div>
              <div className="flex items-center gap-2">
                {task.executionResult?.branch && (
                  <div className={cn(
                    "text-gray-500 bg-gray-100 px-2 py-0.5 rounded font-mono",
                    cardSize === 'expanded' ? "text-xs" : "text-xs"
                  )}>
                    {task.executionResult.branch}
                  </div>
                )}
                {task.model && (
                  <div className={cn(
                    "text-gray-400 bg-gray-100 px-2 py-0.5 rounded",
                    cardSize === 'expanded' ? "text-xs" : "text-xs"
                  )}>
                    {task.model}
                  </div>
                )}
              </div>
            </div>
          )}
          
          {/* Footer compacto */}
          {cardSize === 'compact' && (
            <div className="text-xs text-gray-400 mt-1">
              {format(new Date(task.createdAt), 'dd/MM HH:mm', { locale: es })}
            </div>
          )}
        </div>
        {/* Botón para abrir en nueva pestaña */}
        <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
          <a
            href={`/tasks/${task.id}`}
            target="_blank"
            rel="noopener noreferrer"
            onClick={(e) => e.stopPropagation()}
            className="p-1 text-gray-400 hover:text-blue-600 hover:bg-blue-50 rounded"
            title="Abrir en nueva pestaña"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
            </svg>
          </a>
        </div>
      </div>
    </motion.div>
  );
}

