'use client';

import Link from 'next/link';
import { useEffect, useState } from 'react';
import { format } from 'date-fns';
import { es } from 'date-fns/locale';
import { motion } from 'framer-motion';
import { Task } from '../../types/task';
import { getStatusBadgeColor, getStatusLabel, isTaskActive, calculateElapsedTime } from '../../utils/task-helpers';
import { cn } from '../../utils/cn';
import { StreamingContent } from './StreamingContent';

interface TaskCardProps {
  task: Task;
  onStop?: () => void;
  onResume?: () => void;
  onDelete?: () => void;
  onCommit?: () => void;
  showActions?: boolean;
  compact?: boolean;
  isCommitting?: boolean;
}

export function TaskCard({ task, onStop, onResume, onDelete, onCommit, showActions = true, compact = false, isCommitting = false }: TaskCardProps) {
  const [timerTick, setTimerTick] = useState(0);

  useEffect(() => {
    if (task.status !== 'processing' || !task.processingStartedAt) {
      return;
    }

    const interval = setInterval(() => {
      setTimerTick((prev) => prev + 1);
    }, 1000);

    return () => clearInterval(interval);
  }, [task.status, task.processingStartedAt]);

  return (
    <div className={cn(
      "border border-gray-200 rounded-lg transition-colors",
      compact ? "p-3" : "p-5",
      "hover:border-gray-300"
    )}>
      <div className="flex justify-between items-start mb-3">
        <div className="flex-1 pr-4">
          <span className="text-black font-medium text-sm block mb-1">{task.repository}</span>
          <p className={cn("text-gray-600 mb-2", compact ? "text-xs line-clamp-1" : "text-sm line-clamp-2")}>
            {task.instruction}
          </p>
          <p className="text-xs text-gray-400">
            Creada: {format(new Date(task.createdAt), 'dd MMM yyyy, HH:mm', { locale: es })}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className={cn(
            "px-3 py-1.5 rounded-md text-xs font-medium whitespace-nowrap",
            task.status === 'pending_approval' 
              ? "bg-yellow-100 text-yellow-800"
              : task.status === 'pending_commit_approval'
              ? "bg-purple-100 text-purple-800"
              : getStatusBadgeColor(task.status)
          )}>
            {task.status === 'pending_approval' 
              ? 'Pendiente Aprobación Plan' 
              : task.status === 'pending_commit_approval'
              ? 'Pendiente Aprobación Commit'
              : getStatusLabel(task.status)}
          </span>
          {/* Botón de Comitar - SIEMPRE VISIBLE si hay plan o contenido generado (incluso si está detenida o completada) */}
          {((task.pendingApproval && (task.pendingApproval.actions?.length > 0 || task.pendingApproval.approved)) || 
            (task.streamingContent && task.streamingContent.length > 0) ||
            (task.result?.plan && (task.result.plan.files_to_create?.length > 0 || task.result.plan.files_to_modify?.length > 0))) && onCommit && (
            <button
              onClick={onCommit}
              disabled={isCommitting}
              className="px-3 py-1.5 bg-green-600 text-white hover:bg-green-700 rounded-md transition-colors flex items-center gap-1.5 text-xs font-medium shadow-sm whitespace-nowrap disabled:opacity-70 disabled:cursor-not-allowed"
              title={isCommitting ? "Comitando cambios..." : "Comitar cambios en GitHub"}
              aria-label="Comitar cambios"
            >
              {isCommitting ? (
                <>
                  <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  <span>Comitando...</span>
                </>
              ) : (
                <>
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  <span>Comitar</span>
                </>
              )}
            </button>
          )}
          {showActions && (
            <>
              {task.status === 'stopped' && onResume && (
                <button
                  onClick={onResume}
                  className="p-1.5 text-green-500 hover:text-green-700 hover:bg-green-50 rounded transition-colors"
                  title="Reanudar procesamiento"
                  aria-label="Reanudar tarea"
                >
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M8 5v14l11-7z" />
                  </svg>
                </button>
              )}
              {/* Botón de parar durante procesamiento */}
              {(isTaskActive(task) && !task.pendingApproval?.approved && onStop) && (
                <button
                  onClick={onStop}
                  className="p-1.5 text-orange-500 hover:text-orange-700 hover:bg-orange-50 rounded transition-colors"
                  title="Detener procesamiento"
                  aria-label="Detener tarea"
                >
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M6 6h12v12H6z" />
                  </svg>
                </button>
              )}
              <Link
                href={`/tasks/${task.id}`}
                className="p-1.5 text-gray-400 hover:text-blue-600 hover:bg-blue-50 rounded transition-colors"
                title="Ver detalles de la tarea"
                aria-label="Ver detalles"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                </svg>
              </Link>
              {onDelete && (
                <button
                  onClick={onDelete}
                  className="p-1.5 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded transition-colors"
                  title="Eliminar tarea"
                  aria-label="Eliminar tarea"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                </button>
              )}
            </>
          )}
        </div>
      </div>

      {/* Contenido de streaming mejorado */}
      {(task.status === 'processing' || task.status === 'stopped') && task.streamingContent && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="mt-4 bg-gradient-to-br from-yellow-50 to-amber-50 border border-yellow-200 rounded-lg shadow-sm overflow-hidden"
        >
          <div className="p-3 bg-yellow-100 border-b border-yellow-200">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                {task.status === 'processing' ? (
                  <>
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                      className="text-lg"
                    >
                      🔄
                    </motion.div>
                    <span className="text-xs font-semibold text-yellow-900">Generando contenido...</span>
                  </>
                ) : (
                  <>
                    <span className="text-lg">⏸️</span>
                    <span className="text-xs font-semibold text-yellow-900">Contenido generado (tarea detenida)</span>
                  </>
                )}
              </div>
              {task.status === 'processing' && (
                <motion.div
                  className="flex items-center gap-1"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                >
                  <motion.div
                    className="w-2 h-2 bg-green-500 rounded-full"
                    animate={{ scale: [1, 1.2, 1], opacity: [1, 0.7, 1] }}
                    transition={{ duration: 1.5, repeat: Infinity }}
                  />
                  <span className="text-xs text-yellow-700">En vivo</span>
                </motion.div>
              )}
            </div>
          </div>
          <div className="bg-white">
            <StreamingContent
              content={typeof task.streamingContent === 'string' 
                ? task.streamingContent 
                : JSON.stringify(task.streamingContent, null, 2)}
              isProcessing={task.status === 'processing'}
              className="min-h-[200px]"
            />
          </div>
        </motion.div>
      )}

      {/* Plan de cambios pendiente (incluso si está detenida) */}
      {task.pendingApproval && (
        <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <div className="flex items-center justify-between mb-3">
            <p className="text-xs font-medium text-blue-800 flex items-center gap-2">
              <span>📋</span>
              {task.pendingApproval.approved ? (
                <>
                  <span className="animate-pulse">🔄</span>
                  <span>Plan aprobado - Listo para comitar</span>
                </>
              ) : (
                <span>Plan de cambios listo para aprobar</span>
              )}
            </p>
            {task.status === 'stopped' && !task.pendingApproval.approved && (
              <span className="text-xs text-orange-600">(Tarea detenida - puedes reanudar para aprobar)</span>
            )}
            {/* Botón de Comitar visible en la sección del plan */}
            {task.pendingApproval.approved && onCommit && (
              <button
                onClick={onCommit}
                className="px-3 py-1.5 bg-green-600 text-white hover:bg-green-700 rounded-md transition-colors flex items-center gap-1.5 text-xs font-medium shadow-sm whitespace-nowrap"
                title="Comitar cambios en GitHub"
                aria-label="Comitar cambios"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                <span>Comitar</span>
              </button>
            )}
          </div>
          <div className="space-y-2">
            <div className="text-sm text-blue-700">
              <strong>Mensaje de commit:</strong>
              <p className="mt-1 p-2 bg-white rounded border border-blue-200 text-xs font-mono">
                {task.pendingApproval.commitMessage}
              </p>
            </div>
            <div className="text-sm text-blue-700">
              <strong>Archivos a modificar ({task.pendingApproval.actions.length}):</strong>
              <ul className="mt-1 space-y-1">
                {task.pendingApproval.actions.map((action, idx) => (
                  <li key={idx} className="text-xs flex items-center gap-2">
                    <span className={action.action === 'create' ? 'text-green-600' : 'text-blue-600'}>
                      {action.action === 'create' ? '➕' : '✏️'}
                    </span>
                    <code className="bg-white px-2 py-1 rounded border border-blue-200">
                      {action.path}
                    </code>
                    <span className="text-gray-500">
                      ({action.content.length} caracteres)
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* Cronómetro para tareas procesando */}
      {task.status === 'processing' && task.processingStartedAt && (
        <div className="mt-2 flex items-center gap-1 text-xs text-blue-600 font-mono">
          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          {/* timerTick solo fuerza re-render; no se usa directamente */}
          <span className="tabular-nums">{calculateElapsedTime(task)}</span>
        </div>
      )}

      {/* Error */}
      {task.error && (
        <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-sm text-red-700">
            <strong>Error:</strong> {task.error}
          </p>
        </div>
      )}

      {/* Estado de ejecución */}
      {task.executionStatus && (
        <div className="mt-3 p-3 bg-purple-50 border border-purple-200 rounded-lg">
          <p className="text-xs font-medium text-purple-800 mb-2 flex items-center gap-2">
            {task.executionStatus === 'executing' && (
              <>
                <span className="animate-spin">⚙️</span>
                Ejecutando cambios en GitHub...
              </>
            )}
            {task.executionStatus === 'completed' && task.executionResult?.success && (
              <>
                <span>✅</span>
                Cambios aplicados exitosamente
              </>
            )}
            {task.executionStatus === 'failed' && (
              <>
                <span>❌</span>
                Error al aplicar cambios
              </>
            )}
          </p>
          {task.executionResult?.commitUrl && (
            <a
              href={task.executionResult.commitUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-purple-700 hover:text-purple-900 underline flex items-center gap-1"
            >
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
              </svg>
              Ver commit {task.executionResult.commitSha?.substring(0, 7)}
            </a>
          )}
          {task.executionResult?.error && (
            <p className="text-xs text-red-600 mt-1">
              {task.executionResult.error}
            </p>
          )}
        </div>
      )}

      {/* Resultados */}
      {task.status === 'completed' && task.result && (
        <div className="mt-4 space-y-3">
          {task.result.content && (
            <div className="p-3 bg-green-50 border border-green-200 rounded-lg">
              <p className="text-xs font-medium text-green-800 mb-2">Resultado:</p>
              <p className="text-sm text-green-700 whitespace-pre-wrap">
                {typeof task.result.content === 'string' 
                  ? task.result.content 
                  : JSON.stringify(task.result.content, null, 2)}
              </p>
            </div>
          )}

          {task.result.plan && (
            <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
              <p className="text-xs font-medium text-blue-800 mb-2">Plan de Acción:</p>
              {task.result.plan.steps && Array.isArray(task.result.plan.steps) && (
                <ol className="list-decimal list-inside text-sm text-blue-700 space-y-1">
                  {task.result.plan.steps.map((step: any, idx: number) => (
                    <li key={idx}>
                      {typeof step === 'string' ? step : JSON.stringify(step)}
                    </li>
                  ))}
                </ol>
              )}
              {task.result.plan.files_to_create && (
                <div className="mt-2">
                  <p className="text-xs font-medium text-blue-800">Archivos a crear:</p>
                  {Array.isArray(task.result.plan.files_to_create) ? (
                    <ul className="list-disc list-inside text-sm text-blue-700">
                      {task.result.plan.files_to_create.map((file: any, idx: number) => (
                        <li key={idx}>
                          {typeof file === 'string' ? file : JSON.stringify(file)}
                        </li>
                      ))}
                    </ul>
                  ) : typeof task.result.plan.files_to_create === 'object' ? (
                    <ul className="list-disc list-inside text-sm text-blue-700">
                      {Object.keys(task.result.plan.files_to_create).map((file: string, idx: number) => (
                        <li key={idx}>{file}</li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-xs text-blue-600">
                      {JSON.stringify(task.result.plan.files_to_create)}
                    </p>
                  )}
                </div>
              )}
              {task.result.plan.files_to_modify && (
                <div className="mt-2">
                  <p className="text-xs font-medium text-blue-800">Archivos a modificar:</p>
                  {Array.isArray(task.result.plan.files_to_modify) ? (
                    <ul className="list-disc list-inside text-sm text-blue-700">
                      {task.result.plan.files_to_modify.map((file: any, idx: number) => (
                        <li key={idx}>
                          {typeof file === 'string' ? file : JSON.stringify(file)}
                        </li>
                      ))}
                    </ul>
                  ) : typeof task.result.plan.files_to_modify === 'object' ? (
                    <ul className="list-disc list-inside text-sm text-blue-700">
                      {Object.keys(task.result.plan.files_to_modify).map((file: string, idx: number) => (
                        <li key={idx}>{file}</li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-xs text-blue-600">
                      {JSON.stringify(task.result.plan.files_to_modify)}
                    </p>
                  )}
                </div>
              )}
            </div>
          )}

          {task.result.code && (
            <div className="p-3 bg-gray-50 border border-gray-200 rounded-lg">
              <p className="text-xs font-medium text-gray-800 mb-2">Código Generado:</p>
              <pre className="text-xs text-gray-700 overflow-x-auto bg-white p-2 rounded border">
                <code>
                  {typeof task.result.code === 'string' 
                    ? task.result.code 
                    : JSON.stringify(task.result.code, null, 2)}
                </code>
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

