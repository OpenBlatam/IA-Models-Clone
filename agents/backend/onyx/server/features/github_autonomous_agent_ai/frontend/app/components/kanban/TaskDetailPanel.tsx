'use client';

import { motion } from 'framer-motion';
import { format } from 'date-fns';
import { es } from 'date-fns/locale';
import { Task } from '../../types/task';
import { cn } from '../../utils/cn';

interface TaskDetailPanelProps {
  task: Task | null;
  onClose: () => void;
  onDelete: (taskId: string) => void;
}

export function TaskDetailPanel({ task, onClose, onDelete }: TaskDetailPanelProps) {
  if (!task) return null;

  return (
    <>
      <div
        className="fixed inset-0 bg-black bg-opacity-50 z-40"
        onClick={onClose}
      />
      <motion.div
        initial={{ x: '100%' }}
        animate={{ x: 0 }}
        exit={{ x: '100%' }}
        transition={{ type: 'spring', damping: 30, stiffness: 300 }}
        className="fixed right-0 top-0 h-full w-full max-w-2xl bg-white shadow-2xl z-50 overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="sticky top-0 bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between z-10">
          <h2 className="text-xl font-bold">Detalles de la Tarea</h2>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            aria-label="Cerrar panel"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        
        <div className="p-6 space-y-6">
          <div>
            <h3 className="text-sm font-medium text-gray-500 mb-2">Repositorio</h3>
            <p className="text-base font-medium">{task.repository}</p>
          </div>
          
          <div>
            <h3 className="text-sm font-medium text-gray-500 mb-2">Instrucción</h3>
            <p className="text-base">{task.instruction}</p>
          </div>
          
          <div>
            <h3 className="text-sm font-medium text-gray-500 mb-2">Estado</h3>
            <span className={cn(
              "inline-block px-3 py-1 rounded-full text-sm font-medium",
              task.status === 'completed' ? 'bg-green-100 text-green-800' :
              task.status === 'processing' || task.status === 'running' ? 'bg-blue-100 text-blue-800' :
              task.status === 'failed' ? 'bg-red-100 text-red-800' :
              'bg-gray-100 text-gray-800'
            )}>
              {task.status}
            </span>
          </div>
          
          <div className="grid grid-cols-2 gap-4">
            <div>
              <h3 className="text-sm font-medium text-gray-500 mb-2">Creada</h3>
              <p className="text-sm">{format(new Date(task.createdAt), 'dd MMM yyyy, HH:mm', { locale: es })}</p>
            </div>
            {task.processingStartedAt && (
              <div>
                <h3 className="text-sm font-medium text-gray-500 mb-2">Inicio</h3>
                <p className="text-sm">{format(new Date(task.processingStartedAt), 'dd MMM yyyy, HH:mm', { locale: es })}</p>
              </div>
            )}
          </div>
          
          {task.model && (
            <div>
              <h3 className="text-sm font-medium text-gray-500 mb-2">Modelo</h3>
              <p className="text-sm font-mono bg-gray-100 px-3 py-1 rounded inline-block">{task.model}</p>
            </div>
          )}
          
          {task.error && (
            <div>
              <h3 className="text-sm font-medium text-red-600 mb-2">Error</h3>
              <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <p className="text-sm text-red-800 whitespace-pre-wrap">{task.error}</p>
              </div>
            </div>
          )}
          
          {task.executionResult && (
            <div>
              <h3 className="text-sm font-medium text-gray-500 mb-2">Resultado</h3>
              <div className={cn(
                "border rounded-lg p-4",
                task.executionResult.success ? "bg-green-50 border-green-200" : "bg-red-50 border-red-200"
              )}>
                <p className="text-sm font-medium mb-2">
                  {task.executionResult.success ? '✓ Éxito' : '✗ Error'}
                </p>
                {task.executionResult.commitSha && (
                  <p className="text-xs font-mono text-gray-600 mb-1">
                    Commit: {task.executionResult.commitSha}
                  </p>
                )}
                {task.executionResult.branch && (
                  <p className="text-xs text-gray-600 mb-1">
                    Branch: {task.executionResult.branch}
                  </p>
                )}
                {task.executionResult.commitUrl && (
                  <a
                    href={task.executionResult.commitUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm text-blue-600 hover:text-blue-800 underline inline-flex items-center gap-1"
                  >
                    Ver commit
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                    </svg>
                  </a>
                )}
              </div>
            </div>
          )}
          
          {task.streamingContent && (
            <div>
              <h3 className="text-sm font-medium text-gray-500 mb-2">Contenido</h3>
              <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 max-h-96 overflow-y-auto">
                <pre className="text-xs whitespace-pre-wrap font-mono">
                  {typeof task.streamingContent === 'string' 
                    ? task.streamingContent 
                    : JSON.stringify(task.streamingContent, null, 2)}
                </pre>
              </div>
            </div>
          )}
          
          <div className="flex gap-3 pt-4 border-t border-gray-200">
            <a
              href={`/tasks/${task.id}`}
              target="_blank"
              rel="noopener noreferrer"
              className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-center text-sm font-medium"
            >
              Abrir en nueva pestaña
            </a>
            <button
              onClick={() => {
                if (confirm('¿Eliminar esta tarea?')) {
                  onDelete(task.id);
                  onClose();
                }
              }}
              className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors text-sm font-medium"
            >
              Eliminar
            </button>
          </div>
        </div>
      </motion.div>
    </>
  );
}

