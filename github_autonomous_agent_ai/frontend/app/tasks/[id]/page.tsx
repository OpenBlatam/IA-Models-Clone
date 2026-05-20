'use client';

import { useState, useEffect, useCallback } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import { toast } from 'sonner';
import { Task } from '../../types/task';
import { tasksAPI } from '../../lib/tasks-api';
import { getStatusBadgeColor, getStatusLabel, calculateElapsedTime } from '../../utils/task-helpers';

export default function TaskDetailPage() {
  const params = useParams();
  const router = useRouter();
  const taskId = params.id as string;
  const [task, setTask] = useState<Task | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [elapsedTime, setElapsedTime] = useState<string>('00:00:00');

  const fetchTask = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const fetchedTask = await tasksAPI.getTask(taskId);
      if (!fetchedTask) {
        setTask(null);
        setError('Tarea no encontrada');
      } else {
        setTask(fetchedTask);
      }
    } catch (e: any) {
      console.error('Error loading task:', e);
      setError(e?.message || 'No se pudo cargar la tarea');
    } finally {
      setLoading(false);
    }
  }, [taskId]);

  useEffect(() => {
    fetchTask();
  }, [fetchTask]);

  // Actualizar la tarea periódicamente si está procesando
  useEffect(() => {
    if (task && (task.status === 'processing' || task.status === 'pending')) {
      const interval = setInterval(() => {
        fetchTask();
      }, 4000); // Actualizar cada 4 segundos

      return () => clearInterval(interval);
    }
  }, [task?.status, fetchTask]);

  // Cronómetro para tareas en procesamiento
  useEffect(() => {
    if (!task || !task.processingStartedAt || task.status !== 'processing') {
      setElapsedTime('00:00:00');
      return;
    }

    const interval = setInterval(() => {
      const elapsed = calculateElapsedTime(task);
      setElapsedTime(elapsed || '00:00:00');
    }, 1000); // Actualizar cada segundo

    return () => clearInterval(interval);
  }, [task]);

  // Detener el procesamiento de la tarea
  const handleStopTask = async () => {
    if (!task) {
      return;
    }

    if (confirm('¿Estás seguro de que quieres detener el procesamiento de esta tarea?')) {
      try {
        await tasksAPI.stopTask(taskId);
        toast.info('Procesamiento detenido');
        await fetchTask();
      } catch (e) {
        console.error('Error stopping task:', e);
        toast.error('No se pudo detener la tarea');
      }
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-white flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  if (!task) {
    return (
      <div className="min-h-screen bg-white">
        <div className="max-w-4xl mx-auto px-6 py-12">
          <div className="text-center">
            <h1 className="text-2xl font-bold mb-4">Tarea no encontrada</h1>
            <p className="text-gray-600 mb-6">
              {error || 'La tarea que buscas no existe o ha sido eliminada.'}
            </p>
            <Link
              href="/agent-control"
              className="inline-block px-6 py-3 bg-black text-white rounded-lg hover:bg-gray-800 transition-colors"
            >
              Volver a Agent Control
            </Link>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-white text-black">
      {/* Header */}
      <header className="relative z-10 bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-5 md:px-6 lg:px-8 py-3.5 md:py-4">
          <div className="flex items-center justify-between">
            <Link href="/" className="flex items-center gap-2.5 text-base md:text-lg text-black hover:opacity-80 transition-opacity">
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
                  <path d="M7 20L12 4L17 20H14.5L12 12.5L9.5 20H7Z" fill="url(#gradient-header)"/>
                </svg>
              </div>
              <span className="font-normal">
                <span className="font-light">GitHub</span> <span className="font-normal">Autonomous Agent AI</span>
              </span>
            </Link>
            
            <nav className="flex items-center gap-4">
              <Link href="/agent-control" className="text-black hover:opacity-70 transition-opacity font-normal text-sm">
                Agent Control
              </Link>
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-5xl mx-auto px-5 md:px-6 lg:px-8 py-10 md:py-12">
        <div className="mb-6">
          <Link
            href="/agent-control"
            className="inline-flex items-center gap-2 text-sm text-gray-600 hover:text-black transition-colors"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
            Volver a Agent Control
          </Link>
        </div>

        <div className="space-y-6">
          {/* Header de la tarea */}
          <div className="border-b border-gray-200 pb-6">
            <div className="flex items-start justify-between mb-4">
              <div>
                <h1 className="text-3xl font-bold mb-2">Detalles de la Tarea</h1>
                <p className="text-sm text-gray-500">
                  ID: <span className="font-mono">{task.id}</span>
                </p>
              </div>
              <div className="flex items-center gap-3">
                {task.status === 'processing' && task.processingStartedAt && (
                  <div className="text-right">
                    <div className="text-xs text-gray-500 mb-1">Tiempo transcurrido</div>
                    <div className="text-lg font-mono font-bold text-blue-600">{elapsedTime}</div>
                  </div>
                )}
                <span className={`px-4 py-2 rounded-lg text-sm font-medium ${getStatusBadgeColor(task.status)}`}>
                  {getStatusLabel(task.status)}
                </span>
                {(task.status === 'processing' || task.status === 'pending') && (
                  <button
                    onClick={handleStopTask}
                    className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors flex items-center gap-2"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
                    </svg>
                    Detener
                  </button>
                )}
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-gray-600 mb-1">Repositorio</p>
                <p className="font-medium">{task.repository}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600 mb-1">Fecha de creación</p>
                <p className="font-medium">{new Date(task.createdAt).toLocaleString('es-ES')}</p>
              </div>
            </div>
          </div>

          {/* Instrucción */}
          <div className="bg-gray-50 rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-3">Instrucción</h2>
            <p className="text-gray-700 whitespace-pre-wrap">{task.instruction}</p>
          </div>

          {/* Error si existe */}
          {task.error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-6">
              <h2 className="text-xl font-semibold mb-3 text-red-800">Error</h2>
              <p className="text-red-700 whitespace-pre-wrap">{task.error}</p>
            </div>
          )}

          {/* Resultados si la tarea está completada */}
          {task.status === 'completed' && task.result && (
            <div className="space-y-6">
              {/* Resultado principal */}
              {task.result.content && (
                <div className="bg-green-50 border border-green-200 rounded-lg p-6">
                  <h2 className="text-xl font-semibold mb-3 text-green-800">Resultado</h2>
                  <div className="prose max-w-none">
                    <p className="text-green-700 whitespace-pre-wrap">
                      {typeof task.result.content === 'string' 
                        ? task.result.content 
                        : JSON.stringify(task.result.content, null, 2)}
                    </p>
                  </div>
                </div>
              )}

              {/* Plan de acción */}
              {task.result.plan && (
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
                  <h2 className="text-xl font-semibold mb-4 text-blue-800">Plan de Acción</h2>
                  
                  {task.result.plan.steps && Array.isArray(task.result.plan.steps) && task.result.plan.steps.length > 0 && (
                    <div className="mb-4">
                      <h3 className="text-sm font-semibold text-blue-800 mb-2">Pasos a seguir:</h3>
                      <ol className="list-decimal list-inside space-y-2 text-blue-700">
                        {task.result.plan.steps.map((step: any, idx: number) => (
                          <li key={idx} className="pl-2">
                            {typeof step === 'string' ? step : JSON.stringify(step)}
                          </li>
                        ))}
                      </ol>
                    </div>
                  )}

                  {task.result.plan.files_to_create && (
                    <div className="mb-4">
                      <h3 className="text-sm font-semibold text-blue-800 mb-2">Archivos a crear:</h3>
                      {Array.isArray(task.result.plan.files_to_create) && task.result.plan.files_to_create.length > 0 ? (
                        <ul className="list-disc list-inside space-y-1 text-blue-700">
                          {task.result.plan.files_to_create.map((file: any, idx: number) => (
                            <li key={idx} className="pl-2">
                              {typeof file === 'string' ? file : JSON.stringify(file)}
                            </li>
                          ))}
                        </ul>
                      ) : typeof task.result.plan.files_to_create === 'object' && task.result.plan.files_to_create !== null ? (
                        <ul className="list-disc list-inside space-y-1 text-blue-700">
                          {Object.keys(task.result.plan.files_to_create).map((file: string, idx: number) => (
                            <li key={idx} className="pl-2">{file}</li>
                          ))}
                        </ul>
                      ) : null}
                    </div>
                  )}

                  {task.result.plan.files_to_modify && (
                    <div>
                      <h3 className="text-sm font-semibold text-blue-800 mb-2">Archivos a modificar:</h3>
                      {Array.isArray(task.result.plan.files_to_modify) && task.result.plan.files_to_modify.length > 0 ? (
                        <ul className="list-disc list-inside space-y-1 text-blue-700">
                          {task.result.plan.files_to_modify.map((file: any, idx: number) => (
                            <li key={idx} className="pl-2">
                              {typeof file === 'string' ? file : JSON.stringify(file)}
                            </li>
                          ))}
                        </ul>
                      ) : typeof task.result.plan.files_to_modify === 'object' && task.result.plan.files_to_modify !== null ? (
                        <ul className="list-disc list-inside space-y-1 text-blue-700">
                          {Object.keys(task.result.plan.files_to_modify).map((file: string, idx: number) => (
                            <li key={idx} className="pl-2">{file}</li>
                          ))}
                        </ul>
                      ) : null}
                    </div>
                  )}
                </div>
              )}

              {/* Código generado */}
              {task.result.code && (
                <div className="bg-gray-50 border border-gray-200 rounded-lg p-6">
                  <h2 className="text-xl font-semibold mb-3 text-gray-800">Código Generado</h2>
                  <div className="bg-black rounded-lg p-4 overflow-x-auto">
                    <pre className="text-sm text-green-400 font-mono">
                      <code>
                        {typeof task.result.code === 'string' 
                          ? task.result.code 
                          : JSON.stringify(task.result.code, null, 2)}
                      </code>
                    </pre>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Estado de procesamiento */}
          {task.status === 'processing' && (
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-yellow-600 mx-auto mb-4"></div>
              <p className="text-yellow-800 font-medium">La tarea se está procesando...</p>
              {task.processingStartedAt && (
                <p className="text-lg font-mono font-bold text-yellow-700 mt-2">
                  Tiempo: {elapsedTime}
                </p>
              )}
              <p className="text-sm text-yellow-600 mt-2">Esta página se actualiza automáticamente</p>
            </div>
          )}

          {/* Estado detenido */}
          {task.status === 'stopped' && (
            <div className="bg-orange-50 border border-orange-200 rounded-lg p-6 text-center">
              <p className="text-orange-800 font-medium">⏹️ Procesamiento detenido</p>
              <p className="text-sm text-orange-600 mt-2">El procesamiento fue detenido por el usuario</p>
            </div>
          )}

          {/* Estado pendiente */}
          {task.status === 'pending' && (
            <div className="bg-gray-50 border border-gray-200 rounded-lg p-6 text-center">
              <p className="text-gray-800 font-medium">La tarea está pendiente de procesamiento</p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

