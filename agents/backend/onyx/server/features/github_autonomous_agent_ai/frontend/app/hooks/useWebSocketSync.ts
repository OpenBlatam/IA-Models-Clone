/**
 * Hook para sincronización en tiempo real con backend usando WebSocket.
 */

import { useEffect, useRef, useCallback } from 'react';
import { useTaskStore } from '../store/task-store';
import { getWebSocketClient } from '../lib/api-client';
import { Task } from '../types/task';

interface UseWebSocketSyncOptions {
  enabled?: boolean;
  onTaskUpdate?: (task: Task) => void;
  onTaskCreated?: (task: Task) => void;
  onTaskCompleted?: (task: Task) => void;
  onTaskFailed?: (task: Task) => void;
  onAgentStatusChange?: (status: any) => void;
}

/**
 * Hook para sincronizar tareas con backend usando WebSocket.
 */
export function useWebSocketSync(options: UseWebSocketSyncOptions = {}) {
  const {
    enabled = true,
    onTaskUpdate,
    onTaskCreated,
    onTaskCompleted,
    onTaskFailed,
    onAgentStatusChange
  } = options;

  const { updateTask, addTask } = useTaskStore();
  const wsRef = useRef<ReturnType<typeof getWebSocketClient> | null>(null);
  const connectedRef = useRef(false);

  const connect = useCallback(async () => {
    if (!enabled || connectedRef.current) return;

    // Verificar si estamos en el cliente (no en SSR)
    if (typeof window === 'undefined') {
      console.warn('⚠️ WebSocket: No disponible en servidor (SSR)');
      return;
    }

    try {
      const ws = getWebSocketClient();
      if (!ws) {
        throw new Error('No se pudo crear el cliente WebSocket');
      }
      wsRef.current = ws;

      // Configurar listeners para eventos de tareas
      ws.on('task_update', (data: any) => {
        try {
          const task = data as Task;
          console.log('📡 WebSocket: task_update recibido', task.id, task.status);
          updateTask(task.id, task);
          onTaskUpdate?.(task);
        } catch (error) {
          console.error('Error procesando task_update:', error);
        }
      });

      ws.on('task_created', (data: any) => {
        try {
          const task = data as Task;
          console.log('📡 WebSocket: task_created recibido', task.id);
          addTask(task);
          onTaskCreated?.(task);
        } catch (error) {
          console.error('Error procesando task_created:', error);
        }
      });

      ws.on('task_completed', (data: any) => {
        try {
          const task = data as Task;
          console.log('📡 WebSocket: task_completed recibido', task.id);
          updateTask(task.id, { ...task, status: 'completed' });
          onTaskCompleted?.(task);
        } catch (error) {
          console.error('Error procesando task_completed:', error);
        }
      });

      ws.on('task_failed', (data: any) => {
        try {
          const task = data as Task;
          console.log('📡 WebSocket: task_failed recibido', task.id);
          updateTask(task.id, { ...task, status: 'failed' });
          onTaskFailed?.(task);
        } catch (error) {
          console.error('Error procesando task_failed:', error);
        }
      });

      ws.on('agent_status', (data: any) => {
        try {
          onAgentStatusChange?.(data);
        } catch (error) {
          console.error('Error procesando agent_status:', error);
        }
      });

      // Manejar errores de conexión
      ws.on('error', (error: any) => {
        console.error('❌ WebSocket error:', error);
        connectedRef.current = false;
      });

      // Manejar desconexión y reconectar
      ws.on('close', () => {
        console.warn('⚠️ WebSocket desconectado, intentando reconectar...');
        connectedRef.current = false;
        // Intentar reconectar después de 3 segundos
        setTimeout(() => {
          if (enabled && !connectedRef.current) {
            connect();
          }
        }, 3000);
      });

      // Conectar con timeout
      try {
        await Promise.race([
          ws.connect(),
          new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Timeout al conectar WebSocket después de 10 segundos')), 10000)
          )
        ]);
        connectedRef.current = true;
        console.log('✅ WebSocket conectado para eventos en tiempo real');
      } catch (connectError) {
        // Si falla la conexión, lanzar el error para que sea manejado por el catch externo
        throw connectError;
      }
    } catch (error) {
      // Extraer toda la información posible del error de forma segura
      let errorMessage = 'Error desconocido al conectar WebSocket';
      let errorStack: string | undefined;
      let errorDetails: Record<string, any> = {};
      
      try {
        if (error instanceof Error) {
          errorMessage = error.message || 'Error sin mensaje';
          errorStack = error.stack;
          errorDetails = {
            name: error.name || 'Error',
            message: error.message || '',
            stack: error.stack || '',
          };
          
          // Extraer propiedades adicionales si existen
          if ((error as any).url) errorDetails.url = (error as any).url;
          if ((error as any).readyState !== undefined) errorDetails.readyState = (error as any).readyState;
        } else if (typeof error === 'string') {
          errorMessage = error;
          errorDetails = { message: error };
        } else if (error && typeof error === 'object') {
          // Manejar Event objects y otros objetos de error
          const errorObj = error as any;
          
          // Intentar extraer información de objetos de error
          errorMessage = errorObj.message || 
                        errorObj.error || 
                        errorObj.type || 
                        errorObj.toString?.() ||
                        'Error de objeto desconocido';
          
          // Si es un Event object, extraer información útil
          if (errorObj.type === 'error' || errorObj.target) {
            errorMessage = `WebSocket error event: ${errorObj.type || 'unknown'}`;
            if (errorObj.target) {
              const target = errorObj.target as WebSocket;
              errorDetails = {
                type: errorObj.type,
                readyState: target?.readyState,
                url: target?.url,
                code: target?.CONNECTING !== undefined ? 'WebSocket' : undefined,
              };
            }
          } else {
            // Intentar copiar propiedades del objeto de forma segura
            try {
              errorDetails = Object.keys(errorObj).reduce((acc, key) => {
                try {
                  const value = errorObj[key];
                  // Solo incluir valores serializables
                  if (value !== undefined && value !== null) {
                    if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
                      acc[key] = value;
                    } else if (typeof value === 'object' && value.constructor === Object) {
                      acc[key] = 'object';
                    }
                  }
                } catch (e) {
                  // Ignorar errores al acceder a propiedades
                }
                return acc;
              }, {} as Record<string, any>);
            } catch (e) {
              errorDetails = { _error: 'No se pudieron extraer propiedades' };
            }
          }
        } else {
          errorMessage = String(error) || 'Error desconocido';
          errorDetails = { value: String(error) };
        }
      } catch (parseError) {
        // Si falla todo, al menos mostrar algo
        errorMessage = 'Error al procesar el error del WebSocket';
        errorDetails = { 
          originalError: String(error),
          parseError: String(parseError)
        };
      }
      
      // Log del error con información estructurada (siempre mostrar algo útil)
      const logData: Record<string, any> = {
        message: errorMessage,
      };
      
      if (errorStack) {
        logData.stack = errorStack;
      }
      
      if (Object.keys(errorDetails).length > 0) {
        logData.details = errorDetails;
      }
      
      // Solo agregar type y constructor si son útiles
      if (error !== null && error !== undefined) {
        logData.errorType = typeof error;
        if (error.constructor && error.constructor.name) {
          logData.constructor = error.constructor.name;
        }
      }
      
      console.error('❌ Error conectando WebSocket:', logData);
      
      // Log adicional para debugging si es un Error
      if (error instanceof Error) {
        console.error('Error completo:', {
          name: error.name,
          message: error.message,
          stack: error.stack,
        });
      }
      
      connectedRef.current = false;
      
      // Intentar reconectar después de 5 segundos si falla (solo en cliente)
      if (typeof window !== 'undefined') {
        setTimeout(() => {
          if (enabled && !connectedRef.current) {
            console.log('🔄 Intentando reconectar WebSocket...');
            connect();
          }
        }, 5000);
      }
    }
  }, [enabled, updateTask, addTask, onTaskUpdate, onTaskCreated, onTaskCompleted, onTaskFailed, onAgentStatusChange]);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.disconnect();
      wsRef.current = null;
      connectedRef.current = false;
      console.log('🔌 WebSocket desconectado');
    }
  }, []);

  useEffect(() => {
    if (enabled && typeof window !== 'undefined') {
      // Intentar conectar, pero no bloquear si falla
      connect().catch((error) => {
        // El error ya se maneja en connect(), solo loguear aquí para debugging
        console.warn('⚠️ WebSocket: No se pudo conectar (modo standalone activo)', error);
      });
    }

    return () => {
      disconnect();
    };
  }, [enabled, connect, disconnect]);

  return {
    connected: connectedRef.current,
    connect,
    disconnect
  };
}



