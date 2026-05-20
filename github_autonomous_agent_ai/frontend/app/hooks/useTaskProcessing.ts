import { useState, useCallback } from 'react';
import { toast } from 'sonner';
import { Task } from '../types/task';
import { GitHubRepository } from '../lib/github-api';
import { deepseekAPI } from '../lib/deepseek-api';
import { openrouterAPI } from '../lib/openrouter-api';
import { AIModel, getModelConfig } from '../lib/ai-providers';
import { canResumeTask } from '../utils/task-helpers';
import { githubExecutor } from '../lib/github-executor';
import { tasksAPI } from '../lib/tasks-api';

interface UseTaskProcessingProps {
  updateTask: (taskId: string, updates: Partial<Task>) => void;
  tasks?: Task[]; // Agregar tasks para verificar duplicados
}

/**
 * Hook personalizado para manejar el procesamiento de tareas con múltiples proveedores de IA
 */
export function useTaskProcessing({ updateTask, tasks = [] }: UseTaskProcessingProps) {
  const [abortControllers, setAbortControllers] = useState<Map<string, AbortController>>(new Map());
  const [processingTasks, setProcessingTasks] = useState<Set<string>>(new Set());

  // Procesar tarea con el modelo seleccionado usando streaming
  // Ahora delega al backend para ejecución persistente
  const processTask = useCallback(async (
    taskId: string,
    instruction: string,
    repository: string,
    repoInfo: GitHubRepository,
    model: AIModel = 'deepseek-chat',
    forceResume: boolean = false // Permite forzar reanudación incluso si está en 'processing'
  ) => {
    // Verificar si ya está siendo procesada localmente
    if (processingTasks.has(taskId) && !forceResume) {
      console.log('⏸️ Tarea ya está siendo procesada localmente, ignorando:', taskId);
      return;
    }

    // Verificar el estado actual de la tarea
    const currentTask = tasks.find(t => t.id === taskId);
    // Si forceResume es true, permitir reiniciar incluso si está en 'processing'
    // Esto es útil cuando se recarga la página y la tarea está en 'processing' pero no está siendo procesada
    if (currentTask && (currentTask.status === 'processing' || currentTask.status === 'running') && !forceResume) {
      console.log('⏸️ Tarea ya está en estado de procesamiento, ignorando:', taskId);
      return;
    }
    
    if (forceResume && currentTask && (currentTask.status === 'processing' || currentTask.status === 'running')) {
      console.log('🔄 Forzando reanudación de tarea en procesamiento:', taskId);
    }

    console.log('🚀 Iniciando procesamiento de tarea (backend):', { taskId, instruction, repository });
    
    // Marcar como procesando localmente
    setProcessingTasks(prev => new Set(prev).add(taskId));
    
    try {
      // Actualizar estado a processing
      await updateTask(taskId, {
        status: 'processing',
        processingStartedAt: new Date().toISOString(),
        streamingContent: '',
        repoInfo: repoInfo,
        model: model,
      });

      // Iniciar procesamiento en el backend (se ejecutará en background)
      await tasksAPI.startProcessing(taskId);
      
      console.log('✅ Procesamiento iniciado en el backend');
    } catch (error: any) {
      console.error('❌ Error iniciando procesamiento:', error);
      // Remover de procesamiento en caso de error
      setProcessingTasks(prev => {
        const next = new Set(prev);
        next.delete(taskId);
        return next;
      });
      await updateTask(taskId, {
        status: 'failed',
        error: error.message || 'Error al iniciar procesamiento',
      });
      throw error;
    }
  }, [updateTask, tasks, processingTasks]);

  // Detener procesamiento de una tarea
  const stopTask = useCallback(async (taskId: string) => {
    console.log('🛑 Deteniendo tarea manualmente:', taskId);
    
    try {
      // Detener en el backend
      await tasksAPI.stopTask(taskId);
      
      // Abortar el controller si existe (por si acaso hay procesamiento local)
      const controller = abortControllers.get(taskId);
      if (controller) {
        controller.abort();
        setAbortControllers(prev => {
          const newMap = new Map(prev);
          newMap.delete(taskId);
          return newMap;
        });
      }
      
      // Actualizar estado de la tarea a 'stopped'
      await updateTask(taskId, {
        status: 'stopped',
        error: 'Procesamiento detenido por el usuario',
      });
    } catch (error: any) {
      console.error('Error stopping task:', error);
      // Actualizar localmente aunque falle el backend
      await updateTask(taskId, {
        status: 'stopped',
        error: 'Procesamiento detenido por el usuario',
      });
    }
  }, [abortControllers, updateTask]);

  // Reanudar tareas que estaban procesando o pendientes
  // Ahora las tareas se procesan en el backend, así que solo verificamos si necesitan reanudarse
  const resumeTasks = useCallback((tasksToResume: Task[]) => {
    tasksToResume.forEach((task) => {
      if (!task.repoInfo) return;
      
      // Verificar si ya está siendo procesada
      if (processingTasks.has(task.id)) {
        console.log('⏸️ Tarea ya está siendo procesada (local), ignorando:', task.id);
        return;
      }
      
      // Si está pendiente o detenida, iniciar procesamiento en el backend
      if ((task.status === 'pending' || task.status === 'stopped') && canResumeTask(task)) {
        console.log(`🔄 Reanudando tarea ${task.status} en backend:`, task.id);
        // No usar setTimeout, procesar inmediatamente pero de forma asíncrona
        processTask(
          task.id,
          task.instruction,
          task.repository,
          task.repoInfo!,
          (task.model as any) || 'deepseek-chat'
        ).catch((error) => {
          console.error('Error al reanudar tarea:', error);
        });
      } else if (task.status === 'processing' || task.status === 'running') {
        // Si está procesando, verificar que el backend la esté procesando
        // Si no está en el archivo processing.json, reiniciar el procesamiento
        console.log(`🔄 Verificando/reanudando tarea en procesamiento: ${task.id}`);
        
        // IMPORTANTE: Usar forceResume=true para permitir reiniciar tareas en 'processing'
        // Esto asegura que al recargar la página, las tareas continúen procesando
        processTask(
          task.id,
          task.instruction,
          task.repository,
          task.repoInfo!,
          (task.model as any) || 'deepseek-chat',
          true // forceResume: permite reiniciar incluso si está en 'processing'
        ).catch((error) => {
          console.error('Error al verificar/reanudar tarea en procesamiento:', error);
          // No marcar como error si ya está procesando en el backend
        });
      }
    });
  }, [processTask, abortControllers, processingTasks]);

  // Reanudar una tarea específica
  const resumeTask = useCallback((taskId: string, task: Task) => {
    if (!task.repoInfo) {
      console.error('❌ No se puede reanudar: falta información del repositorio');
      return;
    }

    if (processingTasks.has(taskId)) {
      console.log('⏸️ Tarea ya está siendo procesada');
      return;
    }

    if (!canResumeTask(task)) {
      console.log('⏸️ Tarea no puede ser reanudada');
      return;
    }

    console.log(`🔄 Reanudando tarea ${taskId}...`);
    processTask(
      taskId,
      task.instruction,
      task.repository,
      task.repoInfo,
      (task.model as any) || 'deepseek-chat'
    ).catch((error) => {
      console.error('Error al reanudar tarea:', error);
    });
  }, [processTask, processingTasks]);

  // Aprobar plan (sin ejecutar todavía)
  const approvePlan = useCallback((taskId: string, task: Task) => {
    console.log('✅ Plan aprobado por el usuario (sin ejecutar todavía)');
    if (!task.pendingApproval) {
      console.error('❌ No hay plan pendiente de aprobación');
      return;
    }
    
    // Marcar el plan como aprobado pero mantener el status en 'processing' para que el botón de parar siga visible
    updateTask(taskId, {
      status: 'processing', // Mantener en processing para que el botón de parar siga visible
      pendingApproval: {
        ...task.pendingApproval,
        approved: true, // Marcar como aprobado
      },
    });
    toast.success('Plan aprobado', {
      description: 'Presiona el botón "Pausa" para ejecutar el commit',
    });
  }, [updateTask]);

  // Ejecutar plan aprobado (cuando se presiona el botón de ejecutar)
  // Refactorizado para que NO SE DETENGA - proceso completamente asíncrono y resiliente
  const executePlan = useCallback(async (
    taskId: string,
    task: Task
  ) => {
    // Validaciones mínimas - continuar incluso si falta algo
    if (!task.pendingApproval) {
      console.warn('⚠️ No hay aprobación pendiente, continuando de todas formas');
    }

    if (!task.repoInfo) {
      console.error('❌ Falta información del repositorio - no se puede continuar');
      return;
    }

    // Aprobar plan automáticamente si no está aprobado (sin await para no bloquear)
    if (!task.pendingApproval?.approved) {
      console.log('✅ Aprobando plan automáticamente antes de ejecutar...');
      updateTask(taskId, {
        pendingApproval: {
          ...task.pendingApproval!,
          approved: true,
        },
      }).catch(err => {
        console.warn('⚠️ Error aprobando plan (continuando de todas formas):', err);
      });
    }

    const { actions, commitMessage } = task.pendingApproval || { actions: [], commitMessage: 'feat: Cambios automáticos' };

    // Validar que hay acciones
    if (!actions || actions.length === 0) {
      console.warn('⚠️ No hay acciones para ejecutar');
      return;
    }

    console.log('🚀 Ejecutando plan aprobado - proceso continuará sin detenerse...');
    
    // Actualizar estado sin bloquear
    updateTask(taskId, {
      status: 'processing',
      executionStatus: 'executing',
    }).catch(err => {
      console.warn('⚠️ Error actualizando estado (continuando de todas formas):', err);
    });

    // Ejecutar en background - NO bloquear, continuar inmediatamente
    // El proceso continuará sin importar qué pase
    githubExecutor.executeActions({
      repository: task.repository,
      branch: task.repoInfo.default_branch || 'main',
      actions: actions,
      commitMessage: commitMessage,
    })
      .then((executionResult) => {
        if (executionResult.success) {
          console.log('✅ Acciones ejecutadas exitosamente:', executionResult.commitSha);
          
          // Actualizar tarea en background sin bloquear
          updateTask(taskId, {
            status: 'pending_commit_approval',
            executionStatus: 'completed',
            executionResult: executionResult,
            pendingApproval: undefined,
            pendingCommitApproval: {
              commitSha: executionResult.commitSha,
              commitUrl: executionResult.commitUrl,
              commitMessage: commitMessage,
              branch: executionResult.branch,
            },
          }).catch(err => {
            console.warn('⚠️ Error actualizando tarea después del commit (commit ya ejecutado):', err);
          });
          
          console.log('⏳ Commit ejecutado, esperando aprobación del usuario');
          toast.success('Commit completado', {
            description: 'Revisa el commit y decide si aprobarlo o rechazarlo',
          });
        } else {
          console.error('❌ Error al ejecutar acciones:', executionResult.error);
          // Actualizar estado pero no detener el proceso
          updateTask(taskId, {
            status: 'failed',
            executionStatus: 'failed',
            executionResult: executionResult,
            error: executionResult.error || 'Error al ejecutar acciones',
          }).catch(err => {
            console.warn('⚠️ Error actualizando estado de error:', err);
          });
          
          toast.error('Error al aplicar cambios', {
            description: executionResult.error || 'No se pudieron aplicar los cambios en GitHub',
          });
        }
      })
      .catch((execError: any) => {
        // Capturar error pero no detener - el proceso puede continuar en el backend
        console.error('❌ Error al ejecutar acciones:', execError);
        
        // Intentar actualizar estado pero no bloquear
        updateTask(taskId, {
          status: 'failed',
          executionStatus: 'failed',
          executionResult: {
            success: false,
            error: execError.message || 'Error al ejecutar acciones',
          },
          error: execError.message || 'Error al ejecutar acciones',
        }).catch(err => {
          console.warn('⚠️ Error actualizando estado después de error:', err);
        });
        
        toast.error('Error al ejecutar acciones', {
          description: execError.message || 'Error inesperado - El proceso puede continuar en el backend',
        });
      });
      
    // Retornar inmediatamente - el proceso continúa en background sin detenerse
    console.log('🚀 Commit iniciado en background - proceso continuará sin detenerse');
  }, [updateTask]);

  // Rechazar cambios pendientes (plan)
  const rejectChanges = useCallback((taskId: string) => {
    console.log('❌ Plan rechazado por el usuario');
    updateTask(taskId, {
      status: 'completed',
      pendingApproval: undefined,
      error: 'Plan rechazado por el usuario',
    });
    toast.info('Plan rechazado', {
      description: 'El plan no se ejecutó',
    });
  }, [updateTask]);

  // Aprobar commit ejecutado
  const approveCommit = useCallback((taskId: string) => {
    console.log('✅ Commit aprobado por el usuario');
    updateTask(taskId, {
      status: 'completed',
      pendingCommitApproval: undefined,
    });
    toast.success('Commit aprobado', {
      description: 'El commit se ha aprobado exitosamente',
    });
  }, [updateTask]);

  // Rechazar commit ejecutado (revertir)
  const rejectCommit = useCallback((taskId: string) => {
    console.log('❌ Commit rechazado por el usuario');
    updateTask(taskId, {
      status: 'completed',
      pendingCommitApproval: undefined,
      error: 'Commit rechazado por el usuario',
    });
    toast.info('Commit rechazado', {
      description: 'El commit fue rechazado (nota: el commit ya está en GitHub, puedes revertirlo manualmente)',
    });
  }, [updateTask]);

  return {
    processTask,
    stopTask,
    resumeTask,
    resumeTasks,
    abortControllers,
    approvePlan,
    executePlan,
    rejectChanges,
    approveCommit,
    rejectCommit,
  };
}

