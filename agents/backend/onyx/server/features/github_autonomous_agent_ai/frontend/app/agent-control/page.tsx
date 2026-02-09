'use client';

import { useState, useEffect } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { toast } from 'sonner';
import GithubAuth from '../components/GithubAuth';
import RepositorySelector from '../components/RepositorySelector';
import FolderExplorer from '../components/FolderExplorer';
import ModelSelector from '../components/ModelSelector';
import { TaskCard } from '../components/tasks/TaskCard';
import { ApprovalModal } from '../components/ApprovalModal';
import { CommitApprovalModal } from '../components/CommitApprovalModal';
import { CommitPreviewModal } from '../components/CommitPreviewModal';
import { AgentsSection } from '../components/kanban/AgentsSection';
import { BackendSyncIndicator } from '../components/BackendSyncIndicator';
import { GitHubRepository, GitHubUser } from '../lib/github-api';
import { useTaskStore } from '../store/task-store';
import { useTaskProcessing } from '../hooks/useTaskProcessing';
import { useTaskResume } from '../hooks/useTaskResume';
import { getTasksByStatus } from '../utils/task-helpers';
import { createTaskSchema, type CreateTaskFormData } from '../lib/validations';
import { AIModel } from '../lib/ai-providers';
import { cn } from '../utils/cn';
import { Task } from '../types/task';
import { githubExecutor } from '../lib/github-executor';

export default function AgentControlPage() {
  const [selectedRepository, setSelectedRepository] = useState<GitHubRepository | null>(null);
  const [selectedFolder, setSelectedFolder] = useState<string>('');
  const [selectedModel, setSelectedModel] = useState<AIModel>('deepseek-chat');
  const [githubUser, setGithubUser] = useState<GitHubUser | null>(null);
  const [authError, setAuthError] = useState<string | null>(null);
  
  // Filtros y búsqueda
  const [taskSearchQuery, setTaskSearchQuery] = useState('');
  const [taskStatusFilter, setTaskStatusFilter] = useState<'all' | 'pending' | 'processing' | 'completed' | 'failed'>('all');
  const [taskSortBy, setTaskSortBy] = useState<'date' | 'status' | 'repository'>('date');
  const [taskSortOrder, setTaskSortOrder] = useState<'asc' | 'desc'>('desc');

  // React Hook Form
  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
    reset,
    setValue,
    watch,
  } = useForm<CreateTaskFormData>({
    resolver: zodResolver(createTaskSchema),
    defaultValues: {
      instruction: '',
      repository: '',
      folder: '',
    },
  });

  // Zustand store
  const {
    tasks,
    isLoading: tasksLoading,
    addTask,
    updateTask,
    deleteTask,
    deleteAllTasks,
    loadTasks,
  } = useTaskStore();

  const { processTask, stopTask, resumeTask, approvePlan, executePlan, rejectChanges, approveCommit, rejectCommit } = useTaskProcessing({ updateTask, tasks });
  
  // Estado para el modal de aprobación
  const [approvalTask, setApprovalTask] = useState<Task | null>(null);
  
  // Estado para el modal de preview de commit
  const [commitPreviewTask, setCommitPreviewTask] = useState<Task | null>(null);
  // Estado para rastrear tareas que se están comitando (para mostrar loading)
  const [committingTasks, setCommittingTasks] = useState<Set<string>>(new Set());
  
  // Cargar tareas al montar y sincronizar periódicamente
  // Cargar tareas al montar
  useEffect(() => {
    loadTasks(true);
  }, []); // Solo una vez al montar

  // Sincronización para tareas en procesamiento
  useEffect(() => {
    const hasProcessingTasks = tasks.some(t => 
      t.status === 'processing' || t.status === 'running' || t.status === 'pending_approval'
    );
    
    if (!hasProcessingTasks) {
      return; // No hay tareas procesando, no hacer nada
    }
    
    // Sincronizar cada 10 segundos solo si hay tareas procesando (reducido de 5 a 10)
    const syncInterval = setInterval(() => {
      loadTasks();
    }, 10000);

    return () => {
      clearInterval(syncInterval);
    };
    // Remover loadTasks de las dependencias para evitar recrear el intervalo constantemente
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tasks]);

  // Sincronizar cuando la pestaña vuelve a estar visible
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (!document.hidden) {
        loadTasks();
      }
    };
    document.addEventListener('visibilitychange', handleVisibilityChange);

    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [loadTasks]);

  // Reanudar tareas automáticamente (ahora desde el backend)
  useTaskResume();

  // Atajos de teclado
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      // Ctrl/Cmd + K para enfocar búsqueda
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        const searchInput = document.querySelector('input[placeholder*="Buscar"]') as HTMLInputElement;
        if (searchInput) {
          searchInput.focus();
        }
      }
      // Escape para limpiar filtros
      if (e.key === 'Escape' && (taskSearchQuery || taskStatusFilter !== 'all')) {
        setTaskSearchQuery('');
        setTaskStatusFilter('all');
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [taskSearchQuery, taskStatusFilter]);

  // Detectar tareas pendientes de aprobación del PLAN y mostrar modal
  // Solo mostrar si el plan NO está aprobado todavía
  useEffect(() => {
    const pendingApprovalTask = tasks.find(t => 
      t.status === 'pending_approval' && 
      t.pendingApproval && 
      !t.pendingApproval.approved // Solo si no está aprobado todavía
    );
    if (pendingApprovalTask && !approvalTask) {
      setApprovalTask(pendingApprovalTask);
    } else if (!pendingApprovalTask && approvalTask) {
      setApprovalTask(null);
    }
  }, [tasks, approvalTask]);

  // Detectar tareas pendientes de aprobación del COMMIT
  const [commitApprovalTask, setCommitApprovalTask] = useState<Task | null>(null);
  useEffect(() => {
    const pendingCommitTask = tasks.find(t => t.status === 'pending_commit_approval' && t.pendingCommitApproval);
    if (pendingCommitTask && !commitApprovalTask) {
      setCommitApprovalTask(pendingCommitTask);
    } else if (!pendingCommitTask && commitApprovalTask) {
      setCommitApprovalTask(null);
    }
  }, [tasks, commitApprovalTask]);

  // Handlers para aprobación del PLAN
  const handleApprove = () => {
    if (approvalTask) {
      approvePlan(approvalTask.id, approvalTask);
      setApprovalTask(null);
    }
  };

  const handleReject = () => {
    if (approvalTask) {
      rejectChanges(approvalTask.id);
      setApprovalTask(null);
    }
  };

  // Handlers para aprobación del COMMIT
  const handleApproveCommit = () => {
    if (commitApprovalTask) {
      approveCommit(commitApprovalTask.id);
      setCommitApprovalTask(null);
    }
  };

  const handleRejectCommit = () => {
    if (commitApprovalTask) {
      rejectCommit(commitApprovalTask.id);
      setCommitApprovalTask(null);
    }
  };

  // Handlers
  const handleAuthSuccess = (user: GitHubUser) => {
    setGithubUser(user);
    setAuthError(null);
  };

  const handleAuthError = (error: string) => {
    setAuthError(error);
  };

  const handleRepositorySelect = (repo: GitHubRepository) => {
    setSelectedRepository(repo);
    setSelectedFolder('');
    setValue('repository', repo.full_name);
    setValue('folder', '');
  };

  const handleFolderSelect = (folderPath: string) => {
    setSelectedFolder(folderPath);
    setValue('folder', folderPath);
  };

  const onSubmit = async (data: CreateTaskFormData) => {
    console.log('📝 Formulario enviado:', { 
      instructionLength: data.instruction?.length || 0,
      instructionPreview: data.instruction?.substring(0, 100) || '',
      selectedRepository: selectedRepository?.full_name,
      selectedModel 
    });
    
    if (!selectedRepository) {
      console.error('❌ No hay repositorio seleccionado');
      toast.error('Por favor selecciona un repositorio');
      return;
    }

    // Validar y limpiar la instrucción
    const cleanInstruction = data.instruction?.trim() || '';
    
    if (!cleanInstruction || cleanInstruction.length < 10) {
      console.error('❌ Instrucción inválida:', {
        length: cleanInstruction.length,
        preview: cleanInstruction.substring(0, 50),
      });
      toast.error('La instrucción debe tener al menos 10 caracteres');
      return;
    }
    
    // Verificar que la instrucción no sea solo logs de consola
    if (cleanInstruction.includes('task-store.ts:') || 
        cleanInstruction.includes('report-hmr-latency.ts:') ||
        cleanInstruction.includes('[Fast Refresh]') ||
        cleanInstruction.includes('📥 Tareas cargadas')) {
      console.error('❌ La instrucción parece contener logs de consola');
      toast.error('Por favor escribe una instrucción válida, no copies logs de la consola');
      return;
    }

    try {
      console.log('✅ Creando nueva tarea...');
      
      // Crear nueva tarea con instrucción limpia
      const newTask = await addTask({
        repository: selectedRepository.full_name,
        instruction: cleanInstruction,
        status: 'pending',
        repoInfo: selectedRepository,
        model: selectedModel,
      });

      console.log('✅ Tarea creada:', newTask);

      if (!newTask || !newTask.id) {
        console.error('❌ Error: La tarea no se creó correctamente');
        toast.error('Error al crear la tarea', {
          description: 'No se pudo crear la tarea. Por favor, intenta de nuevo.',
        });
        return;
      }

      toast.success('Tarea creada exitosamente', {
        description: `Procesando: ${data.instruction.substring(0, 50)}...`,
      });

      // Resetear el formulario (solo la instrucción)
      reset({
        instruction: '',
        repository: '', // Se establecerá después si hay repositorio seleccionado
        folder: '',
      });

      // Restablecer el valor del repositorio si hay uno seleccionado
      // Esto es necesario porque el formulario valida el campo 'repository'
      if (selectedRepository) {
        setValue('repository', selectedRepository.full_name);
      }
      if (selectedFolder) {
        setValue('folder', selectedFolder);
      }

      // Procesar automáticamente con el modelo seleccionado (ahora en el backend)
      console.log('🚀 Iniciando procesamiento de tarea en backend...');
      setTimeout(async () => {
        try {
          await processTask(
            newTask.id,
            newTask.instruction,
            newTask.repository,
            selectedRepository,
            selectedModel
          );
        } catch (error: any) {
          console.error('❌ Error no capturado en processTask:', error);
          await updateTask(newTask.id, {
            status: 'failed',
            error: error.message || 'Error inesperado al procesar la tarea',
          });
          toast.error('Error al procesar la tarea', {
            description: error.message || 'Error inesperado',
          });
        }
      }, 100);
    } catch (error: any) {
      console.error('❌ Error creating task:', error);
      console.error('Error stack:', error.stack);
      toast.error('Error al crear la tarea', {
        description: error.message || 'Error inesperado',
      });
    }
  };

  const handleDeleteTask = (taskId: string) => {
    deleteTask(taskId);
    toast.success('Tarea eliminada');
  };

  const handleDeleteAllTasks = () => {
    deleteAllTasks();
    toast.success('Todas las tareas han sido eliminadas');
  };

  const handleDeleteUnprocessedTasks = () => {
    const unprocessedTasks = tasks.filter(
      task => task.status === 'pending' || task.status === 'processing' || task.status === 'failed'
    );
    
    if (unprocessedTasks.length === 0) {
      toast.info('No hay tareas sin procesar para eliminar');
      return;
    }

    const unprocessedIds = unprocessedTasks.map(t => t.id);
    unprocessedIds.forEach(id => deleteTask(id));
    toast.success(`${unprocessedTasks.length} tarea(s) eliminada(s)`);
  };

  const handleStopTask = async (taskId: string) => {
    const task = tasks.find(t => t.id === taskId);
    
    // Si el plan está aprobado pero no ejecutado, ejecutar el commit
    if (task && task.pendingApproval?.approved && task.status === 'processing') {
      console.log('🚀 Ejecutando commit desde botón de pausa...');
      await executePlan(taskId, task);
      toast.info('Ejecutando commit...');
    } else if (task && task.status === 'pending_approval' && task.pendingApproval) {
      // Si el plan está listo pero no aprobado, también ejecutar (por si acaso)
      console.log('🚀 Ejecutando plan desde botón de parar...');
      await executePlan(taskId, task);
      toast.info('Ejecutando plan...');
    } else {
      // Durante el procesamiento normal, parar normalmente
      stopTask(taskId);
      toast.info('Tarea detenida');
    }
  };

  const handleResumeTask = (taskId: string) => {
    const task = tasks.find(t => t.id === taskId);
    if (task) {
      resumeTask(taskId, task);
      toast.info('Tarea reanudada');
    }
  };

  const handleCommitClick = async (taskId: string) => {
    const task = tasks.find(t => t.id === taskId);
    if (!task) return;
    
    // Verificar si hay plan o contenido para mostrar
    const hasPlan = task.pendingApproval && task.pendingApproval.actions && task.pendingApproval.actions.length > 0;
    const hasContent = task.streamingContent && 
      (typeof task.streamingContent === 'string' 
        ? task.streamingContent.length > 0 
        : true);
    
    if (!hasPlan && !hasContent) {
      toast.warning('No hay contenido para comitar', {
        description: 'La tarea no tiene contenido generado ni plan estructurado.',
      });
      return;
    }
    
    // Abrir el modal para mostrar qué se va a comitar
    setCommitPreviewTask(task);
  };

  const handleConfirmCommit = async () => {
    if (!commitPreviewTask) return;
    
    const taskId = commitPreviewTask.id;
    
    // Protección: evitar ejecutar commit múltiples veces
    if (committingTasks.has(taskId)) {
      console.warn('⚠️ Commit ya en proceso, ignorando solicitud duplicada');
      return;
    }
    
    const task = tasks.find(t => t.id === taskId) || commitPreviewTask;
    
    // Marcar como comitando - NO cerrar el modal hasta que termine completamente
    setCommittingTasks(prev => new Set(prev).add(taskId));
    
    // Función helper para limpiar estado y cerrar modal
    const cleanupAndClose = (success: boolean, message?: string) => {
      setCommittingTasks(prev => {
        const next = new Set(prev);
        next.delete(taskId);
        return next;
      });
      
      if (success && message) {
        toast.success(message);
      } else if (!success && message) {
        toast.error(message);
      }
      
      // Cerrar modal solo después de que todo termine
      setTimeout(() => {
        setCommitPreviewTask(null);
      }, 500);
    };

    // Ejecutar commit de forma completamente asíncrona sin bloquear
    // No usar await para que no se detenga el proceso
    (async () => {
      try {
        // Si hay plan con acciones, ejecutarlo
        if (task.pendingApproval?.actions && task.pendingApproval.actions.length > 0) {
          // Aprobar el plan si no está aprobado (sin await para no bloquear)
          if (!task.pendingApproval.approved) {
            approvePlan(taskId, task).catch(err => {
              console.warn('⚠️ Error aprobando plan (continuando de todas formas):', err);
            });
          }
          
          // Notificar inmediatamente sin esperar
          toast.info('Iniciando commit...', {
            description: 'El proceso continuará en background sin detenerse',
          });
          
          // Ejecutar plan en background - NO esperar, continuar inmediatamente
          // Usar setTimeout para asegurar que no bloquee el hilo principal
          setTimeout(() => {
            executePlan(taskId, { ...task, pendingApproval: { ...task.pendingApproval, approved: true } })
              .then(() => {
                // El proceso continúa en background, solo notificar y cerrar
                cleanupAndClose(true, 'Commit completado - Los cambios se han aplicado en GitHub');
              })
              .catch((error) => {
                // Incluso si hay error, el proceso puede continuar en el backend
                console.error('Error en executePlan:', error);
                cleanupAndClose(false, `Error al iniciar commit: ${error.message || 'Error desconocido'}`);
              });
          }, 0);
          
          return; // Salir inmediatamente, el proceso continúa en background
        }
        
        // Si no hay plan pero hay contenido, crear archivo .md (no JSON)
        const streamingContent = typeof task.streamingContent === 'string' 
          ? task.streamingContent 
          : task.streamingContent 
            ? (typeof task.streamingContent === 'object' 
                ? JSON.stringify(task.streamingContent, null, 2)
                : String(task.streamingContent))
            : '';
        
        if (streamingContent && streamingContent.length > 0) {
          // Usar ID de tarea para generar nombre consistente - evita múltiples archivos
          // El nombre se genera una vez y se mantiene estable
          const fileName = `generated-content-${taskId}.md`;
          // Formatear como markdown, no como JSON
          const markdownContent = streamingContent.startsWith('```json') || streamingContent.startsWith('{')
            ? streamingContent.replace(/^```json\s*/, '').replace(/\s*```$/, '').trim()
            : streamingContent;
          
          const commitMessage = task.instruction 
            ? `feat: ${task.instruction.substring(0, 50)}${task.instruction.length > 50 ? '...' : ''}`
            : 'feat: Contenido generado automáticamente';
          
          toast.info('Comitando contenido como archivo .md...', {
            description: 'El proceso continuará sin detenerse',
          });
          
          // Ejecutar en background - NO esperar respuesta, continuar inmediatamente
          setTimeout(() => {
            githubExecutor.executeActions({
              repository: task.repository,
              branch: task.repoInfo?.default_branch || 'main',
              actions: [{
                path: fileName,
                content: markdownContent,
                action: 'create',
              }],
              commitMessage: commitMessage,
            })
              .then((executionResult) => {
                if (executionResult.success) {
                  // Actualizar tarea en background sin bloquear
                  updateTask(taskId, {
                    status: 'completed',
                    executionResult: executionResult,
                  }).catch(err => console.warn('Error actualizando tarea:', err));
                  
                  cleanupAndClose(true, `Commit completado - Contenido guardado en ${fileName}`);
                } else {
                  cleanupAndClose(false, `Error al comitar: ${executionResult.error || 'No se pudo crear el commit'}`);
                }
              })
              .catch((error: any) => {
                // Incluso con error, intentar continuar
                console.error('Error ejecutando acciones:', error);
                cleanupAndClose(false, `Error al comitar: ${error.message || 'Error inesperado'}`);
              });
          }, 0);
          
          return; // Salir pero el proceso continúa en background
        }
        
        // Si no hay nada para comitar
        cleanupAndClose(false, 'No hay contenido para comitar - La tarea no tiene contenido generado ni plan');
        
      } catch (error: any) {
        // Capturar cualquier error pero no detener el proceso
        console.error('Error inesperado en handleConfirmCommit:', error);
        cleanupAndClose(false, `Error inesperado: ${error.message || 'Error desconocido'}`);
      }
    })(); // Ejecutar inmediatamente sin await
  };

  // Estadísticas de tareas
  const taskStats = {
    total: tasks.length,
    pending: getTasksByStatus(tasks, 'pending').length,
    processing: getTasksByStatus(tasks, 'processing').length,
    completed: getTasksByStatus(tasks, 'completed').length,
    failed: getTasksByStatus(tasks, 'failed').length,
  };

  // Filtrar y ordenar tareas
  const filteredAndSortedTasks = tasks
    .filter(task => {
      // Filtro por estado
      if (taskStatusFilter !== 'all' && task.status !== taskStatusFilter) {
        return false;
      }
      // Filtro por búsqueda
      if (taskSearchQuery) {
        const query = taskSearchQuery.toLowerCase();
        return (
          task.instruction.toLowerCase().includes(query) ||
          task.repository.toLowerCase().includes(query) ||
          task.id.toLowerCase().includes(query)
        );
      }
      return true;
    })
    .sort((a, b) => {
      let comparison = 0;
      
      if (taskSortBy === 'date') {
        comparison = new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime();
      } else if (taskSortBy === 'status') {
        comparison = a.status.localeCompare(b.status);
      } else if (taskSortBy === 'repository') {
        comparison = a.repository.localeCompare(b.repository);
      }
      
      return taskSortOrder === 'asc' ? comparison : -comparison;
    });

  return (
    <div className="min-h-screen bg-white text-black">
      {/* Header */}
      <header className="relative z-10 bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-5 md:px-6 lg:px-8 py-3.5 md:py-4">
          <div className="flex items-center justify-between">
            <a href="/" className="flex items-center gap-2.5 text-base md:text-lg text-black hover:opacity-80 transition-opacity">
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
            </a>
            
            <nav className="flex items-center gap-4">
              <a href="/" className="text-black hover:opacity-70 transition-opacity font-normal text-sm">
                Inicio
              </a>
              <a href="/agent-control" className="text-black font-semibold text-sm border-b-2 border-black">
                Agent Control
              </a>
              <a href="/continuous-agent" className="text-black hover:opacity-70 transition-opacity font-normal text-sm">
                Agentes Continuos
              </a>
              <a href="/kanban" className="text-black hover:opacity-70 transition-opacity font-normal text-sm">
                Kanban
              </a>
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-5 md:px-6 lg:px-8 py-10 md:py-12">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl md:text-4xl font-bold text-black">Agent Control</h1>
            <p className="text-sm text-gray-500 mt-2">
              Gestiona y monitorea tus tareas de agente autónomo
            </p>
          </div>
          {githubUser && (
            <div className="hidden md:flex items-center gap-3 px-4 py-2 bg-gray-50 rounded-lg border border-gray-200">
              <img 
                src={githubUser.avatar_url} 
                alt={githubUser.login}
                className="w-8 h-8 rounded-full"
              />
              <div>
                <p className="text-xs font-medium text-gray-900">{githubUser.login}</p>
                <p className="text-xs text-gray-500">Conectado</p>
              </div>
            </div>
          )}
        </div>
        
        {/* GitHub Authentication */}
        <div className="mb-8">
          <GithubAuth 
            onAuthSuccess={handleAuthSuccess}
            onAuthError={handleAuthError}
          />
          {authError && (
            <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
              {authError}
            </div>
          )}
        </div>

        {githubUser && (
          <div className="mb-8 space-y-6">
            <RepositorySelector 
              onSelect={handleRepositorySelect}
              selectedRepository={selectedRepository}
              authenticated={!!githubUser}
            />
            
            {selectedRepository && (
              <FolderExplorer
                repository={selectedRepository}
                onFolderSelect={handleFolderSelect}
                selectedFolder={selectedFolder}
              />
            )}
          </div>
        )}

        <div className="grid md:grid-cols-2 gap-10">
          <div className="space-y-6">
            <h3 className="text-black text-xl font-normal">Estado del Agente</h3>
            
            <div className="space-y-4">
              <div>
                <span className="text-black text-sm">Estado: </span>
                <span className="px-3 py-1.5 rounded-md text-sm font-medium bg-gray-100 text-gray-800">
                  Modo Standalone (Sin Backend)
                </span>
              </div>
              
              <div className="flex items-center gap-2">
                <span className="text-black text-sm">Sincronización: </span>
                <BackendSyncIndicator />
              </div>
              
              <div className="text-black text-sm">
                <span className="font-medium">Tareas totales: </span>
                {taskStats.total}
              </div>
              
              <div className="grid grid-cols-2 gap-3 text-black text-sm">
                <div className="p-3 bg-yellow-50 border border-yellow-200 rounded-lg hover:shadow-sm transition-shadow">
                  <div className="flex items-center justify-between mb-1">
                    <div className="text-xs text-yellow-700 font-medium">Pendientes</div>
                    <div className="w-2 h-2 bg-yellow-500 rounded-full animate-pulse"></div>
                  </div>
                  <div className="text-xl font-bold text-yellow-800">{taskStats.pending}</div>
                  {taskStats.total > 0 && (
                    <div className="text-xs text-yellow-600 mt-1">
                      {Math.round((taskStats.pending / taskStats.total) * 100)}% del total
                    </div>
                  )}
                </div>
                <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg hover:shadow-sm transition-shadow">
                  <div className="flex items-center justify-between mb-1">
                    <div className="text-xs text-blue-700 font-medium">Procesando</div>
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                  </div>
                  <div className="text-xl font-bold text-blue-800">{taskStats.processing}</div>
                  {taskStats.total > 0 && (
                    <div className="text-xs text-blue-600 mt-1">
                      {Math.round((taskStats.processing / taskStats.total) * 100)}% del total
                    </div>
                  )}
                </div>
                <div className="p-3 bg-green-50 border border-green-200 rounded-lg hover:shadow-sm transition-shadow">
                  <div className="flex items-center justify-between mb-1">
                    <div className="text-xs text-green-700 font-medium">Completadas</div>
                    <svg className="w-3 h-3 text-green-500" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <div className="text-xl font-bold text-green-800">{taskStats.completed}</div>
                  {taskStats.total > 0 && (
                    <div className="text-xs text-green-600 mt-1">
                      {Math.round((taskStats.completed / taskStats.total) * 100)}% del total
                    </div>
                  )}
                </div>
                <div className="p-3 bg-red-50 border border-red-200 rounded-lg hover:shadow-sm transition-shadow">
                  <div className="flex items-center justify-between mb-1">
                    <div className="text-xs text-red-700 font-medium">Fallidas</div>
                    <svg className="w-3 h-3 text-red-500" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <div className="text-xl font-bold text-red-800">{taskStats.failed}</div>
                  {taskStats.total > 0 && (
                    <div className="text-xs text-red-600 mt-1">
                      {Math.round((taskStats.failed / taskStats.total) * 100)}% del total
                    </div>
                  )}
                </div>
              </div>
              
              {taskStats.total > 0 && (
                <div className="pt-2 border-t border-gray-200">
                  <div className="text-xs text-gray-600 mb-2">Distribución</div>
                  <div className="space-y-1.5">
                    {[
                      { label: 'Completadas', value: taskStats.completed, color: 'bg-green-500', total: taskStats.total },
                      { label: 'Procesando', value: taskStats.processing, color: 'bg-blue-500', total: taskStats.total },
                      { label: 'Fallidas', value: taskStats.failed, color: 'bg-red-500', total: taskStats.total },
                      { label: 'Pendientes', value: taskStats.pending, color: 'bg-gray-400', total: taskStats.total },
                    ].map((item) => {
                      const percentage = item.total > 0 ? (item.value / item.total) * 100 : 0;
                      return (
                        <div key={item.label} className="flex items-center gap-2">
                          <div className="w-20 text-xs text-gray-600">{item.label}</div>
                          <div className="flex-1 bg-gray-200 rounded-full h-4 relative overflow-hidden">
                            <div
                              className={cn("h-full rounded-full transition-all duration-500", item.color)}
                              style={{ width: `${percentage}%` }}
                            />
                          </div>
                          <div className="w-8 text-xs text-gray-600 text-right">{item.value}</div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>

            <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg space-y-2">
              <p className="text-sm text-blue-700">
                <strong>Modo Cursor AI con DeepSeek:</strong> El agente funciona como Cursor - genera código y lo ejecuta automáticamente en GitHub.
              </p>
              <p className="text-xs text-blue-600">
                <strong>🚀 Ejecución Automática:</strong> Después de generar el plan, el agente crea/modifica archivos y hace commits automáticamente en tu repositorio.
              </p>
              <p className="text-xs text-blue-600">
                <strong>⚠️ Persistencia:</strong> Las tareas se guardan automáticamente y se reanudan al recargar la página. Solo se detienen cuando presionas el botón "Detener".
              </p>
              <p className="text-xs text-blue-600">
                <strong>🤖 Modelo por defecto:</strong> DeepSeek Chat está configurado como modelo predeterminado para el procesamiento de tareas.
              </p>
            </div>

            {/* Resumen rápido */}
            {taskStats.total > 0 && (
              <div className="p-4 bg-gradient-to-br from-gray-50 to-gray-100 border border-gray-200 rounded-lg">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="text-sm font-semibold text-gray-800">Resumen Rápido</h4>
                  <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                </div>
                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div className="p-2 bg-white rounded border border-gray-200">
                    <div className="text-gray-500 mb-1">Tasa de éxito</div>
                    <div className="text-lg font-bold text-gray-800">
                      {taskStats.completed + taskStats.failed > 0
                        ? Math.round((taskStats.completed / (taskStats.completed + taskStats.failed)) * 100)
                        : 0}%
                    </div>
                  </div>
                  <div className="p-2 bg-white rounded border border-gray-200">
                    <div className="text-gray-500 mb-1">Con commits</div>
                    <div className="text-lg font-bold text-gray-800">
                      {tasks.filter(t => t.executionResult?.commitSha).length}
                    </div>
                  </div>
                  <div className="p-2 bg-white rounded border border-gray-200">
                    <div className="text-gray-500 mb-1">Última tarea</div>
                    <div className="text-sm font-semibold text-gray-800">
                      {tasks.length > 0
                        ? new Date(tasks.sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime())[0].createdAt).toLocaleDateString('es-ES', { day: 'numeric', month: 'short' })
                        : 'N/A'}
                    </div>
                  </div>
                  <div className="p-2 bg-white rounded border border-gray-200">
                    <div className="text-gray-500 mb-1">Repositorios</div>
                    <div className="text-lg font-bold text-gray-800">
                      {new Set(tasks.map(t => t.repository)).size}
                    </div>
                  </div>
                </div>
              </div>
            )}

            <div className="mt-6">
              <div className="flex items-center justify-between mb-4">
                <h4 className="text-black text-lg font-medium">Agentes Continuos</h4>
                <a
                  href="/continuous-agent"
                  className="text-sm text-blue-600 hover:text-blue-800 underline"
                >
                  Ver todos →
                </a>
              </div>
              <AgentsSection />
            </div>
          </div>

          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h3 className="text-black text-xl font-normal">Nueva Tarea</h3>
              {selectedModel && (
                <div className="flex items-center gap-2 text-xs text-gray-600">
                  <span>Modelo:</span>
                  <span className="px-2 py-1 bg-gray-100 rounded font-medium">{selectedModel}</span>
                </div>
              )}
            </div>
            
            {selectedRepository && (
              <div className="mb-4 space-y-2">
                <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-xs text-gray-600 mb-1">Repositorio seleccionado:</p>
                      <p className="font-medium text-blue-700 text-sm">{selectedRepository.full_name}</p>
                    </div>
                    {selectedRepository.private && (
                      <span className="px-2 py-1 text-xs bg-blue-100 text-blue-700 rounded">Privado</span>
                    )}
                  </div>
                </div>
                {selectedFolder && (
                  <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                    <p className="text-xs text-gray-600 mb-1">Carpeta seleccionada:</p>
                    <p className="font-medium text-green-700 text-sm">{selectedFolder}</p>
                  </div>
                )}
              </div>
            )}
            
            <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
              <ModelSelector
                selectedModel={selectedModel}
                onModelChange={setSelectedModel}
              />
              
              <div>
                <label className="block text-black text-sm mb-2 font-medium">
                  Instrucción
                </label>
                <textarea
                  {...register('instruction')}
                  placeholder="Ej: Crear un archivo README.md con información del proyecto y un componente React llamado Button.tsx"
                  className={cn(
                    "w-full px-4 py-3 bg-white border rounded-lg text-black focus:outline-none focus:ring-2 focus:ring-black focus:border-transparent text-sm",
                    errors.instruction ? "border-red-500 focus:ring-red-500" : "border-gray-300"
                  )}
                  rows={8}
                  maxLength={200000}
                />
                <div className="mt-1 text-xs text-gray-500 text-right">
                  {watch('instruction')?.length || 0} / 200,000 caracteres
                </div>
                {errors.instruction && (
                  <p className="mt-1 text-sm text-red-600">{errors.instruction.message}</p>
                )}
              </div>
              
              <button
                type="submit"
                disabled={isSubmitting || !selectedRepository}
                onClick={(e) => {
                  console.log('🔘 Botón clickeado:', { 
                    isSubmitting, 
                    selectedRepository: !!selectedRepository,
                    formData: watch()
                  });
                  
                  if (!selectedRepository) {
                    e.preventDefault();
                    toast.error('Por favor selecciona un repositorio primero');
                    return;
                  }
                  
                  const formData = watch();
                  if (!formData.instruction || formData.instruction.trim().length < 10) {
                    e.preventDefault();
                    toast.error('La instrucción debe tener al menos 10 caracteres');
                    return;
                  }
                }}
                className="w-full bg-black text-white px-6 py-3 rounded-lg hover:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-sm font-normal"
              >
                {isSubmitting ? 'Creando...' : 'Crear y Procesar'}
              </button>
              
              {/* Mostrar errores del formulario */}
              {Object.keys(errors).length > 0 && (
                <div className="mt-2 p-3 bg-red-50 border border-red-200 rounded-lg">
                  <p className="text-xs font-medium text-red-800 mb-1">Errores del formulario:</p>
                  <ul className="text-xs text-red-700 list-disc list-inside">
                    {errors.instruction && <li>{errors.instruction.message}</li>}
                    {errors.repository && <li>{errors.repository.message}</li>}
                    {errors.folder && <li>{errors.folder.message}</li>}
                  </ul>
                </div>
              )}
            </form>
          </div>
        </div>

        <div className="mt-16 space-y-6">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div>
              <h3 className="text-black text-xl font-normal">Tareas Recientes</h3>
              {tasks.length > 0 && (
                <p className="text-sm text-gray-500 mt-1">
                  Mostrando {filteredAndSortedTasks.length} de {tasks.length} {tasks.length === 1 ? 'tarea' : 'tareas'}
                  {(taskSearchQuery || taskStatusFilter !== 'all') && (
                    <span className="ml-2 text-blue-600">
                      (filtradas)
                    </span>
                  )}
                </p>
              )}
            </div>
            {tasks.length > 0 && (
              <div className="flex flex-wrap gap-2">
                <a
                  href="/kanban"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="px-4 py-2 text-sm bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 transition-colors flex items-center gap-2"
                  title="Abrir vista Kanban en nueva ventana"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2" />
                  </svg>
                  Vista Kanban
                </a>
                <button
                  onClick={handleDeleteUnprocessedTasks}
                  className="px-4 py-2 text-sm bg-orange-100 text-orange-700 rounded-lg hover:bg-orange-200 transition-colors"
                  title="Eliminar tareas sin procesar (pendientes, procesando, fallidas)"
                >
                  Eliminar sin procesar
                </button>
                <button
                  onClick={handleDeleteAllTasks}
                  className="px-4 py-2 text-sm bg-red-100 text-red-700 rounded-lg hover:bg-red-200 transition-colors"
                  title="Eliminar todas las tareas"
                >
                  Eliminar todas
                </button>
              </div>
            )}
          </div>

          {/* Filtros y búsqueda */}
          {tasks.length > 0 && (
            <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 space-y-4">
              <div className="grid md:grid-cols-2 gap-4">
                {/* Búsqueda */}
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-2">
                    Buscar tareas
                    <span className="ml-2 text-gray-400 font-normal">(Ctrl+K)</span>
                  </label>
                  <div className="relative">
                    <input
                      type="text"
                      value={taskSearchQuery}
                      onChange={(e) => setTaskSearchQuery(e.target.value)}
                      placeholder="Buscar por instrucción, repositorio o ID..."
                      className="w-full px-3 py-2 pl-10 text-sm border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-black focus:border-transparent"
                    />
                    <svg className="absolute left-3 top-2.5 h-4 w-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                    {taskSearchQuery && (
                      <button
                        onClick={() => setTaskSearchQuery('')}
                        className="absolute right-2 top-2 text-gray-400 hover:text-gray-600"
                      >
                        <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    )}
                  </div>
                </div>

                {/* Filtro por estado */}
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-2">
                    Filtrar por estado
                  </label>
                  <select
                    value={taskStatusFilter}
                    onChange={(e) => setTaskStatusFilter(e.target.value as any)}
                    className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-black focus:border-transparent bg-white"
                  >
                    <option value="all">Todos los estados</option>
                    <option value="pending">Pendientes</option>
                    <option value="processing">Procesando</option>
                    <option value="completed">Completadas</option>
                    <option value="failed">Fallidas</option>
                  </select>
                </div>
              </div>

              {/* Ordenamiento */}
              <div className="flex flex-wrap items-center gap-4">
                <label className="text-xs font-medium text-gray-700">Ordenar por:</label>
                <div className="flex gap-2">
                  <button
                    onClick={() => {
                      setTaskSortBy('date');
                      setTaskSortOrder(taskSortBy === 'date' && taskSortOrder === 'desc' ? 'asc' : 'desc');
                    }}
                    className={cn(
                      "px-3 py-1.5 text-xs rounded-lg transition-colors",
                      taskSortBy === 'date' 
                        ? "bg-black text-white" 
                        : "bg-gray-200 text-gray-700 hover:bg-gray-300"
                    )}
                  >
                    Fecha {taskSortBy === 'date' && (taskSortOrder === 'desc' ? '↓' : '↑')}
                  </button>
                  <button
                    onClick={() => {
                      setTaskSortBy('status');
                      setTaskSortOrder(taskSortBy === 'status' && taskSortOrder === 'asc' ? 'desc' : 'asc');
                    }}
                    className={cn(
                      "px-3 py-1.5 text-xs rounded-lg transition-colors",
                      taskSortBy === 'status' 
                        ? "bg-black text-white" 
                        : "bg-gray-200 text-gray-700 hover:bg-gray-300"
                    )}
                  >
                    Estado {taskSortBy === 'status' && (taskSortOrder === 'asc' ? '↑' : '↓')}
                  </button>
                  <button
                    onClick={() => {
                      setTaskSortBy('repository');
                      setTaskSortOrder(taskSortBy === 'repository' && taskSortOrder === 'asc' ? 'desc' : 'asc');
                    }}
                    className={cn(
                      "px-3 py-1.5 text-xs rounded-lg transition-colors",
                      taskSortBy === 'repository' 
                        ? "bg-black text-white" 
                        : "bg-gray-200 text-gray-700 hover:bg-gray-300"
                    )}
                  >
                    Repositorio {taskSortBy === 'repository' && (taskSortOrder === 'asc' ? '↑' : '↓')}
                  </button>
                </div>
                {(taskSearchQuery || taskStatusFilter !== 'all') && (
                  <button
                    onClick={() => {
                      setTaskSearchQuery('');
                      setTaskStatusFilter('all');
                    }}
                    className="px-3 py-1.5 text-xs bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors"
                  >
                    Limpiar filtros
                  </button>
                )}
              </div>
            </div>
          )}
          
          <div className="space-y-3">
            {tasks.length === 0 ? (
              <div className="text-center py-12 bg-gray-50 rounded-lg border border-gray-200">
                <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
                <p className="mt-4 text-gray-500 text-sm">No hay tareas aún</p>
                <p className="mt-1 text-gray-400 text-xs">Crea tu primera tarea usando el formulario de arriba</p>
              </div>
            ) : filteredAndSortedTasks.length === 0 ? (
              <div className="text-center py-12 bg-gray-50 rounded-lg border border-gray-200">
                <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
                <p className="mt-4 text-gray-500 text-sm">No se encontraron tareas con los filtros aplicados</p>
                <button
                  onClick={() => {
                    setTaskSearchQuery('');
                    setTaskStatusFilter('all');
                  }}
                  className="mt-2 px-4 py-2 text-sm bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors"
                >
                  Limpiar filtros
                </button>
              </div>
            ) : (
              filteredAndSortedTasks.map((task) => (
                <TaskCard
                  key={task.id}
                  task={task}
                  onStop={() => handleStopTask(task.id)}
                  onResume={() => handleResumeTask(task.id)}
                  onDelete={() => handleDeleteTask(task.id)}
                  onCommit={() => handleCommitClick(task.id)}
                  isCommitting={committingTasks.has(task.id)}
                />
              ))
            )}
          </div>
        </div>
      </main>

      {/* Modal de Aprobación del PLAN */}
      {approvalTask && (
        <ApprovalModal
          task={approvalTask}
          isOpen={!!approvalTask}
          onApprove={handleApprove}
          onReject={handleReject}
          onClose={() => setApprovalTask(null)}
        />
      )}

      {/* Modal de Aprobación del COMMIT */}
      {commitApprovalTask && (
        <CommitApprovalModal
          task={commitApprovalTask}
          isOpen={!!commitApprovalTask}
          onApprove={handleApproveCommit}
          onReject={handleRejectCommit}
          onClose={() => setCommitApprovalTask(null)}
        />
      )}

      {/* Modal de Preview de Commit */}
      {commitPreviewTask && (
        <CommitPreviewModal
          task={commitPreviewTask}
          isOpen={!!commitPreviewTask}
          onCommit={handleConfirmCommit}
          onClose={() => {
            // Solo permitir cerrar si no está comitando
            if (!committingTasks.has(commitPreviewTask.id)) {
              setCommitPreviewTask(null);
            }
          }}
          isLoading={committingTasks.has(commitPreviewTask.id)}
        />
      )}
    </div>
  );
}
