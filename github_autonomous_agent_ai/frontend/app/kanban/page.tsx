'use client';

import { useEffect, useState, useMemo, useCallback, useRef } from 'react';
import Link from 'next/link';
import { format } from 'date-fns';
import { es } from 'date-fns/locale';
import { motion } from 'framer-motion';
import {
  DndContext,
  DragOverlay,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  DragEndEvent,
} from '@dnd-kit/core';
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
} from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { Task, TaskStatus } from '../types/task';
import { useTaskStore } from '../store/task-store';
import { getStatusColor, calculateElapsedTime } from '../utils/task-helpers';
import { KANBAN_COLUMNS } from '../constants/task-constants';
import { toast } from 'sonner';
import { cn } from '../utils/cn';
import { 
  SortableTaskCard, 
  KanbanFilters, 
  StatsPanel, 
  RepoSummary, 
  RecentActivity, 
  TimelineView, 
  ListView, 
  CalendarView, 
  KanbanBoard, 
  TaskDetailPanel, 
  QuickView,
  AgentsSection
} from '../components/kanban';
import { useKanbanFilters } from '../hooks/useKanbanFilters';
import { useKanbanStats } from '../hooks/useKanbanStats';
import { useKanbanDragDrop } from '../hooks/useKanbanDragDrop';
import { useKanbanExport } from '../hooks/useKanbanExport';
import { useKanbanKeyboard } from '../hooks/useKanbanKeyboard';
import { useKanbanPersistence } from '../hooks/useKanbanPersistence';
import { useKanbanSearch } from '../hooks/useKanbanSearch';
import { useLocalStorage } from '../hooks/useLocalStorage';
import { useWebSocketSync } from '../hooks/useWebSocketSync';
import { useNotifications } from '../hooks/useNotifications';
import { useKanbanModals } from '../hooks/useKanbanModals';
import { useKanbanState } from '../hooks/useKanbanState';
import { useKanbanFeatures } from '../hooks/useKanbanFeatures';
import { useKanbanTaskFiltering } from '../hooks/useKanbanTaskFiltering';
import { useKanbanWebSocketHandlers } from '../hooks/useKanbanWebSocketHandlers';
import { KanbanHeader, LoadingSpinner, ViewModeToggle, GroupingControls } from '../components/kanban';
import { NotificationsPanel } from '../components/kanban/panels';

export default function KanbanPage() {
  // Usar selector específico para evitar re-renders innecesarios
  const tasks = useTaskStore((state) => state.tasks);
  const isLoading = useTaskStore((state) => state.isLoading);
  const loadTasks = useTaskStore((state) => state.loadTasks);
  const updateTask = useTaskStore((state) => state.updateTask);
  const deleteTask = useTaskStore((state) => state.deleteTask);
  
  // Centralized state management using custom hooks
  const kanbanState = useKanbanState();
  const kanbanModals = useKanbanModals();
  const kanbanFeatures = useKanbanFeatures(tasks);
  
  // Destructure for easier access (keeping backward compatibility)
  const {
    searchQuery, setSearchQuery,
    selectedRepository, setSelectedRepository,
    selectedTask, setSelectedTask,
    quickViewTask, setQuickViewTask,
    selectedTasks, setSelectedTasks,
    comparisonMode, setComparisonMode,
    modelFilter, setModelFilter,
    tagFilter, setTagFilter,
    priorityFilter, setPriorityFilter,
    errorFilter, setErrorFilter,
    statusFilter, setStatusFilter,
    dateFilter, setDateFilter,
    sortBy, setSortBy,
    sortOrder, setSortOrder,
    viewMode, setViewMode,
    groupByRepo, setGroupByRepo,
    groupByModel, setGroupByModel,
    compactMode, setCompactMode,
    cardSize, setCardSize,
    showStats, setShowStats,
    showMetrics, setShowMetrics,
    darkMode, setDarkMode,
    calendarMonth, setCalendarMonth,
    calendarYear, setCalendarYear,
    searchInStreaming, setSearchInStreaming,
    showSearchSuggestions, setShowSearchSuggestions,
    clearAllFilters,
  } = kanbanState;
  
  const {
    showNotifications, setShowNotifications,
    showAdvancedSearch, setShowAdvancedSearch,
    showTaskComparison, setShowTaskComparison,
    showExportMenu, setShowExportMenu,
    showTagManager, setShowTagManager,
    showShareMenu, setShowShareMenu,
    showNetworkView, setShowNetworkView,
    showHeatmap, setShowHeatmap,
    showComments, setShowComments,
    showReminders, setShowReminders,
    showMilestones, setShowMilestones,
    showDependencies, setShowDependencies,
    showCharts, setShowCharts,
    showShareDialog, setShowShareDialog,
    showProgressView, setShowProgressView,
    showExecutiveSummary, setShowExecutiveSummary,
    showCollaboration, setShowCollaboration,
    showTemplates, setShowTemplates,
    showSmartAlerts, setShowSmartAlerts,
    showActivityFeed, setShowActivityFeed,
    showCustomDashboard, setShowCustomDashboard,
    showSavedFilters, setShowSavedFilters,
    showLabelManager, setShowLabelManager,
    showAuditLogs, setShowAuditLogs,
    showBackupRestore, setShowBackupRestore,
    showComparison, setShowComparison,
    showQuickActions, setShowQuickActions,
    showBulkMenu, setShowBulkMenu,
    showRepoSummary, setShowRepoSummary,
    showRecentActivity, setShowRecentActivity,
    showAnalytics, setShowAnalytics,
    presentationMode, setPresentationMode,
    closeAllModals,
  } = kanbanModals;
  
  const {
    taskTags, setTaskTags,
    customLabels, setCustomLabels,
    taskPriorities, setTaskPriorities,
    taskComments, setTaskComments,
    reminders, setReminders,
    milestones, setMilestones,
    taskDependencies, setTaskDependencies,
    taskHistory, setTaskHistory,
    activityFeed, setActivityFeed,
    smartAlerts, setSmartAlerts,
    auditLogs, setAuditLogs,
    taskTemplates, setTaskTemplates,
    customViews, setCustomViews,
    activeCustomView, setActiveCustomView,
    customColumns, setCustomColumns,
    dashboardLayout, setDashboardLayout,
    autoReports, setAutoReports,
    reportFrequency, setReportFrequency,
    soundEnabled, setSoundEnabled,
    quickActions, setQuickActions,
    bulkOperations, setBulkOperations,
  } = kanbanFeatures;
  
  // Additional local state
  const [recentActivity, setRecentActivity] = useState<Array<{task: Task; action: string; timestamp: Date}>>([]);
  const [wsConnected, setWsConnected] = useState(false);
  const [autoRefresh, setAutoRefresh] = useLocalStorage<boolean>('kanban-auto-refresh', true);
  const [refreshInterval, setRefreshInterval] = useLocalStorage<number>('kanban-refresh-interval', 30);
  
  // Notificaciones mejoradas
  const { 
    notifications, 
    addNotification, 
    unreadCount,
    markAllAsRead,
    clearAll: clearNotifications
  } = useNotifications({ maxNotifications: 100 });
  
  // Trackear actividad reciente y historial de cambios
  useEffect(() => {
    const newActivity: Array<{task: Task; action: string; timestamp: Date}> = [];
    
    // Tareas recién creadas (últimas 10)
    const recentTasks = [...tasks]
      .sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime())
      .slice(0, 10);
    
    recentTasks.forEach(task => {
      const createdTime = new Date(task.createdAt);
      const now = new Date();
      const diffMinutes = Math.floor((now.getTime() - createdTime.getTime()) / 1000 / 60);
      
      if (diffMinutes < 60) {
        newActivity.push({
          task,
          action: 'creada',
          timestamp: createdTime
        });
      }
    });
    
    // Tareas completadas recientemente
    const completedTasks = tasks
      .filter(t => t.status === 'completed')
      .sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime())
      .slice(0, 5);
    
    completedTasks.forEach(task => {
      newActivity.push({
        task,
        action: 'completada',
        timestamp: new Date(task.createdAt)
      });
    });
    
    setRecentActivity(newActivity.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime()).slice(0, 20));
    
    // Trackear cambios en tareas para historial
    tasks.forEach(task => {
      const existingHistory = taskHistory.get(task.id) || [];
      const lastChange = existingHistory[existingHistory.length - 1];
      
      // Detectar cambios significativos
      if (!lastChange || 
          lastChange.changes.status !== task.status ||
          lastChange.changes.error !== task.error) {
        setTaskHistory(prev => {
          const newHistory = new Map(prev);
          const taskHistory = newHistory.get(task.id) || [];
          newHistory.set(task.id, [
            ...taskHistory,
            {
              timestamp: new Date(),
              changes: {
                status: task.status,
                error: task.error,
                streamingContent: task.streamingContent,
              }
            }
          ].slice(-50)); // Mantener solo últimos 50 cambios
          return newHistory;
        });
      }
    });
  }, [tasks, taskHistory]);
  
  // Usar hook de persistencia
  const { savedFilters, saveCurrentFilters: saveFilter, applySavedFilter, deleteSavedFilter } = useKanbanPersistence();
  
  // Persistir preferencias
  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('kanban-search', searchQuery);
    }
  }, [searchQuery]);
  
  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('kanban-repo-filter', selectedRepository);
    }
  }, [selectedRepository]);
  
  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('kanban-stats', showStats.toString());
    }
  }, [showStats]);
  
  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('kanban-group-repo', groupByRepo.toString());
    }
  }, [groupByRepo]);
  
  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('kanban-view-mode', viewMode);
    }
  }, [viewMode]);
  
  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('kanban-status-filter', statusFilter);
    }
  }, [statusFilter]);
  
  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('kanban-date-filter', dateFilter);
    }
  }, [dateFilter]);
  
  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('kanban-sort', sortBy);
    }
  }, [sortBy]);
  
  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('kanban-sort-order', sortOrder);
    }
  }, [sortOrder]);
  
  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('kanban-card-size', cardSize);
    }
  }, [cardSize]);
  
  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('kanban-group-model', groupByModel.toString());
    }
  }, [groupByModel]);
  
  // Guardar filtros
  // Función para guardar filtros actuales
  const handleSaveCurrentFilters = useCallback(() => {
    const name = prompt('Nombre para este filtro:');
    if (name) {
      saveFilter({
        searchQuery,
        selectedRepository,
        statusFilter,
        dateFilter,
        sortBy,
        sortOrder,
      }, name);
      toast.success('Filtro guardado', {
        description: `Filtro "${name}" guardado correctamente`,
      });
    }
  }, [searchQuery, selectedRepository, statusFilter, dateFilter, sortBy, sortOrder, saveFilter]);
  
  // Función para aplicar filtro guardado
  const handleApplySavedFilter = useCallback((filter: any) => {
    const filters = applySavedFilter(filter);
    setSearchQuery(filters.searchQuery || '');
    setSelectedRepository(filters.selectedRepository || 'all');
    setStatusFilter((filters.statusFilter as TaskStatus | 'all') || 'all');
    setDateFilter((filters.dateFilter as 'all' | 'today' | 'week' | 'month' | 'custom') || 'all');
    setSortBy((filters.sortBy as 'date' | 'status' | 'repo' | 'model') || 'date');
    setSortOrder((filters.sortOrder as 'asc' | 'desc') || 'desc');
    toast.success('Filtro aplicado', {
      description: `Filtro "${filter.name}" aplicado`,
    });
  }, [applySavedFilter]);
  
  // Función para eliminar filtro guardado
  const handleDeleteSavedFilter = useCallback((index: number) => {
    deleteSavedFilter(index);
    toast.success('Filtro eliminado');
  }, [deleteSavedFilter]);
  
  // Obtener repositorios únicos
  const repositories = useMemo(() => {
    const repos = new Set(tasks.map(t => t.repository));
    return Array.from(repos).sort();
  }, [tasks]);
  
  // Obtener modelos únicos
  const models = useMemo(() => {
    const modelSet = new Set(tasks.map(t => t.model).filter(Boolean));
    return Array.from(modelSet).sort();
  }, [tasks]);
  
  // Usar hooks personalizados para lógica de filtrado y estadísticas
  const { stats } = useKanbanStats(tasks);
  const { filteredTasks: baseFilteredTasks } = useKanbanFilters({
    tasks,
    searchQuery,
    selectedRepository,
    statusFilter,
    dateFilter,
    sortBy,
    sortOrder,
  });
  
  // Aplicar filtro por modelo adicional
  const filteredTasks = useMemo(() => {
    if (modelFilter === 'all') return baseFilteredTasks;
    return baseFilteredTasks.filter(task => task.model === modelFilter);
  }, [baseFilteredTasks, modelFilter]);
  
  // Usar hook de exportación
  const { handleExportTasks: exportTasks, handleExportToCSV: exportToCSV } = useKanbanExport();
  
  // Sugerencias de búsqueda
  const searchSuggestions = useMemo(() => {
    if (!searchQuery || searchQuery.length < 2) return [];
    
    const query = searchQuery.toLowerCase();
    const suggestions: string[] = [];
    
    // Sugerencias de repositorios
    repositories.forEach(repo => {
      if (repo.toLowerCase().includes(query) && !query.startsWith('repo:')) {
        suggestions.push(`repo:${repo}`);
      }
    });
    
    // Sugerencias de estados
    KANBAN_COLUMNS.forEach(col => {
      if (col.label.toLowerCase().includes(query) && !query.startsWith('status:')) {
        suggestions.push(`status:${col.id}`);
      }
    });
    
    // Sugerencias de modelos
    const models = Array.from(new Set(tasks.map(t => t.model).filter(Boolean)));
    models.forEach(model => {
      if (model && model.toLowerCase().includes(query) && !query.startsWith('model:')) {
        suggestions.push(`model:${model}`);
      }
    });
    
    return suggestions.slice(0, 5);
  }, [searchQuery, repositories, tasks]);
  
  const [showQuickStats, setShowQuickStats] = useState(false);

  const sensors = useSensors(
    useSensor(PointerSensor, {
      activationConstraint: {
        distance: 8, // Requiere 8px de movimiento antes de activar drag
      },
    }),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  );
  
  // Usar hook de drag and drop
  const {
    activeId,
    dragOverColumn,
    handleDragStart,
    handleDragOver,
    handleDragEnd,
    handleDragCancel,
  } = useKanbanDragDrop({
    tasks,
    updateTask,
    onActivityUpdate: (activity: { task: Task; action: string; timestamp: Date }) => {
      setRecentActivity(prev => [activity, ...prev].slice(0, 20));
    },
  });

  // WebSocket event handlers
  const wsHandlers = useKanbanWebSocketHandlers({
    soundEnabled,
    onActivityUpdate: (activity) => {
      setRecentActivity(prev => [activity, ...prev].slice(0, 20));
    },
  });

  // Sincronización en tiempo real con WebSocket
  const { connected: wsConnectedState } = useWebSocketSync({
    enabled: true,
    onTaskUpdate: wsHandlers.handleTaskUpdate,
    onTaskCreated: wsHandlers.handleTaskCreated,
    onTaskCompleted: wsHandlers.handleTaskCompleted,
    onTaskFailed: wsHandlers.handleTaskFailed,
  });
  
  // Actualizar estado de conexión WebSocket
  useEffect(() => {
    setWsConnected(wsConnectedState);
  }, [wsConnectedState]);

  useEffect(() => {
    let mounted = true;
    
    // Cargar tareas al montar
    loadTasks(true);
    
    // Auto-refresh inteligente: solo si está habilitado y WebSocket no está conectado
    let interval: NodeJS.Timeout | null = null;
    
    if (autoRefresh) {
      const intervalMs = refreshInterval * 1000;
      interval = setInterval(() => {
      if (mounted && !isLoading) {
          // Solo sincronizar si WebSocket no está conectado (el WebSocket maneja eventos en tiempo real)
          if (!wsConnected) {
            console.log('🔄 Auto-refresh: sincronizando tareas (WebSocket desconectado)');
        loadTasks();
      }
        }
      }, intervalMs);
    }
    
    return () => {
      mounted = false;
      if (interval) {
      clearInterval(interval);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoRefresh, refreshInterval, wsConnected]); // Re-ejecutar si cambian las configuraciones

  // Usar hook de atajos de teclado mejorado
  useKanbanKeyboard({
    onFocusSearch: () => {
      const searchInput = document.querySelector('input[type="text"][placeholder*="Buscar"]') as HTMLInputElement;
      if (searchInput) {
        searchInput.focus();
        searchInput.select();
      }
    },
    onCreateTask: () => {
      window.location.href = '/agent-control';
    },
    onClosePanel: () => {
      setSelectedTask(null);
      setShowTaskComparison(false);
      setShowNotifications(false);
    },
    isPanelOpen: !!selectedTask || showTaskComparison || showNotifications,
  });
  
  // Atajos de teclado adicionales
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ctrl/Cmd + K para búsqueda avanzada
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        setShowAdvancedSearch(!showAdvancedSearch);
      }
      
      // Ctrl/Cmd + D para seleccionar/deseleccionar tarea actual
      if ((e.ctrlKey || e.metaKey) && e.key === 'd' && selectedTask) {
        e.preventDefault();
        setSelectedTasks(prev => {
          const newSet = new Set(prev);
          if (newSet.has(selectedTask.id)) {
            newSet.delete(selectedTask.id);
          } else {
            newSet.add(selectedTask.id);
          }
          return newSet;
        });
      }
      
      // Ctrl/Cmd + Shift + C para comparar tareas seleccionadas
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'C') {
        e.preventDefault();
        if (selectedTasks.size >= 2) {
          setShowTaskComparison(true);
        } else {
          toast.info('Selecciona al menos 2 tareas para comparar');
        }
      }
      
      // Ctrl/Cmd + M para toggle métricas
      if ((e.ctrlKey || e.metaKey) && e.key === 'm') {
        e.preventDefault();
        setShowMetrics(!showMetrics);
      }
      
      // Ctrl/Cmd + , para toggle modo compacto
      if ((e.ctrlKey || e.metaKey) && e.key === ',') {
        e.preventDefault();
        setCompactMode(!compactMode);
      }
      
      // Ctrl/Cmd + P para modo presentación
      if ((e.ctrlKey || e.metaKey) && e.key === 'p' && !e.shiftKey) {
        e.preventDefault();
        setPresentationMode(!presentationMode);
      }
      
      // Ctrl/Cmd + Shift + S para compartir
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'S') {
        e.preventDefault();
        setShowShareDialog(true);
      }
      
      // Ctrl/Cmd + G para gráficos
      if ((e.ctrlKey || e.metaKey) && e.key === 'g' && !e.shiftKey) {
        e.preventDefault();
        setShowCharts(!showCharts);
      }
      
      // F11 para pantalla completa
      if (e.key === 'F11') {
        e.preventDefault();
        if (!document.fullscreenElement) {
          document.documentElement.requestFullscreen();
        } else {
          document.exitFullscreen();
        }
      }
      
      // Escape para limpiar selección y cerrar modales
      if (e.key === 'Escape') {
        if (selectedTasks.size > 0) {
          setSelectedTasks(new Set());
        }
        if (presentationMode) {
          setPresentationMode(false);
        }
        if (showNetworkView) {
          setShowNetworkView(false);
        }
        if (showHeatmap) {
          setShowHeatmap(false);
        }
        if (showMilestones) {
          setShowMilestones(false);
        }
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [showAdvancedSearch, selectedTask, selectedTasks, showMetrics, compactMode, setCompactMode]);
  
  // Cerrar panel de notificaciones al hacer clic fuera
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as HTMLElement;
      if (showNotifications && !target.closest('.notifications-panel')) {
        setShowNotifications(false);
      }
      if (showExportMenu && !target.closest('.export-menu')) {
        setShowExportMenu(false);
      }
      if (showTagManager && !target.closest('.tag-manager')) {
        setShowTagManager(false);
      }
    };
    
    if (showNotifications || showExportMenu || showTagManager) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [showNotifications, showExportMenu, showTagManager]);
  
  // Verificar recordatorios
  useEffect(() => {
    const checkReminders = () => {
      const now = new Date();
      reminders.forEach(reminder => {
        const reminderTime = new Date(reminder.time);
        if (reminderTime <= now && reminderTime > new Date(now.getTime() - 60000)) {
          const task = tasks.find(t => t.id === reminder.taskId);
          addNotification(
            'warning',
            'Recordatorio',
            reminder.message
          );
          // Remover recordatorio después de notificar
          setReminders(prev => prev.filter(r => r.id !== reminder.id));
        }
      });
    };
    
    const interval = setInterval(checkReminders, 30000); // Verificar cada 30 segundos
    return () => clearInterval(interval);
  }, [reminders, tasks, addNotification]);
  
  // Alertas inteligentes
  useEffect(() => {
    const newAlerts: Array<{id: string; type: string; message: string; timestamp: Date}> = [];
    
    // Alerta: Tasa de éxito baja
    const successRate = tasks.length > 0 
      ? (tasks.filter(t => t.status === 'completed').length / tasks.length) * 100 
      : 100;
    if (successRate < 50 && tasks.length > 10) {
      newAlerts.push({
        id: 'low-success-rate',
        type: 'warning',
        message: `Tasa de éxito baja: ${Math.round(successRate)}%. Considera revisar las tareas fallidas.`,
        timestamp: new Date()
      });
    }
    
    // Alerta: Muchas tareas en proceso
    const processingCount = tasks.filter(t => t.status === 'processing' || t.status === 'running').length;
    if (processingCount > 20) {
      newAlerts.push({
        id: 'many-processing',
        type: 'info',
        message: `${processingCount} tareas en proceso. El sistema está muy activo.`,
        timestamp: new Date()
      });
    }
    
    // Alerta: Tareas fallidas recientes
    const recentFailures = tasks.filter(t => 
      t.status === 'failed' && 
      new Date(t.updatedAt || t.createdAt).getTime() > Date.now() - 3600000
    );
    if (recentFailures.length > 5) {
      newAlerts.push({
        id: 'recent-failures',
        type: 'error',
        message: `${recentFailures.length} tareas fallaron en la última hora.`,
        timestamp: new Date()
      });
    }
    
    // Alerta: Sin actividad reciente
    const lastActivity = tasks.length > 0 
      ? Math.max(...tasks.map(t => new Date(t.updatedAt || t.createdAt).getTime()))
      : 0;
    const hoursSinceActivity = (Date.now() - lastActivity) / (1000 * 60 * 60);
    if (hoursSinceActivity > 24 && tasks.length > 0) {
      newAlerts.push({
        id: 'no-recent-activity',
        type: 'warning',
        message: `Sin actividad en las últimas ${Math.round(hoursSinceActivity)} horas.`,
        timestamp: new Date()
      });
    }
    
    setSmartAlerts(newAlerts);
  }, [tasks]);
  
  // Feed de actividad mejorado
  useEffect(() => {
    const newActivities: Array<{id: string; type: string; message: string; timestamp: Date; taskId?: string}> = [];
    
    // Actividad: Tareas creadas
    tasks.filter(t => {
      const created = new Date(t.createdAt);
      return (Date.now() - created.getTime()) < 3600000; // Última hora
    }).forEach(task => {
      newActivities.push({
        id: `created-${task.id}`,
        type: 'created',
        message: `Nueva tarea creada: ${task.repository}`,
        timestamp: new Date(task.createdAt),
        taskId: task.id
      });
    });
    
    // Actividad: Tareas completadas
    tasks.filter(t => 
      t.status === 'completed' && 
      new Date(t.updatedAt || t.createdAt).getTime() > Date.now() - 3600000
    ).forEach(task => {
      newActivities.push({
        id: `completed-${task.id}`,
        type: 'completed',
        message: `Tarea completada: ${task.repository}`,
        timestamp: new Date(task.updatedAt || task.createdAt),
        taskId: task.id
      });
    });
    
    // Actividad: Tareas fallidas
    tasks.filter(t => 
      t.status === 'failed' && 
      new Date(t.updatedAt || t.createdAt).getTime() > Date.now() - 3600000
    ).forEach(task => {
      newActivities.push({
        id: `failed-${task.id}`,
        type: 'failed',
        message: `Tarea fallida: ${task.repository}`,
        timestamp: new Date(task.updatedAt || task.createdAt),
        taskId: task.id
      });
    });
    
    setActivityFeed(newActivities.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime()).slice(0, 50));
  }, [tasks]);
  
  // Reportes automáticos
  useEffect(() => {
    if (!autoReports) return;
    
    const generateReport = () => {
      const report = {
        date: new Date().toISOString(),
        totalTasks: tasks.length,
        completed: tasks.filter(t => t.status === 'completed').length,
        failed: tasks.filter(t => t.status === 'failed').length,
        processing: tasks.filter(t => t.status === 'processing' || t.status === 'running').length,
        successRate: tasks.length > 0 
          ? Math.round((tasks.filter(t => t.status === 'completed').length / tasks.length) * 100)
          : 0,
        topRepositories: repositories.slice(0, 5).map(repo => ({
          name: repo,
          count: tasks.filter(t => t.repository === repo).length
        }))
      };
      
      // Guardar reporte en localStorage
      const reports = JSON.parse(localStorage.getItem('kanban-reports') || '[]');
      reports.push(report);
      if (reports.length > 100) reports.shift(); // Mantener solo últimos 100
      localStorage.setItem('kanban-reports', JSON.stringify(reports));
      
      addNotification('info', 'Reporte generado', `Reporte ${reportFrequency} creado automáticamente`);
    };
    
    let interval: NodeJS.Timeout;
    if (reportFrequency === 'daily') {
      interval = setInterval(generateReport, 24 * 60 * 60 * 1000);
    } else if (reportFrequency === 'weekly') {
      interval = setInterval(generateReport, 7 * 24 * 60 * 60 * 1000);
    } else {
      interval = setInterval(generateReport, 30 * 24 * 60 * 60 * 1000);
    }
    
    return () => clearInterval(interval);
  }, [autoReports, reportFrequency, tasks, repositories, addNotification]);
  
  // Logs de auditoría
  useEffect(() => {
    const logAction = (action: string, details: string) => {
      setAuditLogs(prev => [{
        id: `log-${Date.now()}`,
        action,
        details,
        timestamp: new Date()
      }, ...prev].slice(0, 1000)); // Mantener últimos 1000 logs
    };
    
    // Log cuando se crea una tarea
    const handleTaskCreate = () => {
      logAction('Tarea creada', 'Nueva tarea agregada al sistema');
    };
    
    // Log cuando se completa una tarea
    tasks.filter(t => t.status === 'completed').forEach(task => {
      const existingLog = auditLogs.find(log => log.details.includes(task.id) && log.action === 'Tarea completada');
      if (!existingLog) {
        logAction('Tarea completada', `Tarea ${task.id} completada: ${task.repository}`);
      }
    });
    
    // Log cuando falla una tarea
    tasks.filter(t => t.status === 'failed').forEach(task => {
      const existingLog = auditLogs.find(log => log.details.includes(task.id) && log.action === 'Tarea fallida');
      if (!existingLog) {
        logAction('Tarea fallida', `Tarea ${task.id} falló: ${task.repository} - ${task.error?.substring(0, 50)}`);
      }
    });
  }, [tasks, auditLogs]);
  
  // Use centralized task filtering hook
  const {
    tasksWithTagFilter,
    tasksWithPriorityFilter,
    tasksWithStreamingSearch,
    tasksWithErrorFilter,
    tasksByColumn,
  } = useKanbanTaskFiltering({
    filteredTasks,
    tagFilter,
    taskTags,
    priorityFilter,
    taskPriorities,
    searchInStreaming,
    searchQuery,
    errorFilter,
  });
  
  // Callbacks de exportación que dependen de tasksWithStreamingSearch
  const handleExportTasks = useCallback(() => {
    exportTasks(tasksWithStreamingSearch);
    toast.success('Tareas exportadas', {
      description: `${tasksWithStreamingSearch.length} tareas exportadas correctamente`,
    });
  }, [tasksWithStreamingSearch, exportTasks]);
  
  const handleExportToCSV = useCallback(() => {
    exportToCSV(tasksWithStreamingSearch);
    toast.success('Tareas exportadas a CSV', {
      description: `${tasksWithStreamingSearch.length} tareas exportadas correctamente`,
    });
  }, [tasksWithStreamingSearch, exportToCSV]);
  
  // Acciones rápidas
  useEffect(() => {
    setQuickActions([
      {
        id: 'clear-filters',
        name: 'Limpiar todos los filtros',
        action: () => {
          setSearchQuery('');
          setSelectedRepository('all');
          setModelFilter('all');
          setTagFilter('all');
          setPriorityFilter('all');
          setErrorFilter('all');
          toast.success('Filtros limpiados');
        }
      },
      {
        id: 'select-all',
        name: 'Seleccionar todas las tareas visibles',
        action: () => {
          setSelectedTasks(new Set(tasksWithErrorFilter.map(t => t.id)));
          toast.success(`${tasksWithErrorFilter.length} tareas seleccionadas`);
        }
      },
      {
        id: 'export-visible',
        name: 'Exportar tareas visibles',
        action: () => {
          handleExportTasks();
        }
      },
      {
        id: 'mark-all-read',
        name: 'Marcar todas las notificaciones como leídas',
        action: () => {
          markAllAsRead();
          toast.success('Notificaciones marcadas como leídas');
        }
      }
    ]);
  }, [tasksWithErrorFilter, handleExportTasks, markAllAsRead]);

  const handleDeleteTask = async (taskId: string) => {
    const taskToDelete = tasks.find((t) => t.id === taskId);
    try {
      await deleteTask(taskId);

      // Agregar a actividad reciente
      if (taskToDelete) {
        setRecentActivity(prev => [{
          task: taskToDelete,
          action: 'eliminada',
          timestamp: new Date()
        }, ...prev].slice(0, 20));
      }

      toast.success('Tarea eliminada', {
        description: 'La tarea ha sido eliminada correctamente',
        duration: 3000,
      });
    } catch (error) {
      toast.error('Error al eliminar', {
        description: 'No se pudo eliminar la tarea',
        duration: 4000,
      });
    }
  };



  // Aplicar modo oscuro
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);
  
  // Extraer tags de tareas automáticamente
  useEffect(() => {
    const newTags = new Map<string, string[]>();
    tasks.forEach(task => {
      const tags: string[] = [];
      // Extraer tags del contenido o instrucción
      const text = `${task.instruction} ${task.error || ''} ${task.repository}`.toLowerCase();
      if (text.includes('urgent') || text.includes('urgente')) tags.push('urgent');
      if (text.includes('bug') || text.includes('error')) tags.push('bug');
      if (text.includes('feature') || text.includes('nueva')) tags.push('feature');
      if (text.includes('refactor')) tags.push('refactor');
      if (text.includes('test') || text.includes('prueba')) tags.push('test');
      if (task.status === 'completed') tags.push('done');
      if (task.status === 'failed') tags.push('failed');
      if (task.status === 'processing') tags.push('in-progress');
      newTags.set(task.id, tags);
    });
    setTaskTags(newTags);
  }, [tasks]);
  
  // Obtener todos los tags únicos
  const allTags = useMemo(() => {
    const tagSet = new Set<string>();
    taskTags.forEach(tags => tags.forEach(tag => tagSet.add(tag)));
    return Array.from(tagSet).sort();
  }, [taskTags]);
  
  // tasksByColumn is now provided by useKanbanTaskFiltering hook
  
  // Detectar prioridades automáticamente
  useEffect(() => {
    const newPriorities = new Map<string, 'low' | 'medium' | 'high' | 'urgent'>();
    tasks.forEach(task => {
      const text = `${task.instruction} ${task.error || ''}`.toLowerCase();
      let priority: 'low' | 'medium' | 'high' | 'urgent' = 'medium';
      
      if (text.includes('urgent') || text.includes('crítico') || text.includes('critical')) {
        priority = 'urgent';
      } else if (text.includes('important') || text.includes('importante') || task.status === 'failed') {
        priority = 'high';
      } else if (text.includes('low') || text.includes('bajo') || task.status === 'completed') {
        priority = 'low';
      }
      
      newPriorities.set(task.id, priority);
    });
    setTaskPriorities(newPriorities);
  }, [tasks]);
  
  // Detectar dependencias entre tareas (mismo repositorio, instrucciones relacionadas)
  useEffect(() => {
    const newDependencies = new Map<string, string[]>();
    tasks.forEach(task => {
      const relatedTasks = tasks.filter(t => 
        t.id !== task.id &&
        t.repository === task.repository &&
        (t.instruction.toLowerCase().includes(task.instruction.toLowerCase().substring(0, 20)) ||
         task.instruction.toLowerCase().includes(t.instruction.toLowerCase().substring(0, 20)))
      );
      if (relatedTasks.length > 0) {
        newDependencies.set(task.id, relatedTasks.map(t => t.id));
      }
    });
    setTaskDependencies(newDependencies);
  }, [tasks]);
  
  
  // Generar milestones automáticamente
  useEffect(() => {
    const newMilestones: Array<{id: string; name: string; date: Date; tasks: string[]}> = [];
    
    // Milestone: Tareas completadas hoy
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const completedToday = tasks.filter(t => {
      if (t.status !== 'completed') return false;
      const completedDate = new Date(t.updatedAt || t.createdAt);
      completedDate.setHours(0, 0, 0, 0);
      return completedDate.getTime() === today.getTime();
    });
    
    if (completedToday.length > 0) {
      newMilestones.push({
        id: 'today',
        name: `Completadas hoy (${completedToday.length})`,
        date: today,
        tasks: completedToday.map(t => t.id)
      });
    }
    
    // Milestone: 10 tareas completadas
    const completed = tasks.filter(t => t.status === 'completed');
    if (completed.length >= 10 && completed.length % 10 === 0) {
      newMilestones.push({
        id: `milestone-${completed.length}`,
        name: `${completed.length} tareas completadas`,
        date: new Date(),
        tasks: completed.slice(-10).map(t => t.id)
      });
    }
    
    setMilestones(newMilestones);
  }, [tasks]);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <div className={cn("min-h-screen transition-colors", darkMode ? "bg-gray-900 text-white" : "bg-gray-50 text-black")}>
      {/* Header */}
      <header className={cn("border-b sticky top-0 z-50 shadow-sm transition-colors", darkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-200")}>
        <div className="max-w-full mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
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
              <h1 className="text-xl font-bold">Vista Kanban</h1>
            </div>
            <div className="flex items-center gap-3">
              {/* Indicador de conexión WebSocket */}
              <div className="flex items-center gap-2">
                <div className={cn(
                  "w-2 h-2 rounded-full",
                  wsConnected ? "bg-green-500 animate-pulse" : "bg-red-500"
                )} title={wsConnected ? "WebSocket conectado" : "WebSocket desconectado"} />
                <span className="text-xs text-gray-500">
                  {wsConnected ? "En tiempo real" : "Modo offline"}
                </span>
              </div>
              
              {/* Contador de notificaciones */}
              {unreadCount > 0 && (
                <button
                  onClick={() => markAllAsRead()}
                  className="relative px-2 py-1 text-sm text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-lg transition-colors"
                  title={`${unreadCount} notificaciones sin leer`}
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
                  </svg>
                  {unreadCount > 0 && (
                    <span className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 text-white text-xs rounded-full flex items-center justify-center">
                      {unreadCount > 9 ? '9+' : unreadCount}
                    </span>
                  )}
                </button>
              )}
              
              <span className="text-sm text-gray-600">
                Total: <span className="font-semibold">{tasks.length}</span> tareas
                {(searchQuery || selectedRepository !== 'all' || tagFilter !== 'all' || priorityFilter !== 'all') && (
                  <span className="ml-2 text-blue-600">
                    ({tasksWithStreamingSearch.length} {tasksWithStreamingSearch.length === 1 ? 'encontrada' : 'encontradas'})
              </span>
                )}
              </span>
              {(searchQuery || selectedRepository !== 'all') && (
                <button
                  onClick={() => {
                    setSearchQuery('');
                    setSelectedRepository('all');
                  }}
                  className="px-3 py-1.5 text-sm text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-lg transition-colors"
                >
                  Limpiar filtros
                </button>
              )}
              {/* Modo oscuro toggle */}
              <button
                onClick={() => setDarkMode(!darkMode)}
                className={cn(
                  "px-3 py-2 rounded-lg text-sm transition-colors flex items-center gap-2",
                  darkMode 
                    ? "bg-yellow-100 text-yellow-700 border border-yellow-300" 
                    : "bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300"
                )}
                title="Modo oscuro"
              >
                {darkMode ? (
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
                  </svg>
                ) : (
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
                  </svg>
                )}
              </button>
              
              {/* Analytics */}
              <button
                onClick={() => setShowAnalytics(!showAnalytics)}
                className={cn(
                  "px-3 py-2 rounded-lg text-sm transition-colors flex items-center gap-2",
                  showAnalytics 
                    ? "bg-purple-100 text-purple-700 border border-purple-300" 
                    : darkMode 
                      ? "bg-gray-700 text-gray-200 hover:bg-gray-600 border border-gray-600"
                      : "bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300"
                )}
                title="Analytics avanzado"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                Analytics
              </button>
              
              {/* Export menu */}
              <div className="relative export-menu">
                <button
                  onClick={() => setShowExportMenu(!showExportMenu)}
                  className={cn(
                    "px-3 py-2 rounded-lg text-sm transition-colors flex items-center gap-2",
                    darkMode 
                      ? "bg-gray-700 text-gray-200 hover:bg-gray-600 border border-gray-600"
                      : "bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300"
                  )}
                  title="Exportar"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  Exportar
                </button>
                
                {showExportMenu && (
                  <div className={cn(
                    "absolute right-0 top-full mt-2 w-48 rounded-lg shadow-xl z-50 p-2",
                    darkMode ? "bg-gray-800 border border-gray-700" : "bg-white border border-gray-200"
                  )}>
                    <button
                      onClick={() => {
                        handleExportTasks();
                        setShowExportMenu(false);
                      }}
                      className="w-full text-left px-3 py-2 text-sm hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
                    >
                      JSON
                    </button>
                    <button
                      onClick={() => {
                        handleExportToCSV();
                        setShowExportMenu(false);
                      }}
                      className="w-full text-left px-3 py-2 text-sm hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
                    >
                      CSV
                    </button>
                    <button
                      onClick={() => {
                        const markdown = tasksWithStreamingSearch.map(t => 
                          `## ${t.repository}\n- **Estado**: ${t.status}\n- **Instrucción**: ${t.instruction}\n- **Modelo**: ${t.model || 'N/A'}\n`
                        ).join('\n');
                        const blob = new Blob([markdown], { type: 'text/markdown' });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `tasks-${new Date().toISOString()}.md`;
                        a.click();
                        toast.success('Exportado a Markdown');
                        setShowExportMenu(false);
                      }}
                      className="w-full text-left px-3 py-2 text-sm hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
                    >
                      Markdown
                    </button>
                    <button
                      onClick={() => {
                        const html = `
<!DOCTYPE html>
<html>
<head><title>Tareas Exportadas</title></head>
<body>
  <h1>Tareas Exportadas - ${new Date().toLocaleDateString()}</h1>
  ${tasksWithStreamingSearch.map(t => `
    <div style="border: 1px solid #ccc; padding: 10px; margin: 10px 0;">
      <h3>${t.repository}</h3>
      <p><strong>Estado:</strong> ${t.status}</p>
      <p><strong>Instrucción:</strong> ${t.instruction}</p>
      <p><strong>Modelo:</strong> ${t.model || 'N/A'}</p>
    </div>
  `).join('')}
</body>
</html>`;
                        const blob = new Blob([html], { type: 'text/html' });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `tasks-${new Date().toISOString()}.html`;
                        a.click();
                        toast.success('Exportado a HTML');
                        setShowExportMenu(false);
                      }}
                      className="w-full text-left px-3 py-2 text-sm hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
                    >
                      HTML
                    </button>
                  </div>
                )}
              </div>
              
              {/* Tags */}
              {allTags.length > 0 && (
                <div className="relative tag-manager">
                  <button
                    onClick={() => setShowTagManager(!showTagManager)}
                    className={cn(
                      "px-3 py-2 rounded-lg text-sm transition-colors flex items-center gap-2",
                      darkMode 
                        ? "bg-gray-700 text-gray-200 hover:bg-gray-600 border border-gray-600"
                        : "bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300"
                    )}
                    title="Gestionar tags"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z" />
                    </svg>
                    Tags
                  </button>
                  
                  {showTagManager && (
                    <div className={cn(
                      "absolute right-0 top-full mt-2 w-64 rounded-lg shadow-xl z-50 p-4",
                      darkMode ? "bg-gray-800 border border-gray-700" : "bg-white border border-gray-200"
                    )}>
                      <div className="mb-3">
                        <label className="block text-sm font-medium mb-2">Filtrar por tag</label>
                        <select
                          value={tagFilter}
                          onChange={(e) => setTagFilter(e.target.value)}
                          className={cn(
                            "w-full px-3 py-2 rounded-lg text-sm",
                            darkMode 
                              ? "bg-gray-700 text-white border border-gray-600"
                              : "bg-white border border-gray-300"
                          )}
                        >
                          <option value="all">Todos los tags</option>
                          {allTags.map(tag => (
                            <option key={tag} value={tag}>{tag}</option>
                          ))}
                        </select>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {allTags.map(tag => (
                          <span
                            key={tag}
                            className={cn(
                              "px-2 py-1 rounded text-xs",
                              tagFilter === tag
                                ? "bg-blue-500 text-white"
                                : darkMode
                                  ? "bg-gray-700 text-gray-200"
                                  : "bg-gray-100 text-gray-700"
                            )}
                          >
                            {tag} ({tasksWithTagFilter.filter(t => (taskTags.get(t.id) || []).includes(tag)).length})
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
              
              <Link
                href="/agent-control"
                className={cn(
                  "px-4 py-2 rounded-lg transition-colors text-sm",
                  darkMode 
                    ? "bg-gray-700 text-white hover:bg-gray-600"
                    : "bg-black text-white hover:bg-gray-800"
                )}
              >
                Agent Control
              </Link>
            </div>
          </div>
        </div>
      </header>

      {/* Barra de búsqueda y filtros */}
      <KanbanFilters
        searchQuery={searchQuery}
        setSearchQuery={setSearchQuery}
        selectedRepository={selectedRepository}
        setSelectedRepository={setSelectedRepository}
        repositories={repositories}
        statusFilter={statusFilter}
        setStatusFilter={setStatusFilter}
        dateFilter={dateFilter}
        setDateFilter={setDateFilter}
        viewMode={viewMode}
        setViewMode={setViewMode}
        sortBy={sortBy}
        setSortBy={setSortBy}
        sortOrder={sortOrder}
        setSortOrder={setSortOrder}
        cardSize={cardSize}
        setCardSize={setCardSize}
        showRepoSummary={showRepoSummary}
        setShowRepoSummary={setShowRepoSummary}
        showRecentActivity={showRecentActivity}
        setShowRecentActivity={setShowRecentActivity}
        recentActivityCount={recentActivity.length}
        groupByRepo={groupByRepo}
        setGroupByRepo={setGroupByRepo}
        showStats={showStats}
        setShowStats={setShowStats}
        savedFilters={savedFilters}
        saveCurrentFilters={handleSaveCurrentFilters}
        applySavedFilter={applySavedFilter}
        deleteSavedFilter={deleteSavedFilter}
        handleExportTasks={handleExportTasks}
        onClearFilters={() => {
          setSearchQuery('');
          setSelectedRepository('all');
        }}
        searchSuggestions={searchSuggestions}
        showSearchSuggestions={showSearchSuggestions}
        setShowSearchSuggestions={setShowSearchSuggestions}
      />
      
      {/* Panel de estadísticas */}
      {showStats && (
        <div className="bg-white border-b border-gray-200 px-4 py-3">
        <div className="flex items-center gap-3 flex-wrap">
          <div className="flex-1 relative min-w-[200px]">
            <svg className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <div className="relative w-full">
              <input
                type="text"
                placeholder="Buscar tareas... (usa 'repo:', 'status:', 'error:' para búsqueda avanzada)"
                value={searchQuery}
                onChange={(e) => {
                  setSearchQuery(e.target.value);
                  setShowSearchSuggestions(true);
                }}
                onFocus={() => setShowSearchSuggestions(true)}
                onBlur={() => setTimeout(() => setShowSearchSuggestions(false), 200)}
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                aria-label="Buscar tareas"
                title="Búsqueda avanzada: usa 'repo:nombre', 'status:completed', 'error:texto', 'model:nombre' para filtrar"
              />
              {(searchQuery.startsWith('repo:') || searchQuery.startsWith('status:') || searchQuery.startsWith('error:') || searchQuery.startsWith('model:')) && (
                <div className="absolute right-12 top-1/2 transform -translate-y-1/2">
                  <span className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded font-medium">
                    Avanzada
                  </span>
                </div>
              )}
              
              {/* Sugerencias de búsqueda */}
              {showSearchSuggestions && searchSuggestions.length > 0 && (
                <div className="absolute z-50 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg max-h-60 overflow-y-auto">
                  {searchSuggestions.map((suggestion, idx) => (
                    <button
                      key={idx}
                      onClick={() => {
                        setSearchQuery(suggestion);
                        setShowSearchSuggestions(false);
                      }}
                      className="w-full text-left px-4 py-2 hover:bg-gray-100 transition-colors text-sm"
                    >
                      <span className="text-gray-600">{suggestion}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
            {searchQuery && (
              <button
                onClick={() => setSearchQuery('')}
                className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600"
                aria-label="Limpiar búsqueda"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            )}
          </div>
          
          {/* Filtro por repositorio */}
          {repositories.length > 0 && (
            <select
              value={selectedRepository}
              onChange={(e) => setSelectedRepository(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
            >
              <option value="all">Todos los repositorios</option>
              {repositories.map((repo) => (
                <option key={repo} value={repo}>{repo}</option>
              ))}
            </select>
          )}
          
          {/* Filtro por modelo */}
          {models.length > 0 && (
            <select
              value={modelFilter}
              onChange={(e) => setModelFilter(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
            >
              <option value="all">Todos los modelos</option>
              {models.map((model) => (
                <option key={model} value={model}>{model}</option>
              ))}
            </select>
          )}
          
          {/* Filtro por estado */}
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value as TaskStatus | 'all')}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
          >
            <option value="all">Todos los estados</option>
            {KANBAN_COLUMNS.map((col) => (
              <option key={col.id} value={col.id}>{col.label}</option>
            ))}
          </select>
          
          {/* Filtro por fecha */}
          <select
            value={dateFilter}
            onChange={(e) => setDateFilter(e.target.value as 'all' | 'today' | 'week' | 'month' | 'custom')}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
          >
            <option value="all">Todas las fechas</option>
            <option value="today">Hoy</option>
            <option value="week">Esta semana</option>
            <option value="month">Este mes</option>
            <option value="custom">Rango personalizado</option>
          </select>
          
          {/* Filtros de rango de fechas personalizado */}
          {dateFilter === 'custom' && (
            <div className="flex items-center gap-2">
              <input
                type="date"
                value={(() => {
                  if (typeof window !== 'undefined') {
                    const saved = localStorage.getItem('kanban-date-from');
                    return saved || '';
                  }
                  return '';
                })()}
                onChange={(e) => {
                  if (typeof window !== 'undefined') {
                    localStorage.setItem('kanban-date-from', e.target.value);
                  }
                  // Forzar re-render
                  setDateFilter('custom');
                }}
                className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
                title="Fecha desde"
              />
              <span className="text-gray-500">-</span>
              <input
                type="date"
                value={(() => {
                  if (typeof window !== 'undefined') {
                    const saved = localStorage.getItem('kanban-date-to');
                    return saved || '';
                  }
                  return '';
                })()}
                onChange={(e) => {
                  if (typeof window !== 'undefined') {
                    localStorage.setItem('kanban-date-to', e.target.value);
                  }
                  // Forzar re-render
                  setDateFilter('custom');
                }}
                className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
                title="Fecha hasta"
              />
            </div>
          )}
          
          {/* Ordenamiento (solo en vista lista) */}
          {viewMode === 'list' && (
            <div className="flex items-center gap-2">
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value as 'date' | 'status' | 'repo' | 'model')}
                className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
              >
                <option value="date">Ordenar por fecha</option>
                <option value="status">Ordenar por estado</option>
                <option value="repo">Ordenar por repositorio</option>
              </select>
              <button
                onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
                className="p-2 border border-gray-300 rounded-lg hover:bg-gray-100 transition-colors"
                title={sortOrder === 'asc' ? 'Orden ascendente' : 'Orden descendente'}
              >
                {sortOrder === 'asc' ? (
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
                  </svg>
                ) : (
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                )}
              </button>
            </div>
          )}
          
          {/* Toggle vista */}
          <div className="flex items-center gap-1 bg-gray-100 rounded-lg p-1 border border-gray-300">
            <button
              onClick={() => setViewMode('kanban')}
              className={cn(
                "px-3 py-1.5 rounded text-sm font-medium transition-colors",
                viewMode === 'kanban' 
                  ? "bg-white text-gray-900 shadow-sm" 
                  : "text-gray-600 hover:text-gray-900"
              )}
              title="Vista Kanban"
            >
              <svg className="w-4 h-4 inline mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2" />
              </svg>
              Kanban
            </button>
            <button
              onClick={() => setViewMode('list')}
              className={cn(
                "px-3 py-1.5 rounded text-sm font-medium transition-colors",
                viewMode === 'list' 
                  ? "bg-white text-gray-900 shadow-sm" 
                  : "text-gray-600 hover:text-gray-900"
              )}
              title="Vista Lista"
            >
              <svg className="w-4 h-4 inline mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
              Lista
            </button>
            <button
              onClick={() => setViewMode('timeline')}
              className={cn(
                "px-3 py-1.5 rounded text-sm font-medium transition-colors",
                viewMode === 'timeline' 
                  ? "bg-white text-gray-900 shadow-sm" 
                  : "text-gray-600 hover:text-gray-900"
              )}
              title="Vista Timeline"
            >
              <svg className="w-4 h-4 inline mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              Timeline
            </button>
          </div>
          
          {/* Resumen por repositorio */}
          <button
            onClick={() => setShowRepoSummary(!showRepoSummary)}
            className={cn(
              "px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2",
              showRepoSummary 
                ? "bg-blue-100 text-blue-700 border border-blue-300" 
                : "bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300"
            )}
            title="Resumen por repositorio"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            Repos
          </button>
          
          {/* Actividad reciente */}
          <button
            onClick={() => setShowRecentActivity(!showRecentActivity)}
            className={cn(
              "px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2 relative",
              showRecentActivity 
                ? "bg-blue-100 text-blue-700 border border-blue-300" 
                : "bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300"
            )}
            title="Actividad reciente"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Actividad
            {recentActivity.length > 0 && (
              <span className="absolute -top-1 -right-1 w-5 h-5 bg-blue-500 text-white text-xs rounded-full flex items-center justify-center">
                {recentActivity.length > 9 ? '9+' : recentActivity.length}
              </span>
            )}
          </button>
          
          {/* Tamaño de tarjeta (solo en vista kanban) */}
          {viewMode === 'kanban' && (
            <div className="flex items-center gap-1 bg-gray-100 rounded-lg p-1 border border-gray-300">
              <button
                onClick={() => setCardSize('compact')}
                className={cn(
                  "px-2 py-1 rounded text-xs transition-colors",
                  cardSize === 'compact' 
                    ? "bg-white text-gray-900 shadow-sm" 
                    : "text-gray-600 hover:text-gray-900"
                )}
                title="Vista compacta"
              >
                <svg className="w-3 h-3 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              </button>
              <button
                onClick={() => setCardSize('normal')}
                className={cn(
                  "px-2 py-1 rounded text-xs transition-colors",
                  cardSize === 'normal' 
                    ? "bg-white text-gray-900 shadow-sm" 
                    : "text-gray-600 hover:text-gray-900"
                )}
                title="Vista normal"
              >
                <svg className="w-3 h-3 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h4a1 1 0 011 1v7a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM14 5a1 1 0 011-1h4a1 1 0 011 1v7a1 1 0 01-1 1h-4a1 1 0 01-1-1V5zM4 16a1 1 0 011-1h4a1 1 0 011 1v3a1 1 0 01-1 1H5a1 1 0 01-1-1v-3zM14 16a1 1 0 011-1h4a1 1 0 011 1v3a1 1 0 01-1 1h-4a1 1 0 01-1-1v-3z" />
                </svg>
              </button>
              <button
                onClick={() => setCardSize('expanded')}
                className={cn(
                  "px-2 py-1 rounded text-xs transition-colors",
                  cardSize === 'expanded' 
                    ? "bg-white text-gray-900 shadow-sm" 
                    : "text-gray-600 hover:text-gray-900"
                )}
                title="Vista expandida"
              >
                <svg className="w-3 h-3 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z" />
                </svg>
              </button>
            </div>
          )}
          
          {/* Filtros guardados */}
          {savedFilters.length > 0 && (
            <div className="relative group">
              <button
                className="px-3 py-2 rounded-lg text-sm font-medium bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300 transition-colors flex items-center gap-2"
                title="Filtros guardados"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" />
                </svg>
                Filtros ({savedFilters.length})
              </button>
              <div className="absolute right-0 top-full mt-2 w-64 bg-white border border-gray-200 rounded-lg shadow-lg z-50 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all">
                <div className="p-2 max-h-96 overflow-y-auto">
                  {savedFilters.map((filter, index) => (
                    <div key={index} className="flex items-center justify-between p-2 hover:bg-gray-50 rounded">
                      <button
                        onClick={() => handleApplySavedFilter(filter)}
                        className="flex-1 text-left text-sm text-gray-700 hover:text-blue-600"
                      >
                        {filter.name}
                      </button>
                      <button
                        onClick={() => handleDeleteSavedFilter(index)}
                        className="p-1 text-red-600 hover:text-red-800 hover:bg-red-50 rounded"
                        title="Eliminar filtro"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                      </button>
                    </div>
                  ))}
                </div>
                <div className="border-t border-gray-200 p-2">
                  <button
                    onClick={handleSaveCurrentFilters}
                    className="w-full px-3 py-2 text-sm text-blue-600 hover:bg-blue-50 rounded-lg text-center"
                  >
                    + Guardar filtros actuales
                  </button>
                </div>
              </div>
            </div>
          )}
          
          {savedFilters.length === 0 && (
            <button
              onClick={handleSaveCurrentFilters}
              className="px-3 py-2 rounded-lg text-sm font-medium bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300 transition-colors flex items-center gap-2"
              title="Guardar filtros actuales"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" />
              </svg>
              Guardar filtros
            </button>
          )}
          
          {/* Botones de exportar */}
          <div className="flex items-center gap-1">
            <button
              onClick={handleExportTasks}
              className="px-3 py-2 rounded-lg text-sm font-medium bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300 transition-colors flex items-center gap-2"
              title="Exportar tareas a JSON"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              JSON
            </button>
            <button
              onClick={handleExportToCSV}
              className="px-3 py-2 rounded-lg text-sm font-medium bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300 transition-colors flex items-center gap-2"
              title="Exportar tareas a CSV"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              CSV
            </button>
          </div>
          
          {/* Botones de agrupar */}
          <div className="flex items-center gap-1">
            <button
              onClick={() => {
                setGroupByRepo(!groupByRepo);
                if (!groupByRepo) setGroupByModel(false);
              }}
              className={cn(
                "px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2",
                groupByRepo 
                  ? "bg-blue-100 text-blue-700 border border-blue-300" 
                  : "bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300"
              )}
              title="Agrupar tareas por repositorio"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
              </svg>
              Repo
            </button>
            <button
              onClick={() => {
                setGroupByModel(!groupByModel);
                if (!groupByModel) setGroupByRepo(false);
              }}
              className={cn(
                "px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2",
                groupByModel 
                  ? "bg-purple-100 text-purple-700 border border-purple-300" 
                  : "bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300"
              )}
              title="Agrupar tareas por modelo"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
              Modelo
            </button>
          </div>
          
          {/* Botón de estadísticas */}
          <button
            onClick={() => setShowStats(!showStats)}
            className={cn(
              "px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2",
              showStats 
                ? "bg-blue-100 text-blue-700 border border-blue-300" 
                : "bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300"
            )}
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            Estadísticas
          </button>
          
          {/* Auto-refresh toggle */}
          <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-100 border border-gray-300">
            <label className="flex items-center gap-2 text-sm cursor-pointer">
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
                className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
              />
              <span className="text-gray-700">Auto-refresh</span>
            </label>
            {autoRefresh && (
              <select
                value={refreshInterval}
                onChange={(e) => setRefreshInterval(Number(e.target.value))}
                className="ml-2 px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                onClick={(e) => e.stopPropagation()}
              >
                <option value={10}>10s</option>
                <option value={30}>30s</option>
                <option value={60}>1m</option>
                <option value={300}>5m</option>
              </select>
            )}
          </div>
          
          {/* Botón de notificaciones */}
          <div className="relative">
            <button
              onClick={() => setShowNotifications(!showNotifications)}
              className="relative px-3 py-2 rounded-lg text-sm font-medium bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300 transition-colors flex items-center gap-2"
              title="Notificaciones"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
              </svg>
              {unreadCount > 0 && (
                <span className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 text-white text-xs rounded-full flex items-center justify-center">
                  {unreadCount > 9 ? '9+' : unreadCount}
                </span>
              )}
            </button>
            
            {/* Panel de notificaciones */}
            {showNotifications && (
              <NotificationsPanel
                notifications={notifications}
                unreadCount={unreadCount}
                darkMode={darkMode}
                onMarkAllAsRead={markAllAsRead}
                onClearAll={clearNotifications}
                onClose={() => setShowNotifications(false)}
              />
            )}
        </div>
          
          {/* Filtro por modelo */}
          {models.length > 0 && (
            <select
              value={modelFilter}
              onChange={(e) => setModelFilter(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
            >
              <option value="all">Todos los modelos</option>
              {models.map((model) => (
                <option key={model} value={model}>{model}</option>
              ))}
            </select>
          )}
          
              {/* Ayuda de atajos mejorada */}
          <div className="relative group">
            <div className="text-xs text-gray-500 flex items-center gap-1 cursor-help">
              <kbd className="px-1.5 py-0.5 bg-gray-100 border border-gray-300 rounded text-xs">?</kbd>
              <span>Atajos</span>
        </div>
            <div className="absolute right-0 top-full mt-2 w-64 bg-white border border-gray-200 rounded-lg shadow-xl z-50 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all p-3">
              <div className="space-y-2 text-xs">
                <div className="flex items-center justify-between">
                  <span className="text-gray-600">Buscar</span>
                  <kbd className="px-1.5 py-0.5 bg-gray-100 border border-gray-300 rounded">/</kbd>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-600">Crear tarea</span>
                  <kbd className="px-1.5 py-0.5 bg-gray-100 border border-gray-300 rounded">c</kbd>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-600">Búsqueda avanzada</span>
                  <kbd className="px-1.5 py-0.5 bg-gray-100 border border-gray-300 rounded">Ctrl+K</kbd>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-600">Seleccionar tarea</span>
                  <kbd className="px-1.5 py-0.5 bg-gray-100 border border-gray-300 rounded">Ctrl+D</kbd>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-600">Comparar tareas</span>
                  <kbd className="px-1.5 py-0.5 bg-gray-100 border border-gray-300 rounded">Ctrl+Shift+C</kbd>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-600">Toggle métricas</span>
                  <kbd className="px-1.5 py-0.5 bg-gray-100 border border-gray-300 rounded">Ctrl+M</kbd>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-600">Modo compacto</span>
                  <kbd className="px-1.5 py-0.5 bg-gray-100 border border-gray-300 rounded">Ctrl+,</kbd>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-600">Cerrar panel</span>
                  <kbd className="px-1.5 py-0.5 bg-gray-100 border border-gray-300 rounded">Esc</kbd>
                </div>
              </div>
            </div>
          </div>
          
          {/* Toggle modo compacto */}
          <button
            onClick={() => setCompactMode(!compactMode)}
            className={cn(
              "px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2",
              compactMode 
                ? "bg-blue-100 text-blue-700 border border-blue-300" 
                : "bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300"
            )}
            title="Modo compacto (Ctrl+,)"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
            Compacto
          </button>
          
          {/* Toggle métricas */}
          <button
            onClick={() => setShowMetrics(!showMetrics)}
            className={cn(
              "px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2",
              showMetrics 
                ? "bg-green-100 text-green-700 border border-green-300" 
                : "bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300"
            )}
            title="Métricas en tiempo real (Ctrl+M)"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
            </svg>
            Métricas
          </button>
          
          {/* Búsqueda avanzada */}
          <button
            onClick={() => setShowAdvancedSearch(!showAdvancedSearch)}
            className={cn(
              "px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2",
              showAdvancedSearch 
                ? "bg-purple-100 text-purple-700 border border-purple-300" 
                : darkMode
                  ? "bg-gray-700 text-gray-200 hover:bg-gray-600 border border-gray-600"
                  : "bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300"
            )}
            title="Búsqueda avanzada (Ctrl+K)"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM13 10h3m-3 4h3m-6-4h.01M11 14h.01" />
            </svg>
            Avanzada
          </button>
          
          {/* Búsqueda en streaming */}
          <label className={cn(
            "flex items-center gap-2 px-3 py-2 rounded-lg text-sm cursor-pointer",
            darkMode
              ? "bg-gray-700 text-gray-200 hover:bg-gray-600 border border-gray-600"
              : "bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300"
          )}>
            <input
              type="checkbox"
              checked={searchInStreaming}
              onChange={(e) => setSearchInStreaming(e.target.checked)}
              className="w-4 h-4"
            />
            <span>Buscar en streaming</span>
          </label>
          
          {/* Filtro por prioridad */}
          <select
            value={priorityFilter}
            onChange={(e) => setPriorityFilter(e.target.value)}
            className={cn(
              "px-3 py-2 rounded-lg text-sm",
              darkMode
                ? "bg-gray-700 text-white border border-gray-600"
                : "bg-white border border-gray-300"
            )}
          >
            <option value="all">Todas las prioridades</option>
            <option value="urgent">Urgente</option>
            <option value="high">Alta</option>
            <option value="medium">Media</option>
            <option value="low">Baja</option>
          </select>
          
          {/* Vista de red */}
          <button
            onClick={() => setShowNetworkView(!showNetworkView)}
            className={cn(
              "px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2",
              showNetworkView 
                ? "bg-indigo-100 text-indigo-700 border border-indigo-300" 
                : darkMode
                  ? "bg-gray-700 text-gray-200 hover:bg-gray-600 border border-gray-600"
                  : "bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300"
            )}
            title="Vista de red de tareas"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9" />
            </svg>
            Red
          </button>
          
          {/* Heatmap */}
          <button
            onClick={() => setShowHeatmap(!showHeatmap)}
            className={cn(
              "px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2",
              showHeatmap 
                ? "bg-orange-100 text-orange-700 border border-orange-300" 
                : darkMode
                  ? "bg-gray-700 text-gray-200 hover:bg-gray-600 border border-gray-600"
                  : "bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300"
            )}
            title="Vista de heatmap"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h4a1 1 0 011 1v7a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM14 5a1 1 0 011-1h4a1 1 0 011 1v7a1 1 0 01-1 1h-4a1 1 0 01-1-1V5zM4 16a1 1 0 011-1h4a1 1 0 011 1v3a1 1 0 01-1 1H5a1 1 0 01-1-1v-3zM14 16a1 1 0 011-1h4a1 1 0 011 1v3a1 1 0 01-1 1h-4a1 1 0 01-1-1v-3z" />
            </svg>
            Heatmap
          </button>
          
          {/* Milestones */}
          {milestones.length > 0 && (
            <button
              onClick={() => setShowMilestones(!showMilestones)}
              className={cn(
                "px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2 relative",
                showMilestones 
                  ? "bg-green-100 text-green-700 border border-green-300" 
                  : darkMode
                    ? "bg-gray-700 text-gray-200 hover:bg-gray-600 border border-gray-600"
                    : "bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300"
              )}
              title="Milestones"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z" />
              </svg>
              Milestones
              <span className="absolute -top-1 -right-1 w-5 h-5 bg-green-500 text-white text-xs rounded-full flex items-center justify-center">
                {milestones.length}
              </span>
            </button>
          )}
          
          {/* Modo presentación */}
          <button
            onClick={() => setPresentationMode(!presentationMode)}
            className={cn(
              "px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2",
              presentationMode 
                ? "bg-red-100 text-red-700 border border-red-300" 
                : darkMode
                  ? "bg-gray-700 text-gray-200 hover:bg-gray-600 border border-gray-600"
                  : "bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300"
            )}
            title="Modo presentación (Ctrl+P)"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
            Presentación
          </button>
          
          {/* Gráficos avanzados */}
          <button
            onClick={() => setShowCharts(!showCharts)}
            className={cn(
              "px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2",
              showCharts 
                ? "bg-pink-100 text-pink-700 border border-pink-300" 
                : darkMode
                  ? "bg-gray-700 text-gray-200 hover:bg-gray-600 border border-gray-600"
                  : "bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300"
            )}
            title="Gráficos avanzados (Ctrl+G)"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
            </svg>
            Gráficos
          </button>
          
          {/* Vista de dependencias */}
          {taskDependencies.size > 0 && (
            <button
              onClick={() => setShowDependencies(!showDependencies)}
              className={cn(
                "px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2",
                showDependencies 
                  ? "bg-teal-100 text-teal-700 border border-teal-300" 
                  : darkMode
                    ? "bg-gray-700 text-gray-200 hover:bg-gray-600 border border-gray-600"
                    : "bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300"
              )}
              title="Vista de dependencias"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
              </svg>
              Dependencias
            </button>
          )}
          
          {/* Vista de progreso */}
          <button
            onClick={() => setShowProgressView(!showProgressView)}
            className={cn(
              "px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2",
              showProgressView 
                ? "bg-cyan-100 text-cyan-700 border border-cyan-300" 
                : darkMode
                  ? "bg-gray-700 text-gray-200 hover:bg-gray-600 border border-gray-600"
                  : "bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300"
            )}
            title="Vista de progreso"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            Progreso
          </button>
          
          {/* Resumen ejecutivo */}
          <button
            onClick={() => setShowExecutiveSummary(!showExecutiveSummary)}
            className={cn(
              "px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2",
              showExecutiveSummary 
                ? "bg-amber-100 text-amber-700 border border-amber-300" 
                : darkMode
                  ? "bg-gray-700 text-gray-200 hover:bg-gray-600 border border-gray-600"
                  : "bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300"
            )}
            title="Resumen ejecutivo"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            Resumen
          </button>
          
          {/* Compartir vista */}
          <button
            onClick={() => setShowShareDialog(true)}
            className={cn(
              "px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2",
              darkMode
                ? "bg-gray-700 text-gray-200 hover:bg-gray-600 border border-gray-600"
                : "bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300"
            )}
            title="Compartir vista (Ctrl+Shift+S)"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z" />
            </svg>
            Compartir
          </button>
          
          {/* Sonido */}
          <label className={cn(
            "flex items-center gap-2 px-3 py-2 rounded-lg text-sm cursor-pointer",
            soundEnabled
              ? "bg-green-100 text-green-700 border border-green-300"
              : darkMode
                ? "bg-gray-700 text-gray-200 hover:bg-gray-600 border border-gray-600"
                : "bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300"
          )}>
            <input
              type="checkbox"
              checked={soundEnabled}
              onChange={(e) => setSoundEnabled(e.target.checked)}
              className="w-4 h-4"
            />
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
            </svg>
            <span>Sonido</span>
          </label>
          
          {/* Filtro por error */}
          <select
            value={errorFilter}
            onChange={(e) => setErrorFilter(e.target.value)}
            className={cn(
              "px-3 py-2 rounded-lg text-sm",
              darkMode
                ? "bg-gray-700 text-white border border-gray-600"
                : "bg-white border border-gray-300"
            )}
          >
            <option value="all">Todos los errores</option>
            <option value="with-errors">Con errores</option>
            <option value="no-errors">Sin errores</option>
          </select>
          
          {/* Alertas inteligentes */}
          {smartAlerts.length > 0 && (
            <button
              onClick={() => setShowSmartAlerts(!showSmartAlerts)}
              className={cn(
                "px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2 relative",
                showSmartAlerts 
                  ? "bg-yellow-100 text-yellow-700 border border-yellow-300" 
                  : darkMode
                    ? "bg-gray-700 text-gray-200 hover:bg-gray-600 border border-gray-600"
                    : "bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300"
              )}
              title="Alertas inteligentes"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
              Alertas
              <span className="absolute -top-1 -right-1 w-5 h-5 bg-yellow-500 text-white text-xs rounded-full flex items-center justify-center">
                {smartAlerts.length}
              </span>
            </button>
          )}
          
          {/* Feed de actividad */}
          <button
            onClick={() => setShowActivityFeed(!showActivityFeed)}
            className={cn(
              "px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2",
              showActivityFeed 
                ? "bg-emerald-100 text-emerald-700 border border-emerald-300" 
                : darkMode
                  ? "bg-gray-700 text-gray-200 hover:bg-gray-600 border border-gray-600"
                  : "bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300"
            )}
            title="Feed de actividad"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            Actividad
          </button>
          
          {/* Plantillas */}
          <button
            onClick={() => setShowTemplates(!showTemplates)}
            className={cn(
              "px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2",
              showTemplates 
                ? "bg-violet-100 text-violet-700 border border-violet-300" 
                : darkMode
                  ? "bg-gray-700 text-gray-200 hover:bg-gray-600 border border-gray-600"
                  : "bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300"
            )}
            title="Plantillas de tareas"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            Plantillas
          </button>
          
          {/* Colaboración */}
          <button
            onClick={() => setShowCollaboration(!showCollaboration)}
            className={cn(
              "px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2",
              showCollaboration 
                ? "bg-sky-100 text-sky-700 border border-sky-300" 
                : darkMode
                  ? "bg-gray-700 text-gray-200 hover:bg-gray-600 border border-gray-600"
                  : "bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300"
            )}
            title="Colaboración en tiempo real"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
            </svg>
            Colaboración
          </button>
          
          {/* Dashboard personalizado */}
          <button
            onClick={() => setShowCustomDashboard(!showCustomDashboard)}
            className={cn(
              "px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2",
              showCustomDashboard 
                ? "bg-rose-100 text-rose-700 border border-rose-300" 
                : darkMode
                  ? "bg-gray-700 text-gray-200 hover:bg-gray-600 border border-gray-600"
                  : "bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300"
            )}
            title="Dashboard personalizado"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h4a1 1 0 011 1v7a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM14 5a1 1 0 011-1h4a1 1 0 011 1v7a1 1 0 01-1 1h-4a1 1 0 01-1-1V5zM4 16a1 1 0 011-1h4a1 1 0 011 1v3a1 1 0 01-1 1H5a1 1 0 01-1-1v-3zM14 16a1 1 0 011-1h4a1 1 0 011 1v3a1 1 0 01-1 1h-4a1 1 0 01-1-1v-3z" />
            </svg>
            Dashboard
          </button>
          
          {/* Filtros guardados */}
          {savedFilters.length > 0 && (
            <button
              onClick={() => setShowSavedFilters(!showSavedFilters)}
              className={cn(
                "px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2",
                showSavedFilters 
                  ? "bg-lime-100 text-lime-700 border border-lime-300" 
                  : darkMode
                    ? "bg-gray-700 text-gray-200 hover:bg-gray-600 border border-gray-600"
                    : "bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300"
              )}
              title="Filtros guardados"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" />
              </svg>
              Filtros
            </button>
          )}
          
          {/* Logs de auditoría */}
          <button
            onClick={() => setShowAuditLogs(!showAuditLogs)}
            className={cn(
              "px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2",
              showAuditLogs 
                ? "bg-slate-100 text-slate-700 border border-slate-300" 
                : darkMode
                  ? "bg-gray-700 text-gray-200 hover:bg-gray-600 border border-gray-600"
                  : "bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300"
            )}
            title="Logs de auditoría"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            Logs
          </button>
          
          {/* Backup/Restore */}
          <button
            onClick={() => setShowBackupRestore(!showBackupRestore)}
            className={cn(
              "px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2",
              showBackupRestore 
                ? "bg-fuchsia-100 text-fuchsia-700 border border-fuchsia-300" 
                : darkMode
                  ? "bg-gray-700 text-gray-200 hover:bg-gray-600 border border-gray-600"
                  : "bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300"
            )}
            title="Backup y Restore"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
            </svg>
            Backup
          </button>
          
          {/* Acciones rápidas */}
          <button
            onClick={() => setShowQuickActions(!showQuickActions)}
            className={cn(
              "px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2",
              showQuickActions 
                ? "bg-orange-100 text-orange-700 border border-orange-300" 
                : darkMode
                  ? "bg-gray-700 text-gray-200 hover:bg-gray-600 border border-gray-600"
                  : "bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300"
            )}
            title="Acciones rápidas"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            Acciones
          </button>
          
          {/* Operaciones masivas */}
          {selectedTasks.size > 0 && (
            <button
              onClick={() => setShowBulkMenu(!showBulkMenu)}
              className={cn(
                "px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2",
                darkMode
                  ? "bg-gray-700 text-gray-200 hover:bg-gray-600 border border-gray-600"
                  : "bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300"
              )}
              title={`Operaciones masivas (${selectedTasks.size} seleccionadas)`}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
              </svg>
              Masivas ({selectedTasks.size})
            </button>
          )}
          
          {/* Comparar tareas */}
          {selectedTasks.size >= 2 && (
            <button
              onClick={() => setShowTaskComparison(true)}
              className="px-3 py-2 rounded-lg text-sm font-medium bg-indigo-100 text-indigo-700 hover:bg-indigo-200 border border-indigo-300 transition-colors flex items-center gap-2"
              title={`Comparar ${selectedTasks.size} tareas (Ctrl+Shift+C)`}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              Comparar ({selectedTasks.size})
            </button>
          )}
        </div>
        </div>
      )}
      
      {/* Panel de estadísticas con gráficos */}
      <StatsPanel stats={stats} showStats={showStats} />

      {/* Sección de Agentes Continuos */}
      <div className="px-4 py-2">
        <AgentsSection />
      </div>

      {/* Resumen por repositorio */}
      <RepoSummary
        repositories={repositories}
        tasks={tasks}
        onSelectRepository={(repo) => {
          setSelectedRepository(repo);
          setShowRepoSummary(false);
        }}
        showRepoSummary={showRepoSummary}
      />
      
      {/* Actividad reciente */}
      <RecentActivity
        recentActivity={recentActivity}
        onSelectTask={setSelectedTask}
        showRecentActivity={showRecentActivity}
      />

      {/* Kanban Board, Lista, Timeline o Calendario */}
      <main className="p-4 overflow-x-auto">
        {viewMode === 'calendar' ? (
          <CalendarView
            tasks={tasksWithStreamingSearch}
            onSelectTask={setSelectedTask}
            month={calendarMonth}
            year={calendarYear}
            onMonthChange={(month, year) => {
              setCalendarMonth(month);
              setCalendarYear(year);
            }}
          />
        ) : viewMode === 'timeline' ? (
          <TimelineView
            tasks={tasksWithStreamingSearch}
            onSelectTask={setSelectedTask}
            onQuickView={setQuickViewTask}
          />
        ) : viewMode === 'list' ? (
          <ListView
            tasks={tasksWithStreamingSearch}
            allTasksCount={tasks.length}
            onSelectTask={setSelectedTask}
            onDeleteTask={handleDeleteTask}
            sortBy={sortBy}
            setSortBy={setSortBy}
            sortOrder={sortOrder}
            setSortOrder={setSortOrder}
          />
        ) : (
          <KanbanBoard
            tasksByColumn={tasksWithErrorFilter.reduce((acc, task) => {
              const status = task.status as TaskStatus;
              if (!acc[status]) acc[status] = [];
              acc[status].push(task);
              return acc;
            }, {} as Record<TaskStatus, Task[]>)}
            cardSize={compactMode ? 'compact' : cardSize}
            groupByRepo={groupByRepo}
            groupByModel={groupByModel}
            activeId={activeId}
            dragOverColumn={dragOverColumn}
            onDragStart={handleDragStart}
            onDragOver={handleDragOver}
            onDragEnd={handleDragEnd}
            onDeleteTask={handleDeleteTask}
            onSelectTask={(task) => {
              // Si Ctrl/Cmd está presionado, agregar a selección múltiple
              if (window.event && (window.event as KeyboardEvent).ctrlKey || (window.event as KeyboardEvent).metaKey) {
                setSelectedTasks(prev => {
                  const newSet = new Set(prev);
                  if (newSet.has(task.id)) {
                    newSet.delete(task.id);
                  } else {
                    newSet.add(task.id);
                  }
                  return newSet;
                });
              } else {
                setSelectedTask(task);
              }
            }}
            onQuickView={setQuickViewTask}
            allTasks={tasks}
            selectedTasks={selectedTasks}
            taskTags={taskTags}
          />
        )}
        
        {/* Quick View en hover */}
        <QuickView task={quickViewTask} />
      </main>

      {/* Panel lateral de detalles */}
      <TaskDetailPanel
        task={selectedTask}
        onClose={() => setSelectedTask(null)}
        onDelete={handleDeleteTask}
      />
      
      {/* Panel de comentarios para tarea seleccionada */}
      {selectedTask && (
        <div className={cn(
          "fixed right-0 top-0 h-full w-96 z-40 shadow-xl overflow-y-auto",
          darkMode ? "bg-gray-800 border-l border-gray-700" : "bg-white border-l border-gray-200"
        )}>
          <div className="p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-bold">Comentarios</h3>
              <button
                onClick={() => setShowComments(prev => {
                  const newMap = new Map(prev);
                  newMap.set(selectedTask.id, !newMap.get(selectedTask.id));
                  return newMap;
                })}
                className="text-gray-400 hover:text-gray-600"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            
            <div className="space-y-4">
              {(taskComments.get(selectedTask.id) || []).map(comment => (
                <div
                  key={comment.id}
                  className={cn(
                    "p-3 rounded-lg",
                    darkMode ? "bg-gray-700" : "bg-gray-50"
                  )}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-semibold">{comment.author}</span>
                    <span className="text-xs text-gray-500">
                      {format(comment.timestamp, 'PPp', { locale: es })}
                    </span>
                  </div>
                  <p className="text-sm">{comment.text}</p>
                </div>
              ))}
              
              <div className="mt-4">
                <textarea
                  placeholder="Agregar comentario..."
                  className={cn(
                    "w-full px-3 py-2 rounded-lg text-sm resize-none",
                    darkMode 
                      ? "bg-gray-700 text-white border border-gray-600"
                      : "bg-white border border-gray-300"
                  )}
                  rows={3}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                      const text = e.currentTarget.value.trim();
                      if (text) {
                        setTaskComments(prev => {
                          const newMap = new Map(prev);
                          const comments = newMap.get(selectedTask.id) || [];
                          newMap.set(selectedTask.id, [
                            ...comments,
                            {
                              id: `comment-${Date.now()}`,
                              text,
                              timestamp: new Date(),
                              author: 'Usuario'
                            }
                          ]);
                          return newMap;
                        });
                        e.currentTarget.value = '';
                        addNotification('info', 'Comentario agregado', 'Tu comentario ha sido guardado');
                      }
                    }
                  }}
                />
                <p className="text-xs text-gray-500 mt-1">Ctrl+Enter para enviar</p>
              </div>
            </div>
            
            {/* Prioridad de la tarea */}
            <div className="mt-6 pt-6 border-t border-gray-200 dark:border-gray-700">
              <label className="block text-sm font-medium mb-2">Prioridad</label>
              <select
                value={taskPriorities.get(selectedTask.id) || 'medium'}
                onChange={(e) => {
                  setTaskPriorities(prev => {
                    const newMap = new Map(prev);
                    newMap.set(selectedTask.id, e.target.value as 'low' | 'medium' | 'high' | 'urgent');
                    return newMap;
                  });
                }}
                className={cn(
                  "w-full px-3 py-2 rounded-lg text-sm",
                  darkMode 
                    ? "bg-gray-700 text-white border border-gray-600"
                    : "bg-white border border-gray-300"
                )}
              >
                <option value="low">Baja</option>
                <option value="medium">Media</option>
                <option value="high">Alta</option>
                <option value="urgent">Urgente</option>
              </select>
            </div>
            
            {/* Agregar recordatorio */}
            <div className="mt-6 pt-6 border-t border-gray-200 dark:border-gray-700">
              <label className="block text-sm font-medium mb-2">Recordatorio</label>
              <input
                type="datetime-local"
                className={cn(
                  "w-full px-3 py-2 rounded-lg text-sm",
                  darkMode 
                    ? "bg-gray-700 text-white border border-gray-600"
                    : "bg-white border border-gray-300"
                )}
                onChange={(e) => {
                  if (e.target.value) {
                    const reminderTime = new Date(e.target.value);
                    setReminders(prev => [...prev, {
                      id: `reminder-${Date.now()}`,
                      taskId: selectedTask.id,
                      message: `Recordatorio para: ${selectedTask.instruction.substring(0, 50)}...`,
                      time: reminderTime
                    }]);
                    addNotification('info', 'Recordatorio creado', `Te recordaremos el ${format(reminderTime, 'PPp', { locale: es })}`);
                    e.target.value = '';
                  }
                }}
              />
            </div>
          </div>
        </div>
      )}
      
      {/* Panel de recordatorios */}
      {reminders.length > 0 && (
        <div className="fixed bottom-4 left-4 z-50">
          <button
            onClick={() => setShowReminders(!showReminders)}
            className={cn(
              "px-4 py-2 rounded-lg shadow-lg flex items-center gap-2 relative",
              darkMode 
                ? "bg-gray-800 text-white border border-gray-700"
                : "bg-white text-gray-700 border border-gray-200"
            )}
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Recordatorios
            <span className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 text-white text-xs rounded-full flex items-center justify-center">
              {reminders.filter(r => new Date(r.time) > new Date()).length}
            </span>
          </button>
          
          {showReminders && (
            <div className={cn(
              "absolute bottom-full left-0 mb-2 w-80 rounded-lg shadow-xl p-4",
              darkMode ? "bg-gray-800 border border-gray-700" : "bg-white border border-gray-200"
            )}>
              <h3 className="font-semibold mb-3">Recordatorios Activos</h3>
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {reminders
                  .filter(r => new Date(r.time) > new Date())
                  .sort((a, b) => a.time.getTime() - b.time.getTime())
                  .map(reminder => {
                    const task = tasks.find(t => t.id === reminder.taskId);
                    return (
                      <div
                        key={reminder.id}
                        className={cn(
                          "p-3 rounded-lg text-sm",
                          darkMode ? "bg-gray-700" : "bg-gray-50"
                        )}
                      >
                        <div className="flex items-center justify-between mb-1">
                          <span className="font-semibold">{task?.repository || 'Tarea'}</span>
                          <button
                            onClick={() => {
                              setReminders(prev => prev.filter(r => r.id !== reminder.id));
                            }}
                            className="text-red-500 hover:text-red-700"
                          >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                            </svg>
                          </button>
                        </div>
                        <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">
                          {reminder.message}
                        </p>
                        <p className="text-xs text-blue-600 dark:text-blue-400">
                          {format(reminder.time, 'PPp', { locale: es })}
                        </p>
    </div>
  );
                  })}
              </div>
            </div>
          )}
        </div>
      )}
      

      
      {/* Panel de búsqueda avanzada */}
      {showAdvancedSearch && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[80vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-bold">Búsqueda Avanzada</h2>
                <button
                  onClick={() => setShowAdvancedSearch(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Operadores de búsqueda
                  </label>
                  <div className="bg-gray-50 p-4 rounded-lg text-sm space-y-2">
                    <div><code className="bg-white px-2 py-1 rounded">repo:nombre</code> - Filtrar por repositorio</div>
                    <div><code className="bg-white px-2 py-1 rounded">status:completed</code> - Filtrar por estado</div>
                    <div><code className="bg-white px-2 py-1 rounded">model:deepseek-chat</code> - Filtrar por modelo</div>
                    <div><code className="bg-white px-2 py-1 rounded">error:texto</code> - Buscar en errores</div>
                    <div><code className="bg-white px-2 py-1 rounded">created:today</code> - Filtrar por fecha</div>
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Búsqueda actual
                  </label>
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Ej: repo:mi-repo status:completed"
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    autoFocus
                  />
                </div>
                
                <div className="text-sm text-gray-600">
                  <strong>Resultados:</strong> {tasksWithStreamingSearch.length} de {tasks.length} tareas
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
      
      {/* Panel de comparación de tareas */}
      {showTaskComparison && selectedTasks.size >= 2 && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-lg shadow-xl max-w-6xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-bold">Comparar Tareas ({selectedTasks.size})</h2>
                <button
                  onClick={() => {
                    setShowTaskComparison(false);
                    setSelectedTasks(new Set());
                  }}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left p-2">Propiedad</th>
                      {Array.from(selectedTasks).map(taskId => {
                        const task = tasks.find(t => t.id === taskId);
                        return task ? (
                          <th key={taskId} className="text-left p-2 border-l">
                            <div className="font-semibold">{task.repository}</div>
                            <div className="text-xs text-gray-500">{task.instruction.substring(0, 30)}...</div>
                          </th>
                        ) : null;
                      })}
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b">
                      <td className="p-2 font-medium">Estado</td>
                      {Array.from(selectedTasks).map(taskId => {
                        const task = tasks.find(t => t.id === taskId);
                        return task ? (
                          <td key={taskId} className="p-2 border-l">
                            <span className={cn(
                              "px-2 py-1 rounded text-xs",
                              task.status === 'completed' && "bg-green-100 text-green-700",
                              task.status === 'failed' && "bg-red-100 text-red-700",
                              task.status === 'processing' && "bg-blue-100 text-blue-700",
                              task.status === 'pending' && "bg-yellow-100 text-yellow-700"
                            )}>
                              {task.status}
                            </span>
                          </td>
                        ) : null;
                      })}
                    </tr>
                    <tr className="border-b">
                      <td className="p-2 font-medium">Modelo</td>
                      {Array.from(selectedTasks).map(taskId => {
                        const task = tasks.find(t => t.id === taskId);
                        return task ? (
                          <td key={taskId} className="p-2 border-l">{task.model || 'N/A'}</td>
                        ) : null;
                      })}
                    </tr>
                    <tr className="border-b">
                      <td className="p-2 font-medium">Creada</td>
                      {Array.from(selectedTasks).map(taskId => {
                        const task = tasks.find(t => t.id === taskId);
                        return task ? (
                          <td key={taskId} className="p-2 border-l text-xs">
                            {format(new Date(task.createdAt), 'PPp', { locale: es })}
                          </td>
                        ) : null;
                      })}
                    </tr>
                    <tr className="border-b">
                      <td className="p-2 font-medium">Error</td>
                      {Array.from(selectedTasks).map(taskId => {
                        const task = tasks.find(t => t.id === taskId);
                        return task ? (
                          <td key={taskId} className="p-2 border-l text-xs text-red-600">
                            {task.error || 'Ninguno'}
                          </td>
                        ) : null;
                      })}
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}
      
      {/* Panel de métricas en tiempo real */}
      {showMetrics && (
        <div className="fixed bottom-4 right-4 bg-white border border-gray-200 rounded-lg shadow-xl z-40 p-4 w-80">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold text-sm">Métricas en Tiempo Real</h3>
            <button
              onClick={() => setShowMetrics(false)}
              className="text-gray-400 hover:text-gray-600"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          
          <div className="space-y-3 text-sm">
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Total tareas</span>
              <span className="font-semibold">{tasks.length}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Completadas</span>
              <span className="font-semibold text-green-600">
                {tasks.filter(t => t.status === 'completed').length}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-600">En proceso</span>
              <span className="font-semibold text-blue-600">
                {tasks.filter(t => t.status === 'processing' || t.status === 'running').length}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Fallidas</span>
              <span className="font-semibold text-red-600">
                {tasks.filter(t => t.status === 'failed').length}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Tasa de éxito</span>
              <span className="font-semibold">
                {tasks.length > 0 
                  ? Math.round((tasks.filter(t => t.status === 'completed').length / tasks.length) * 100)
                  : 0}%
              </span>
            </div>
            <div className="pt-2 border-t">
              <div className="flex items-center justify-between">
                <span className="text-gray-600">WebSocket</span>
                <span className={cn(
                  "font-semibold",
                  wsConnected ? "text-green-600" : "text-red-600"
                )}>
                  {wsConnected ? "Conectado" : "Desconectado"}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
      
      {/* Panel de Analytics Avanzado */}
      {showAnalytics && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
          <div className={cn(
            "rounded-lg shadow-xl max-w-5xl w-full max-h-[90vh] overflow-y-auto",
            darkMode ? "bg-gray-800" : "bg-white"
          )}>
            <div className="p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold">Analytics Avanzado</h2>
                <button
                  onClick={() => setShowAnalytics(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                <div className={cn("p-4 rounded-lg", darkMode ? "bg-gray-700" : "bg-blue-50")}>
                  <div className="text-sm text-gray-600 dark:text-gray-400">Tasa de Éxito</div>
                  <div className="text-2xl font-bold">
                    {tasks.length > 0 
                      ? Math.round((tasks.filter(t => t.status === 'completed').length / tasks.length) * 100)
                      : 0}%
                  </div>
                </div>
                <div className={cn("p-4 rounded-lg", darkMode ? "bg-gray-700" : "bg-green-50")}>
                  <div className="text-sm text-gray-600 dark:text-gray-400">Tiempo Promedio</div>
                  <div className="text-2xl font-bold">
                    {(() => {
                      const completed = tasks.filter(t => t.status === 'completed' && t.processingStartedAt);
                      if (completed.length === 0) return '0m';
                      const avg = completed.reduce((acc, t) => {
                        const start = new Date(t.processingStartedAt!);
                        const end = new Date(t.updatedAt || t.createdAt);
                        return acc + (end.getTime() - start.getTime());
                      }, 0) / completed.length;
                      return `${Math.round(avg / 1000 / 60)}m`;
                    })()}
                  </div>
                </div>
                <div className={cn("p-4 rounded-lg", darkMode ? "bg-gray-700" : "bg-purple-50")}>
                  <div className="text-sm text-gray-600 dark:text-gray-400">Tareas/Hora</div>
                  <div className="text-2xl font-bold">
                    {(() => {
                      const last24h = tasks.filter(t => {
                        const created = new Date(t.createdAt);
                        const now = new Date();
                        return (now.getTime() - created.getTime()) < 24 * 60 * 60 * 1000;
                      });
                      return Math.round(last24h.length / 24);
                    })()}
                  </div>
                </div>
                <div className={cn("p-4 rounded-lg", darkMode ? "bg-gray-700" : "bg-yellow-50")}>
                  <div className="text-sm text-gray-600 dark:text-gray-400">Modelos Únicos</div>
                  <div className="text-2xl font-bold">{models.length}</div>
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div className={cn("p-4 rounded-lg", darkMode ? "bg-gray-700" : "bg-red-50")}>
                  <h3 className="font-semibold mb-3">Prioridades</h3>
                  <div className="space-y-2">
                    {['urgent', 'high', 'medium', 'low'].map(priority => {
                      const count = Array.from(taskPriorities.values()).filter(p => p === priority).length;
                      return (
                        <div key={priority} className="flex justify-between text-sm">
                          <span className="capitalize">{priority}</span>
                          <span className="font-semibold">{count}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
                
                <div className={cn("p-4 rounded-lg", darkMode ? "bg-gray-700" : "bg-blue-50")}>
                  <h3 className="font-semibold mb-3">Estadísticas por Modelo</h3>
                  <div className="space-y-2 max-h-48 overflow-y-auto">
                    {models.map(model => {
                      const modelTasks = tasks.filter(t => t.model === model);
                      const completed = modelTasks.filter(t => t.status === 'completed').length;
                      const successRate = modelTasks.length > 0 ? Math.round((completed / modelTasks.length) * 100) : 0;
                      return (
                        <div key={model} className="text-sm">
                          <div className="flex justify-between mb-1">
                            <span className="truncate">{model}</span>
                            <span className="font-semibold">{successRate}%</span>
                          </div>
                          <div className={cn("h-1 rounded-full", darkMode ? "bg-gray-600" : "bg-gray-200")}>
                            <div 
                              className="h-full rounded-full bg-blue-500"
                              style={{ width: `${successRate}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
                
                <div className={cn("p-4 rounded-lg", darkMode ? "bg-gray-700" : "bg-purple-50")}>
                  <h3 className="font-semibold mb-3">Tags Más Usados</h3>
                  <div className="space-y-2">
                    {allTags.slice(0, 5).map(tag => {
                      const count = Array.from(taskTags.values()).filter(tags => tags.includes(tag)).length;
                      return (
                        <div key={tag} className="flex justify-between text-sm">
                          <span className="capitalize">{tag}</span>
                          <span className="font-semibold">{count}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className={cn("p-4 rounded-lg", darkMode ? "bg-gray-700" : "bg-gray-50")}>
                  <h3 className="font-semibold mb-3">Distribución por Estado</h3>
                  <div className="space-y-2">
                    {KANBAN_COLUMNS.map(col => {
                      const count = tasks.filter(t => t.status === col.id).length;
                      const percentage = tasks.length > 0 ? (count / tasks.length) * 100 : 0;
                      return (
                        <div key={col.id}>
                          <div className="flex justify-between text-sm mb-1">
                            <span>{col.label}</span>
                            <span className="font-semibold">{count} ({Math.round(percentage)}%)</span>
                          </div>
                          <div className={cn("h-2 rounded-full", darkMode ? "bg-gray-600" : "bg-gray-200")}>
                            <div 
                              className="h-full rounded-full"
                              style={{ 
                                width: `${percentage}%`,
                                backgroundColor: getStatusColor(col.id)
                              }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
                
                <div className={cn("p-4 rounded-lg", darkMode ? "bg-gray-700" : "bg-gray-50")}>
                  <h3 className="font-semibold mb-3">Top Repositorios</h3>
                  <div className="space-y-2">
                    {repositories.slice(0, 5).map(repo => {
                      const count = tasks.filter(t => t.repository === repo).length;
                      return (
                        <div key={repo} className="flex justify-between text-sm">
                          <span className="truncate">{repo}</span>
                          <span className="font-semibold">{count}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
      
      {/* Vista de Red de Tareas */}
      {showNetworkView && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
          <div className={cn(
            "rounded-lg shadow-xl max-w-6xl w-full max-h-[90vh] overflow-y-auto",
            darkMode ? "bg-gray-800" : "bg-white"
          )}>
            <div className="p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold">Vista de Red de Tareas</h2>
                <button
                  onClick={() => setShowNetworkView(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* Nodos por repositorio */}
                <div className={cn("p-4 rounded-lg", darkMode ? "bg-gray-700" : "bg-blue-50")}>
                  <h3 className="font-semibold mb-3">Por Repositorio</h3>
                  <div className="space-y-2">
                    {repositories.slice(0, 10).map(repo => {
                      const repoTasks = tasks.filter(t => t.repository === repo);
                      return (
                        <div key={repo} className="flex items-center justify-between text-sm">
                          <span className="truncate">{repo}</span>
                          <span className="font-semibold">{repoTasks.length}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
                
                {/* Nodos por modelo */}
                <div className={cn("p-4 rounded-lg", darkMode ? "bg-gray-700" : "bg-green-50")}>
                  <h3 className="font-semibold mb-3">Por Modelo</h3>
                  <div className="space-y-2">
                    {models.map(model => {
                      const modelTasks = tasks.filter(t => t.model === model);
                      return (
                        <div key={model} className="flex items-center justify-between text-sm">
                          <span className="truncate">{model}</span>
                          <span className="font-semibold">{modelTasks.length}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
                
                {/* Nodos por estado */}
                <div className={cn("p-4 rounded-lg", darkMode ? "bg-gray-700" : "bg-purple-50")}>
                  <h3 className="font-semibold mb-3">Por Estado</h3>
                  <div className="space-y-2">
                    {KANBAN_COLUMNS.map(col => {
                      const statusTasks = tasks.filter(t => t.status === col.id);
                      return (
                        <div key={col.id} className="flex items-center justify-between text-sm">
                          <span>{col.label}</span>
                          <span className="font-semibold">{statusTasks.length}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
              
              <div className="mt-6 p-4 bg-gray-100 dark:bg-gray-700 rounded-lg">
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  <strong>Total de conexiones:</strong> {tasks.length} tareas conectadas
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                  <strong>Repositorios únicos:</strong> {repositories.length} | 
                  <strong> Modelos únicos:</strong> {models.length}
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
      
      {/* Vista de Heatmap */}
      {showHeatmap && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
          <div className={cn(
            "rounded-lg shadow-xl max-w-5xl w-full max-h-[90vh] overflow-y-auto",
            darkMode ? "bg-gray-800" : "bg-white"
          )}>
            <div className="p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold">Heatmap de Actividad</h2>
                <button
                  onClick={() => setShowHeatmap(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              
              <div className="space-y-4">
                <div>
                  <h3 className="font-semibold mb-3">Actividad por Día (Últimos 30 días)</h3>
                  <div className="grid grid-cols-7 gap-2">
                    {Array.from({ length: 30 }, (_, i) => {
                      const date = new Date();
                      date.setDate(date.getDate() - (29 - i));
                      const dayTasks = tasks.filter(t => {
                        const taskDate = new Date(t.createdAt);
                        return taskDate.toDateString() === date.toDateString();
                      });
                      const intensity = Math.min(dayTasks.length / 5, 1);
                      return (
                        <div
                          key={i}
                          className={cn(
                            "aspect-square rounded flex items-center justify-center text-xs",
                            intensity > 0.7 ? "bg-green-500" :
                            intensity > 0.4 ? "bg-green-300" :
                            intensity > 0.1 ? "bg-green-100" :
                            darkMode ? "bg-gray-700" : "bg-gray-100"
                          )}
                          title={`${date.toLocaleDateString()}: ${dayTasks.length} tareas`}
                        >
                          {dayTasks.length > 0 && dayTasks.length}
                        </div>
                      );
                    })}
                  </div>
                </div>
                
                <div>
                  <h3 className="font-semibold mb-3">Actividad por Hora</h3>
                  <div className="grid grid-cols-24 gap-1">
                    {Array.from({ length: 24 }, (_, i) => {
                      const hourTasks = tasks.filter(t => {
                        const taskDate = new Date(t.createdAt);
                        return taskDate.getHours() === i;
                      });
                      const intensity = Math.min(hourTasks.length / 3, 1);
                      return (
                        <div
                          key={i}
                          className={cn(
                            "h-8 rounded text-xs flex items-center justify-center",
                            intensity > 0.7 ? "bg-blue-500" :
                            intensity > 0.4 ? "bg-blue-300" :
                            intensity > 0.1 ? "bg-blue-100" :
                            darkMode ? "bg-gray-700" : "bg-gray-100"
                          )}
                          title={`${i}:00 - ${hourTasks.length} tareas`}
                        >
                          {i % 6 === 0 && i}
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
      
      {/* Panel de Milestones */}
      {showMilestones && milestones.length > 0 && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
          <div className={cn(
            "rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto",
            darkMode ? "bg-gray-800" : "bg-white"
          )}>
            <div className="p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold">Milestones</h2>
                <button
                  onClick={() => setShowMilestones(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              
              <div className="space-y-4">
                {milestones.map(milestone => (
                  <div
                    key={milestone.id}
                    className={cn(
                      "p-4 rounded-lg border-2",
                      darkMode ? "bg-gray-700 border-gray-600" : "bg-green-50 border-green-200"
                    )}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="font-semibold text-lg">{milestone.name}</h3>
                      <span className="text-sm text-gray-600 dark:text-gray-400">
                        {format(milestone.date, 'PP', { locale: es })}
                      </span>
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      {milestone.tasks.length} tareas relacionadas
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
      
      {/* Modo Presentación */}
      {presentationMode && (
        <div className="fixed inset-0 bg-black z-50 flex flex-col">
          <div className="flex-1 flex items-center justify-center p-8">
            <div className="text-center">
              <h1 className="text-6xl font-bold text-white mb-8">bulk</h1>
              <div className="grid grid-cols-4 gap-8 mt-12">
                <div className="bg-gray-800 p-6 rounded-lg">
                  <div className="text-4xl font-bold text-white mb-2">{tasks.length}</div>
                  <div className="text-gray-400">Total Tareas</div>
                </div>
                <div className="bg-green-800 p-6 rounded-lg">
                  <div className="text-4xl font-bold text-white mb-2">
                    {tasks.filter(t => t.status === 'completed').length}
                  </div>
                  <div className="text-gray-400">Completadas</div>
                </div>
                <div className="bg-blue-800 p-6 rounded-lg">
                  <div className="text-4xl font-bold text-white mb-2">
                    {tasks.filter(t => t.status === 'processing' || t.status === 'running').length}
                  </div>
                  <div className="text-gray-400">En Proceso</div>
                </div>
                <div className="bg-purple-800 p-6 rounded-lg">
                  <div className="text-4xl font-bold text-white mb-2">{repositories.length}</div>
                  <div className="text-gray-400">Repositorios</div>
                </div>
              </div>
            </div>
          </div>
          <div className="absolute top-4 right-4">
            <button
              onClick={() => setPresentationMode(false)}
              className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
            >
              Salir (Esc)
            </button>
          </div>
        </div>
      )}
      
      {/* Panel de Alertas Inteligentes */}
      {showSmartAlerts && smartAlerts.length > 0 && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
          <div className={cn(
            "rounded-lg shadow-xl max-w-2xl w-full max-h-[80vh] overflow-y-auto",
            darkMode ? "bg-gray-800" : "bg-white"
          )}>
            <div className="p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold">Alertas Inteligentes</h2>
                <button
                  onClick={() => setShowSmartAlerts(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              
              <div className="space-y-3">
                {smartAlerts.map(alert => (
                  <div
                    key={alert.id}
                    className={cn(
                      "p-4 rounded-lg border-l-4",
                      alert.type === 'error' && "bg-red-50 border-red-500 dark:bg-red-900/20",
                      alert.type === 'warning' && "bg-yellow-50 border-yellow-500 dark:bg-yellow-900/20",
                      alert.type === 'info' && "bg-blue-50 border-blue-500 dark:bg-blue-900/20"
                    )}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className={cn(
                          "font-semibold mb-1",
                          alert.type === 'error' && "text-red-700 dark:text-red-400",
                          alert.type === 'warning' && "text-yellow-700 dark:text-yellow-400",
                          alert.type === 'info' && "text-blue-700 dark:text-blue-400"
                        )}>
                          {alert.type === 'error' && '⚠️ Error'}
                          {alert.type === 'warning' && '⚠️ Advertencia'}
                          {alert.type === 'info' && 'ℹ️ Información'}
                        </div>
                        <p className="text-sm text-gray-700 dark:text-gray-300">{alert.message}</p>
                      </div>
                      <span className="text-xs text-gray-500 ml-4">
                        {format(alert.timestamp, 'HH:mm', { locale: es })}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
      
      {/* Panel de Feed de Actividad */}
      {showActivityFeed && (
        <div className="fixed right-0 top-0 h-full w-96 z-40 shadow-xl overflow-y-auto">
          <div className={cn("h-full", darkMode ? "bg-gray-800 border-l border-gray-700" : "bg-white border-l border-gray-200")}>
            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-bold">Feed de Actividad</h3>
                <button
                  onClick={() => setShowActivityFeed(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              
              <div className="space-y-3">
                {activityFeed.map(activity => (
                  <div
                    key={activity.id}
                    className={cn(
                      "p-3 rounded-lg",
                      darkMode ? "bg-gray-700" : "bg-gray-50"
                    )}
                  >
                    <div className="flex items-start gap-3">
                      <div className={cn(
                        "w-2 h-2 rounded-full mt-2",
                        activity.type === 'created' && "bg-blue-500",
                        activity.type === 'completed' && "bg-green-500",
                        activity.type === 'failed' && "bg-red-500"
                      )} />
                      <div className="flex-1">
                        <p className="text-sm">{activity.message}</p>
                        <p className="text-xs text-gray-500 mt-1">
                          {format(activity.timestamp, 'PPp', { locale: es })}
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
                {activityFeed.length === 0 && (
                  <p className="text-sm text-gray-500 text-center py-8">
                    No hay actividad reciente
                  </p>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
      
      {/* Panel de Plantillas */}
      {showTemplates && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
          <div className={cn(
            "rounded-lg shadow-xl max-w-3xl w-full max-h-[90vh] overflow-y-auto",
            darkMode ? "bg-gray-800" : "bg-white"
          )}>
            <div className="p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold">Plantillas de Tareas</h2>
                <button
                  onClick={() => setShowTemplates(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              
              <div className="space-y-4">
                {taskTemplates.map(template => (
                  <div
                    key={template.id}
                    className={cn(
                      "p-4 rounded-lg border-2 cursor-pointer hover:border-blue-500 transition-colors",
                      darkMode ? "bg-gray-700 border-gray-600" : "bg-gray-50 border-gray-200"
                    )}
                    onClick={() => {
                      // Navegar a crear tarea con plantilla
                      window.location.href = `/agent-control?template=${encodeURIComponent(JSON.stringify(template))}`;
                    }}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="font-semibold">{template.name}</h3>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setTaskTemplates(prev => prev.filter(t => t.id !== template.id));
                          toast.success('Plantilla eliminada');
                        }}
                        className="text-red-500 hover:text-red-700"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                      </button>
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-400">{template.instruction}</p>
                    {template.model && (
                      <span className="inline-block mt-2 px-2 py-1 text-xs bg-blue-100 text-blue-700 rounded">
                        {template.model}
                      </span>
                    )}
                  </div>
                ))}
                
                <div className={cn(
                  "p-4 rounded-lg border-2 border-dashed cursor-pointer hover:border-blue-500 transition-colors",
                  darkMode ? "bg-gray-700 border-gray-600" : "bg-gray-50 border-gray-300"
                )}>
                  <input
                    type="text"
                    placeholder="Nombre de la plantilla"
                    className={cn(
                      "w-full px-3 py-2 rounded-lg text-sm mb-2",
                      darkMode 
                        ? "bg-gray-600 text-white border border-gray-500"
                        : "bg-white border border-gray-300"
                    )}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        const name = e.currentTarget.value.trim();
                        const instructionInput = (e.currentTarget.nextElementSibling as HTMLTextAreaElement);
                        const instruction = instructionInput.value.trim();
                        if (name && instruction) {
                          setTaskTemplates(prev => [...prev, {
                            id: `template-${Date.now()}`,
                            name,
                            instruction,
                            model: undefined
                          }]);
                          e.currentTarget.value = '';
                          instructionInput.value = '';
                          toast.success('Plantilla creada');
                        }
                      }
                    }}
                  />
                  <textarea
                    placeholder="Instrucción de la plantilla"
                    className={cn(
                      "w-full px-3 py-2 rounded-lg text-sm resize-none",
                      darkMode 
                        ? "bg-gray-600 text-white border border-gray-500"
                        : "bg-white border border-gray-300"
                    )}
                    rows={3}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
      
      {/* Panel de Colaboración */}
      {showCollaboration && (
        <div className="fixed right-0 top-0 h-full w-80 z-40 shadow-xl overflow-y-auto">
          <div className={cn("h-full", darkMode ? "bg-gray-800 border-l border-gray-700" : "bg-white border-l border-gray-200")}>
            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-bold">Colaboración</h3>
                <button
                  onClick={() => setShowCollaboration(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              
              <div className="space-y-4">
                <div>
                  <h4 className="font-semibold mb-2">Estado de Conexión</h4>
                  <div className={cn(
                    "p-3 rounded-lg flex items-center gap-2",
                    wsConnected 
                      ? "bg-green-50 text-green-700 dark:bg-green-900/20 dark:text-green-400"
                      : "bg-red-50 text-red-700 dark:bg-red-900/20 dark:text-red-400"
                  )}>
                    <div className={cn(
                      "w-3 h-3 rounded-full",
                      wsConnected ? "bg-green-500" : "bg-red-500"
                    )} />
                    <span className="text-sm">
                      {wsConnected ? 'Conectado en tiempo real' : 'Desconectado'}
                    </span>
                  </div>
                </div>
                
                <div>
                  <h4 className="font-semibold mb-2">Usuarios Activos</h4>
                  <div className="space-y-2">
                    <div className={cn("p-2 rounded-lg flex items-center gap-2", darkMode ? "bg-gray-700" : "bg-gray-50")}>
                      <div className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center text-white text-xs">
                        TU
                      </div>
                      <div className="flex-1">
                        <div className="text-sm font-medium">Tú</div>
                        <div className="text-xs text-gray-500">En línea</div>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div>
                  <h4 className="font-semibold mb-2">Actividad Reciente</h4>
                  <div className="space-y-2 max-h-64 overflow-y-auto">
                    {activityFeed.slice(0, 10).map(activity => (
                      <div
                        key={activity.id}
                        className={cn("p-2 rounded-lg text-xs", darkMode ? "bg-gray-700" : "bg-gray-50")}
                      >
                        {activity.message}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
      
      {/* Panel de Dashboard Personalizado */}
      {showCustomDashboard && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
          <div className={cn(
            "rounded-lg shadow-xl max-w-6xl w-full max-h-[90vh] overflow-y-auto",
            darkMode ? "bg-gray-800" : "bg-white"
          )}>
            <div className="p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold">Dashboard Personalizado</h2>
                <button
                  onClick={() => setShowCustomDashboard(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              
              <div className="mb-4">
                <label className="block text-sm font-medium mb-2">Layout</label>
                <select
                  value={dashboardLayout}
                  onChange={(e) => setDashboardLayout(e.target.value as 'grid' | 'list' | 'compact')}
                  className={cn(
                    "px-3 py-2 rounded-lg text-sm",
                    darkMode 
                      ? "bg-gray-700 text-white border border-gray-600"
                      : "bg-white border border-gray-300"
                  )}
                >
                  <option value="grid">Grid</option>
                  <option value="list">Lista</option>
                  <option value="compact">Compacto</option>
                </select>
              </div>
              
              <div className={cn(
                "grid gap-4",
                dashboardLayout === 'grid' && "grid-cols-3",
                dashboardLayout === 'list' && "grid-cols-1",
                dashboardLayout === 'compact' && "grid-cols-4"
              )}>
                <div className={cn("p-4 rounded-lg", darkMode ? "bg-gray-700" : "bg-blue-50")}>
                  <div className="text-2xl font-bold">{tasks.length}</div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">Total Tareas</div>
                </div>
                <div className={cn("p-4 rounded-lg", darkMode ? "bg-gray-700" : "bg-green-50")}>
                  <div className="text-2xl font-bold">
                    {tasks.filter(t => t.status === 'completed').length}
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">Completadas</div>
                </div>
                <div className={cn("p-4 rounded-lg", darkMode ? "bg-gray-700" : "bg-yellow-50")}>
                  <div className="text-2xl font-bold">
                    {tasks.filter(t => t.status === 'processing' || t.status === 'running').length}
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">En Proceso</div>
                </div>
                <div className={cn("p-4 rounded-lg", darkMode ? "bg-gray-700" : "bg-red-50")}>
                  <div className="text-2xl font-bold">
                    {tasks.filter(t => t.status === 'failed').length}
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">Fallidas</div>
                </div>
                <div className={cn("p-4 rounded-lg", darkMode ? "bg-gray-700" : "bg-purple-50")}>
                  <div className="text-2xl font-bold">{repositories.length}</div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">Repositorios</div>
                </div>
                <div className={cn("p-4 rounded-lg", darkMode ? "bg-gray-700" : "bg-indigo-50")}>
                  <div className="text-2xl font-bold">{models.length}</div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">Modelos</div>
                </div>
              </div>
              
              <div className="mt-6 pt-6 border-t border-gray-200 dark:border-gray-700">
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={autoReports}
                    onChange={(e) => setAutoReports(e.target.checked)}
                    className="w-4 h-4"
                  />
                  <span className="text-sm">Generar reportes automáticos</span>
                </label>
                {autoReports && (
                  <select
                    value={reportFrequency}
                    onChange={(e) => setReportFrequency(e.target.value as 'daily' | 'weekly' | 'monthly')}
                    className={cn(
                      "mt-2 px-3 py-2 rounded-lg text-sm",
                      darkMode 
                        ? "bg-gray-700 text-white border border-gray-600"
                        : "bg-white border border-gray-300"
                    )}
                  >
                    <option value="daily">Diario</option>
                    <option value="weekly">Semanal</option>
                    <option value="monthly">Mensual</option>
                  </select>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
      
      {/* Panel de Filtros Guardados */}
      {showSavedFilters && savedFilters.length > 0 && (
        <div className="fixed right-0 top-0 h-full w-80 z-40 shadow-xl overflow-y-auto">
          <div className={cn("h-full", darkMode ? "bg-gray-800 border-l border-gray-700" : "bg-white border-l border-gray-200")}>
            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-bold">Filtros Guardados</h3>
                <button
                  onClick={() => setShowSavedFilters(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              
              <div className="space-y-2">
                {savedFilters.map((filter, index) => (
                  <div
                    key={index}
                    className={cn(
                      "p-3 rounded-lg cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors",
                      darkMode ? "bg-gray-700" : "bg-gray-50"
                    )}
                    onClick={() => {
                      const filters = applySavedFilter(filter);
                      if (filters.searchQuery) setSearchQuery(filters.searchQuery);
                      if (filters.selectedRepository) setSelectedRepository(filters.selectedRepository);
                      if (filters.statusFilter) setStatusFilter(filters.statusFilter as TaskStatus | 'all');
                      if (filters.dateFilter) setDateFilter(filters.dateFilter);
                      if (filters.sortBy) setSortBy(filters.sortBy);
                      if (filters.sortOrder) setSortOrder(filters.sortOrder as 'asc' | 'desc');
                      toast.success(`Filtro "${filter.name}" aplicado`);
                    }}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-medium text-sm">{filter.name}</span>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteSavedFilter(index);
                          toast.success('Filtro eliminado');
                        }}
                        className="text-red-500 hover:text-red-700"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                      </button>
                    </div>
                  </div>
                ))}
              </div>
              
              <button
                onClick={() => {
                  const filterName = prompt('Nombre del filtro:');
                  if (filterName) {
                    saveFilter({
                      searchQuery,
                      selectedRepository,
                      statusFilter,
                      dateFilter,
                      sortBy,
                      sortOrder
                    }, filterName);
                    toast.success('Filtro guardado');
                  }
                }}
                className={cn(
                  "w-full mt-4 px-4 py-2 rounded-lg text-sm font-medium",
                  darkMode 
                    ? "bg-gray-700 text-white hover:bg-gray-600"
                    : "bg-blue-500 text-white hover:bg-blue-600"
                )}
              >
                Guardar filtro actual
              </button>
            </div>
          </div>
        </div>
      )}
      
      {/* Panel de Logs de Auditoría */}
      {showAuditLogs && (
        <div className="fixed right-0 top-0 h-full w-96 z-40 shadow-xl overflow-y-auto">
          <div className={cn("h-full", darkMode ? "bg-gray-800 border-l border-gray-700" : "bg-white border-l border-gray-200")}>
            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-bold">Logs de Auditoría</h3>
                <button
                  onClick={() => setShowAuditLogs(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              
              <div className="space-y-2 max-h-[calc(100vh-200px)] overflow-y-auto">
                {auditLogs.map(log => (
                  <div
                    key={log.id}
                    className={cn(
                      "p-3 rounded-lg text-sm",
                      darkMode ? "bg-gray-700" : "bg-gray-50"
                    )}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-semibold">{log.action}</span>
                      <span className="text-xs text-gray-500">
                        {format(log.timestamp, 'HH:mm:ss', { locale: es })}
                      </span>
                    </div>
                    <p className="text-xs text-gray-600 dark:text-gray-400">{log.details}</p>
                  </div>
                ))}
                {auditLogs.length === 0 && (
                  <p className="text-sm text-gray-500 text-center py-8">
                    No hay logs de auditoría
                  </p>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
      
      {/* Panel de Backup/Restore */}
      {showBackupRestore && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
          <div className={cn(
            "rounded-lg shadow-xl max-w-2xl w-full",
            darkMode ? "bg-gray-800" : "bg-white"
          )}>
            <div className="p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold">Backup y Restore</h2>
                <button
                  onClick={() => setShowBackupRestore(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              
              <div className="space-y-4">
                <div>
                  <h3 className="font-semibold mb-3">Crear Backup</h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                    Guarda toda la configuración, filtros, plantillas y vistas personalizadas.
                  </p>
                  <button
                    onClick={() => {
                      const backup = {
                        version: '1.0',
                        timestamp: new Date().toISOString(),
                        filters: savedFilters || [],
                        templates: taskTemplates,
                        views: customViews,
                        settings: {
                          darkMode,
                          compactMode,
                          dashboardLayout,
                          autoReports,
                          reportFrequency,
                          soundEnabled
                        }
                      };
                      const blob = new Blob([JSON.stringify(backup, null, 2)], { type: 'application/json' });
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement('a');
                      a.href = url;
                      a.download = `kanban-backup-${new Date().toISOString().split('T')[0]}.json`;
                      a.click();
                      toast.success('Backup creado y descargado');
                    }}
                    className={cn(
                      "w-full px-4 py-2 rounded-lg text-sm font-medium",
                      darkMode 
                        ? "bg-gray-700 text-white hover:bg-gray-600"
                        : "bg-blue-500 text-white hover:bg-blue-600"
                    )}
                  >
                    Descargar Backup
                  </button>
                </div>
                
                <div>
                  <h3 className="font-semibold mb-3">Restaurar Backup</h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                    Restaura una configuración desde un archivo de backup.
                  </p>
                  <input
                    type="file"
                    accept=".json"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) {
                        const reader = new FileReader();
                        reader.onload = (event) => {
                          try {
                            const backup = JSON.parse(event.target?.result as string);
                            if (backup.filters && Array.isArray(backup.filters)) {
                              // Restaurar filtros usando localStorage directamente ya que el hook los maneja
                              localStorage.setItem('kanban-saved-filters', JSON.stringify(backup.filters));
                              window.location.reload(); // Recargar para que el hook los cargue
                            }
                            if (backup.templates) setTaskTemplates(backup.templates);
                            if (backup.views) setCustomViews(backup.views);
                            if (backup.settings) {
                              if (backup.settings.darkMode !== undefined) setDarkMode(backup.settings.darkMode);
                              if (backup.settings.compactMode !== undefined) setCompactMode(backup.settings.compactMode);
                              if (backup.settings.dashboardLayout) setDashboardLayout(backup.settings.dashboardLayout);
                              if (backup.settings.autoReports !== undefined) setAutoReports(backup.settings.autoReports);
                              if (backup.settings.reportFrequency) setReportFrequency(backup.settings.reportFrequency);
                              if (backup.settings.soundEnabled !== undefined) setSoundEnabled(backup.settings.soundEnabled);
                            }
                            toast.success('Backup restaurado correctamente');
                            setShowBackupRestore(false);
                          } catch (error) {
                            toast.error('Error al restaurar backup. Verifica que el archivo sea válido.');
                          }
                        };
                        reader.readAsText(file);
                      }
                    }}
                    className={cn(
                      "w-full px-3 py-2 rounded-lg text-sm",
                      darkMode 
                        ? "bg-gray-700 text-white border border-gray-600"
                        : "bg-white border border-gray-300"
                    )}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
      
      {/* Panel de Acciones Rápidas */}
      {showQuickActions && (
        <div className="fixed right-0 top-0 h-full w-80 z-40 shadow-xl overflow-y-auto">
          <div className={cn("h-full", darkMode ? "bg-gray-800 border-l border-gray-700" : "bg-white border-l border-gray-200")}>
            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-bold">Acciones Rápidas</h3>
                <button
                  onClick={() => setShowQuickActions(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              
              <div className="space-y-2">
                {quickActions.map(action => (
                  <button
                    key={action.id}
                    onClick={() => {
                      action.action();
                      setShowQuickActions(false);
                    }}
                    className={cn(
                      "w-full text-left px-4 py-3 rounded-lg text-sm font-medium transition-colors",
                      darkMode 
                        ? "bg-gray-700 hover:bg-gray-600"
                        : "bg-gray-50 hover:bg-gray-100"
                    )}
                  >
                    {action.name}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
      
      {/* Menú de Operaciones Masivas */}
      {showBulkMenu && selectedTasks.size > 0 && (
        <div className="fixed right-0 top-0 h-full w-80 z-40 shadow-xl overflow-y-auto">
          <div className={cn("h-full", darkMode ? "bg-gray-800 border-l border-gray-700" : "bg-white border-l border-gray-200")}>
            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-bold">Operaciones Masivas</h3>
                <button
                  onClick={() => setShowBulkMenu(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                {selectedTasks.size} tareas seleccionadas
              </p>
              
              <div className="space-y-2">
                <button
                  onClick={() => {
                    const selectedTasksArray = Array.from(selectedTasks);
                    const tasksToExport = tasks.filter(t => selectedTasksArray.includes(t.id));
                    exportTasks(tasksToExport);
                    toast.success(`${tasksToExport.length} tareas exportadas`);
                    setSelectedTasks(new Set());
                    setShowBulkMenu(false);
                  }}
                  className={cn(
                    "w-full text-left px-4 py-3 rounded-lg text-sm font-medium transition-colors",
                    darkMode 
                      ? "bg-gray-700 hover:bg-gray-600"
                      : "bg-gray-50 hover:bg-gray-100"
                  )}
                >
                  Exportar seleccionadas
                </button>
                
                <button
                  onClick={() => {
                    setSelectedTasks(new Set());
                    toast.success('Selección limpiada');
                    setShowBulkMenu(false);
                  }}
                  className={cn(
                    "w-full text-left px-4 py-3 rounded-lg text-sm font-medium transition-colors",
                    darkMode 
                      ? "bg-gray-700 hover:bg-gray-600"
                      : "bg-gray-50 hover:bg-gray-100"
                  )}
                >
                  Limpiar selección
                </button>
                
                <button
                  onClick={() => {
                    const selectedTasksArray = Array.from(selectedTasks);
                    const tasksToDelete = tasks.filter(t => selectedTasksArray.includes(t.id));
                    if (confirm(`¿Eliminar ${tasksToDelete.length} tareas?`)) {
                      tasksToDelete.forEach(task => handleDeleteTask(task.id));
                      setSelectedTasks(new Set());
                      toast.success(`${tasksToDelete.length} tareas eliminadas`);
                      setShowBulkMenu(false);
                    }
                  }}
                  className={cn(
                    "w-full text-left px-4 py-3 rounded-lg text-sm font-medium transition-colors text-red-600",
                    darkMode 
                      ? "bg-red-900/20 hover:bg-red-900/30"
                      : "bg-red-50 hover:bg-red-100"
                  )}
                >
                  Eliminar seleccionadas
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

