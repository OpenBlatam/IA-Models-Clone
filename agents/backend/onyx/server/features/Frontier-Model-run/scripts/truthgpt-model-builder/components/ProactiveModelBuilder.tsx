'use client'

import { useState, useEffect, useCallback, useRef, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Play, Pause, Square, Loader2, CheckCircle, XCircle, Zap, Activity, 
  Trash2, ArrowUp, ArrowDown, RefreshCw, Download, Upload, TrendingUp,
  Clock, FileText, Settings, BarChart3, AlertCircle, Bell, BellOff,
  Search, Filter, X, Eye, Tag, Bookmark, Layers, Sparkles, GitCompare,
  FileText as LogsIcon, Cpu, Save, Database, FileDown, FileSpreadsheet,
  HardDrive, RefreshCw as RotateCcw, Webhook, Moon, Sun, GitBranch,
  Share2, Copy, Check, TestTube, Beaker, FlaskConical, BookOpen,
  Plus, Edit, Star, TrendingUp as TrendingUpIcon, Heart, Download,
  FileDown, BarChart2, Sparkles as SparklesIcon, History, Command,
  Bell, Zap, Search as SearchIcon, HelpCircle, Keyboard, Sparkles,
  Gauge, AlertTriangle, TrendingUp as TrendUp, Activity as ActivityIcon
} from 'lucide-react'
import { toast } from 'react-hot-toast'
import { getNotificationManager } from '@/lib/notification-manager'
import { MODEL_TEMPLATES, ModelTemplate, searchTemplates, getTemplatesByCategory } from '@/lib/modules/management'
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { getKeyboardShortcutsManager } from '@/lib/keyboard-shortcuts'
import ConfigPreview from './ConfigPreview'
import ModelComparator from './ModelComparator'
import { adaptToOptimizationCore } from '@/lib/optimization-core-adapter'
import { adaptiveAnalyze } from '@/lib/adaptive-analyzer'
import { getLogger, LogLevel } from '@/lib/logger'
import { BatchProcessor } from '@/lib/batch-processor'
import { getBackupManager } from '@/lib/backup-manager'
import { getReportGenerator } from '@/lib/report-generator'
import { getAdvancedValidator } from '@/lib/advanced-validator'
import { getWebhookManager } from '@/lib/webhook-manager'
import { getModelVersioning } from '@/lib/modules/management'
import { getThemeManager } from '@/lib/theme-manager'
import { getABTesting } from '@/lib/ab-testing'
import { getCustomTemplates } from '@/lib/custom-templates'
import { getEnhancedAdaptiveAnalyzer } from '@/lib/enhanced-adaptive-analyzer'
import { getAdvancedStatistics } from '@/lib/advanced-statistics'
import { getModelExporter } from '@/lib/modules/management'
import { getFavoritesManager } from '@/lib/favorites-manager'
import { getSmartHistory } from '@/lib/smart-history'
import { getEnhancedNotifications } from '@/lib/enhanced-notifications'
import { getQuickCommands } from '@/lib/quick-commands'
import { getEnhancedKeyboardShortcuts } from '@/lib/enhanced-keyboard-shortcuts'
import { getContextualHelp } from '@/lib/contextual-help'
import { getRealTimeMetrics } from '@/lib/realtime-metrics'
import { getSmartCache } from '@/lib/smart-cache'
import { getIntelligentAlerts } from '@/lib/intelligent-alerts'
import { 
  fadeIn, slideInFromTop, scaleInBounce, staggerContainer, staggerItem,
  cardHover, buttonPress, springAnimation
} from '@/lib/animation-presets'
import AdvancedDashboard from './AdvancedDashboard'
import WebhookConfigPanel from './WebhookConfig'
import ABTestingPanel from './ABTestingPanel'
import CustomTemplatesPanel from './CustomTemplatesPanel'

interface ProactiveBuildResult {
  modelId: string
  modelName: string
  status: 'creating' | 'completed' | 'failed'
  description: string
  error?: string
  progress?: number
  startTime?: number
  endTime?: number
  duration?: number
}

interface QueueItem {
  id: string
  description: string
  priority: number
  createdAt: number
  tags?: string[]
  category?: string
  templateId?: string
}

interface ProactiveModelBuilderProps {
  onModelCreated?: (result: ProactiveBuildResult) => void
  optimizationCorePath?: string
}

export default function ProactiveModelBuilder({
  onModelCreated,
  optimizationCorePath = '../TruthGPT-main/optimization_core',
}: ProactiveModelBuilderProps) {
  const [isActive, setIsActive] = useState(false)
  const [isBuilding, setIsBuilding] = useState(false)
  const [buildQueue, setBuildQueue] = useState<QueueItem[]>([])
  const [completedBuilds, setCompletedBuilds] = useState<ProactiveBuildResult[]>([])
  const [currentBuild, setCurrentBuild] = useState<QueueItem | null>(null)
  const [input, setInput] = useState('')
  const [autoMode, setAutoMode] = useState(false)
  const [showStats, setShowStats] = useState(false)
  const [showMetricsChart, setShowMetricsChart] = useState(false)
  const [buildProgress, setBuildProgress] = useState(0)
  const [retryCount, setRetryCount] = useState(0)
  const [notificationsEnabled, setNotificationsEnabled] = useState(false)
  const [showTemplates, setShowTemplates] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [filterStatus, setFilterStatus] = useState<'all' | 'completed' | 'failed'>('all')
  const [showPreview, setShowPreview] = useState(false)
  const [previewItem, setPreviewItem] = useState<QueueItem | null>(null)
  const [previewConfig, setPreviewConfig] = useState<any>(null)
  const [showHelp, setShowHelp] = useState(false)
  const [showComparator, setShowComparator] = useState(false)
  const [showLogs, setShowLogs] = useState(false)
  const [showAdvancedDashboard, setShowAdvancedDashboard] = useState(false)
  const [batchMode, setBatchMode] = useState(false)
  const [batchConcurrency, setBatchConcurrency] = useState(3)
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const buildStartTimeRef = useRef<number | null>(null)
  const notificationManager = useMemo(() => getNotificationManager(), [])
  const shortcutsManager = useMemo(() => getKeyboardShortcutsManager(), [])
  const logger = useMemo(() => getLogger(), [])
  const backupManager = useMemo(() => getBackupManager(), [])
  const reportGenerator = useMemo(() => getReportGenerator(), [])
  const validator = useMemo(() => getAdvancedValidator(), [])
  const webhookManager = useMemo(() => getWebhookManager(), [])
  const modelVersioning = useMemo(() => getModelVersioning(), [])
  const themeManager = useMemo(() => getThemeManager(), [])
  const [currentTheme, setCurrentTheme] = useState<'dark' | 'light' | 'auto'>('dark')
  const [showWebhooks, setShowWebhooks] = useState(false)
  const [showVersioning, setShowVersioning] = useState(false)
  const [showABTesting, setShowABTesting] = useState(false)
  const [showCustomTemplates, setShowCustomTemplates] = useState(false)
  const [copiedId, setCopiedId] = useState<string | null>(null)
  const abTesting = useMemo(() => getABTesting(), [])
  const customTemplates = useMemo(() => getCustomTemplates(), [])
  const enhancedAnalyzer = useMemo(() => getEnhancedAdaptiveAnalyzer(), [])
  const advancedStats = useMemo(() => getAdvancedStatistics(), [])
  const modelExporter = useMemo(() => getModelExporter(), [])
  const favoritesManager = useMemo(() => getFavoritesManager(), [])
  const smartHistory = useMemo(() => getSmartHistory(), [])
  const enhancedNotifications = useMemo(() => getEnhancedNotifications(), [])
  const quickCommands = useMemo(() => getQuickCommands(), [])
  const enhancedShortcuts = useMemo(() => getEnhancedKeyboardShortcuts(), [])
  const contextualHelp = useMemo(() => getContextualHelp(), [])
  const realtimeMetrics = useMemo(() => getRealTimeMetrics(), [])
  const smartCache = useMemo(() => getSmartCache('proactive-builder', { maxSize: 50, strategy: 'lru' }), [])
  const intelligentAlerts = useMemo(() => getIntelligentAlerts(), [])
  const [showAdvancedStats, setShowAdvancedStats] = useState(false)
  const [showExportMenu, setShowExportMenu] = useState(false)
  const [showFavorites, setShowFavorites] = useState(false)
  const [showSmartHistory, setShowSmartHistory] = useState(false)
  const [showQuickCommands, setShowQuickCommands] = useState(false)
  const [showHelp, setShowHelp] = useState(false)
  const [showRealtimeMetrics, setShowRealtimeMetrics] = useState(false)
  const [showAlerts, setShowAlerts] = useState(false)
  const [helpContext, setHelpContext] = useState<string | null>(null)
  const [historySearchQuery, setHistorySearchQuery] = useState('')
  const [realtimeData, setRealtimeData] = useState<any>(null)

  // Estadísticas calculadas
  const stats = useMemo(() => {
    const total = completedBuilds.length
    const successful = completedBuilds.filter(b => b.status === 'completed').length
    const failed = completedBuilds.filter(b => b.status === 'failed').length
    const avgDuration = completedBuilds
      .filter(b => b.duration)
      .reduce((sum, b) => sum + (b.duration || 0), 0) / (total || 1)
    const successRate = total > 0 ? (successful / total) * 100 : 0
    
    return { total, successful, failed, avgDuration, successRate }
  }, [completedBuilds])

  // Estadísticas avanzadas
  const advancedStatsData = useMemo(() => {
    if (completedBuilds.length === 0) return null
    return advancedStats.calculateAdvancedStats(completedBuilds)
  }, [completedBuilds, advancedStats])

  // Métricas en tiempo real
  useEffect(() => {
    const metrics = realtimeMetrics
    metrics.registerMetric('buildsPerMinute', 'Builds por Minuto', '/min')
    metrics.registerMetric('successRate', 'Tasa de Éxito', '%')
    metrics.registerMetric('avgDuration', 'Duración Promedio', 'ms')
    metrics.registerMetric('queueLength', 'Longitud de Cola', '')

    const updateMetrics = () => {
      const modelMetrics = metrics.calculateModelMetrics(completedBuilds)
      metrics.updateMetric('buildsPerMinute', modelMetrics.buildsPerMinute)
      metrics.updateMetric('successRate', modelMetrics.successRate * 100)
      metrics.updateMetric('avgDuration', modelMetrics.avgDuration)
      metrics.updateMetric('queueLength', buildQueue.length)

      setRealtimeData({
        buildsPerMinute: modelMetrics.buildsPerMinute,
        successRate: modelMetrics.successRate * 100,
        avgDuration: modelMetrics.avgDuration,
        queueLength: buildQueue.length,
      })
    }

    metrics.startAutoUpdate(updateMetrics)
    updateMetrics() // Initial update

    return () => {
      metrics.stopAutoUpdate()
    }
  }, [completedBuilds, buildQueue.length, realtimeMetrics])

  // Alertas inteligentes
  useEffect(() => {
    const unsubscribe = intelligentAlerts.subscribe((alert) => {
      if (alert.severity === 'error' || alert.severity === 'critical') {
        toast.error(alert.message, { icon: '🚨', duration: 6000 })
      } else if (alert.severity === 'warning') {
        toast(alert.message, { icon: '⚠️', duration: 4000 })
      } else {
        toast(alert.message, { icon: 'ℹ️', duration: 3000 })
      }
    })

    // Evaluar alertas periódicamente
    const alertInterval = setInterval(() => {
      intelligentAlerts.evaluate({
        models: completedBuilds,
        queueLength: buildQueue.length,
      })
    }, 30000) // Cada 30 segundos

    return () => {
      unsubscribe()
      clearInterval(alertInterval)
    }
  }, [completedBuilds, buildQueue.length, intelligentAlerts])

  // Datos para gráficos
  const chartData = useMemo(() => {
    const last10 = completedBuilds.slice(-10).reverse()
    return last10.map((build, index) => ({
      name: `#${completedBuilds.length - index}`,
      duration: build.duration ? Math.round(build.duration / 1000) : 0,
      status: build.status === 'completed' ? 1 : 0,
    }))
  }, [completedBuilds])

  // Modelos filtrados
  const filteredCompleted = useMemo(() => {
    let filtered = completedBuilds

    // Filtrar por estado
    if (filterStatus !== 'all') {
      filtered = filtered.filter(b => b.status === filterStatus)
    }

    // Filtrar por búsqueda
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase()
      filtered = filtered.filter(b =>
        b.modelName.toLowerCase().includes(query) ||
        b.description.toLowerCase().includes(query)
      )
    }

    return filtered
  }, [completedBuilds, filterStatus, searchQuery])

  // Habilitar notificaciones
  const enableNotifications = useCallback(async () => {
    const granted = await notificationManager.requestPermission()
    if (granted) {
      setNotificationsEnabled(true)
      toast.success('Notificaciones habilitadas', { icon: '🔔' })
    } else {
      toast.error('Permisos de notificación denegados', { icon: '❌' })
    }
  }, [notificationManager])

  // Agregar descripción a la cola
  const addToQueue = useCallback((description: string, priority: number = 0, template?: ModelTemplate) => {
    if (!description.trim()) {
      toast.error('Por favor ingresa una descripción')
      logger.warn('Intento de agregar descripción vacía a la cola')
      return
    }

    if (description.trim().length < 10) {
      toast.error('La descripción debe tener al menos 10 caracteres')
      logger.warn('Descripción demasiado corta', { length: description.trim().length })
      return
    }

    const newItem: QueueItem = {
      id: `queue-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      description: description.trim(),
      priority: template ? template.priority : priority,
      createdAt: Date.now(),
      tags: template?.tags || [],
      category: template?.category,
      templateId: template?.id,
    }

    setBuildQueue(prev => {
      // Ordenar por prioridad (mayor primero), luego por fecha
      const sorted = [...prev, newItem].sort((a, b) => {
        if (b.priority !== a.priority) return b.priority - a.priority
        return a.createdAt - b.createdAt
      })
      return sorted
    })
    
    logger.info('Modelo agregado a la cola', {
      description: newItem.description,
      priority: newItem.priority,
      templateId: newItem.templateId,
      category: newItem.category,
    })
    
    toast.success('Descripción agregada a la cola', {
      icon: '✅',
    })
  }, [logger])

  // Agregar desde plantilla
  const addTemplateToQueue = useCallback((template: ModelTemplate) => {
    addToQueue(template.description, template.priority, template)
    setShowTemplates(false)
  }, [addToQueue])

  // Eliminar de la cola
  const removeFromQueue = useCallback((id: string) => {
    setBuildQueue(prev => prev.filter(item => item.id !== id))
    toast('Eliminado de la cola', { icon: '🗑️' })
  }, [])

  // Mover en la cola
  const moveInQueue = useCallback((id: string, direction: 'up' | 'down') => {
    setBuildQueue(prev => {
      const index = prev.findIndex(item => item.id === id)
      if (index === -1) return prev
      if (direction === 'up' && index === 0) return prev
      if (direction === 'down' && index === prev.length - 1) return prev
      
      const newQueue = [...prev]
      const targetIndex = direction === 'up' ? index - 1 : index + 1
      ;[newQueue[index], newQueue[targetIndex]] = [newQueue[targetIndex], newQueue[index]]
      return newQueue
    })
  }, [])

  // Cambiar prioridad
  const changePriority = useCallback((id: string, priority: number) => {
    setBuildQueue(prev => prev.map(item => 
      item.id === id ? { ...item, priority } : item
    ))
  }, [])

  // Polling para verificar estado
  const checkBuildStatus = useCallback(async (modelId: string) => {
    try {
      let data: any
      
      // Try to use TruthGPT API if available
      if (apiConnected) {
        try {
          const status = await truthGPTClient.getModelStatus(modelId)
          data = {
            ...status,
            status: status.status === 'completed' ? 'completed' : status.status,
            progress: status.progress || 50,
          }
        } catch (apiError) {
          console.warn('TruthGPT API status error, falling back to legacy API:', apiError)
          // Fallback to legacy API
          const response = await fetch(`/api/model-status/${modelId}`)
          if (response.ok) {
            data = await response.json()
          } else {
            return null
          }
        }
      } else {
        // Use legacy API
        const response = await fetch(`/api/model-status/${modelId}`)
        if (response.ok) {
          data = await response.json()
        } else {
          return null
        }
      }

      if (data.status === 'completed' || data.status === 'failed') {
        if (pollingIntervalRef.current) {
          clearInterval(pollingIntervalRef.current)
          pollingIntervalRef.current = null
        }
        
        const duration = buildStartTimeRef.current 
          ? Date.now() - buildStartTimeRef.current 
          : undefined
        
        setBuildProgress(100)
        buildStartTimeRef.current = null
        
        return { status: data.status, duration }
      }
      if (data.progress) {
        setBuildProgress(data.progress)
      }
    } catch (error) {
      console.error('Error checking build status:', error)
    }
    return null
  }, [apiConnected, truthGPTClient])

  // Procesar siguiente modelo en la cola
  const processNextModel = useCallback(async (retryAttempt: number = 0) => {
    if (buildQueue.length === 0) {
      setIsBuilding(false)
      setCurrentBuild(null)
      setBuildProgress(0)
      return
    }

    setIsBuilding(true)
    const queueItem = buildQueue[0]
    setCurrentBuild(queueItem)
    setBuildQueue(prev => prev.slice(1))
    setBuildProgress(0)
    setRetryCount(retryAttempt)
    buildStartTimeRef.current = Date.now()

    try {
      let result: ProactiveBuildResult

      // Try to use TruthGPT API if available
      if (apiConnected) {
        try {
          const modelName = `truthgpt-${queueItem.description.substring(0, 30).toLowerCase().replace(/[^a-z0-9]+/g, '-')}`
          const apiResult = await createModelFromDescription(queueItem.description, modelName)
          
          result = {
            modelId: apiResult.modelId,
            modelName: apiResult.name,
            status: 'creating',
            description: queueItem.description,
            startTime: buildStartTimeRef.current || Date.now(),
          }
        } catch (apiError) {
          console.warn('TruthGPT API error, falling back to legacy API:', apiError)
          // Fallback to legacy API
          const response = await fetch('/api/create-model-proactive', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              description: queueItem.description,
              optimizationCorePath,
            }),
          })

          if (!response.ok) {
            const errorData = await response.json().catch(() => ({}))
            throw new Error(errorData.error || `HTTP error! status: ${response.status}`)
          }

          result = await response.json()
          result.startTime = buildStartTimeRef.current || Date.now()
        }
      } else {
        // Use legacy API
        const response = await fetch('/api/create-model-proactive', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            description: queueItem.description,
            optimizationCorePath,
          }),
        })

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}))
          throw new Error(errorData.error || `HTTP error! status: ${response.status}`)
        }

        result = await response.json()
        result.startTime = buildStartTimeRef.current || Date.now()
      }
      
      // Iniciar polling para verificar estado
      setBuildProgress(10)
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current)
      }
      
      pollingIntervalRef.current = setInterval(async () => {
        const statusResult = await checkBuildStatus(result.modelId)
        if (statusResult) {
          result.status = statusResult.status as 'completed' | 'failed'
          result.duration = statusResult.duration
          result.endTime = Date.now()
          
          setCompletedBuilds(prev => [...prev, result])
          
          if (onModelCreated) {
            onModelCreated(result)
          }

          if (result.status === 'completed') {
            logger.info('Modelo construido exitosamente', {
              modelName: result.modelName,
              duration: result.duration,
            })
            
            // Agregar al historial inteligente
            smartHistory.addModel(result)
            
            // Notificación mejorada
            enhancedNotifications.createNotification(
              'complete',
              'Modelo Completado',
              `${result.modelName} construido exitosamente en ${Math.round((result.duration || 0) / 1000)}s`,
              { priority: 'high' }
            )
            
            // Crear versión del modelo
            try {
              const config = adaptToOptimizationCore(
                adaptiveAnalyze(result.description),
                result.modelName,
                result.description
              )
              modelVersioning.createVersion(
                result.modelName,
                result,
                config
              )
            } catch (versionError) {
              console.error('Error creating version:', versionError)
            }
            
            // Disparar webhook
            webhookManager.triggerWebhook('model.completed', {
              modelId: result.modelId,
              modelName: result.modelName,
              data: {
                duration: result.duration,
                description: result.description,
              },
            }).catch(error => {
              console.error('Error triggering webhook:', error)
            })
            
            toast.success(`Modelo ${result.modelName} creado exitosamente`, {
              icon: '🎉',
              duration: 5000,
            })
            if (notificationsEnabled) {
              notificationManager.notifyModelCompleted(result.modelName, result.duration)
            }
          } else {
            logger.error('Error al construir modelo', new Error(result.error || 'Error desconocido'), {
              modelName: result.modelName,
            })
            
            // Agregar al historial inteligente
            smartHistory.addModel(result)
            
            // Notificación mejorada
            enhancedNotifications.createNotification(
              'error',
              'Modelo Fallido',
              `${result.modelName}: ${result.error || 'Error desconocido'}`,
              { priority: 'high' }
            )
            
            // Disparar webhook de error
            webhookManager.triggerWebhook('model.failed', {
              modelId: result.modelId,
              modelName: result.modelName,
              data: {
                error: result.error,
                description: result.description,
              },
            }).catch(error => {
              console.error('Error triggering webhook:', error)
            })
            
            toast.error(`Error al crear modelo: ${result.error}`, {
              icon: '❌',
              duration: 5000,
            })
            if (notificationsEnabled) {
              notificationManager.notifyModelFailed(result.modelName, result.error)
            }
          }
          
          // Continuar con el siguiente modelo
          if (isActive && buildQueue.length > 0) {
            setTimeout(() => {
              processNextModel(0)
            }, 1000)
          } else {
            setIsBuilding(false)
            setCurrentBuild(null)
            setBuildProgress(0)
          }
        }
      }, 2000) // Poll cada 2 segundos
      
      // Timeout después de 5 minutos
      setTimeout(() => {
        if (pollingIntervalRef.current) {
          clearInterval(pollingIntervalRef.current)
          pollingIntervalRef.current = null
        }
        if (result.status === 'creating') {
          result.status = 'failed'
          result.error = 'Timeout: El proceso tomó más tiempo del esperado'
          result.duration = Date.now() - (buildStartTimeRef.current || Date.now())
          setCompletedBuilds(prev => [...prev, result])
          setIsBuilding(false)
          setCurrentBuild(null)
          setBuildProgress(0)
        }
      }, 300000) // 5 minutos
      
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Error desconocido'
      
      // Retry automático (máximo 3 intentos)
      if (retryAttempt < 3) {
        toast(`Reintentando... (${retryAttempt + 1}/3)`, { icon: '🔄' })
        setTimeout(() => {
          setBuildQueue(prev => [queueItem, ...prev]) // Re-agregar a la cola
          processNextModel(retryAttempt + 1)
        }, 3000)
        return
      }
      
      const errorResult: ProactiveBuildResult = {
        modelId: `error-${Date.now()}`,
        modelName: `error-model`,
        status: 'failed',
        description: queueItem.description,
        error: errorMessage,
        duration: Date.now() - (buildStartTimeRef.current || Date.now()),
      }
      
      setCompletedBuilds(prev => [...prev, errorResult])
      logger.error('Error al procesar modelo', error instanceof Error ? error : new Error(errorMessage), {
        description: queueItem.description,
        retryAttempt,
      })
      
      toast.error(`Error: ${errorMessage}`, {
        icon: '❌',
        duration: 5000,
      })
      
      // Continuar con el siguiente modelo
      if (isActive && buildQueue.length > 0) {
        setTimeout(() => {
          processNextModel(0)
        }, 2000)
      } else {
        setIsBuilding(false)
        setCurrentBuild(null)
        setBuildProgress(0)
      }
    }
  }, [buildQueue, isActive, optimizationCorePath, onModelCreated, checkBuildStatus])

  // Iniciar construcción en modo batch
  const startBatchBuild = useCallback(async () => {
    if (buildQueue.length === 0) {
      toast.error('No hay modelos en la cola para construir')
      return
    }

    setIsActive(true)
    logger.info('Iniciando construcción en modo batch', {
      queueLength: buildQueue.length,
      concurrency: batchConcurrency,
    })

    const processor = new BatchProcessor({
      maxConcurrency: batchConcurrency,
      retryAttempts: 3,
      onProgress: (completed, total) => {
        setBuildProgress(Math.round((completed / total) * 100))
      },
      onItemComplete: (item, result) => {
        logger.info(`Modelo completado: ${item.description}`, { result })
      },
      onItemError: (item, error) => {
        logger.error(`Error en modelo: ${item.description}`, error)
      },
    })

    try {
      const { results, errors } = await processor.processBatch(
        buildQueue,
        async (item: QueueItem) => {
          const response = await fetch('/api/create-model-proactive', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              description: item.description,
              optimizationCorePath,
            }),
          })

          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`)
          }

          return await response.json()
        }
      )

      // Procesar resultados
      results.forEach(result => {
        setCompletedBuilds(prev => [...prev, result])
        if (onModelCreated) {
          onModelCreated(result)
        }
      })

      // Procesar errores
      errors.forEach(({ item, error }) => {
        const errorResult: ProactiveBuildResult = {
          modelId: `error-${Date.now()}`,
          modelName: `error-${item.description.substring(0, 20)}`,
          status: 'failed',
          description: item.description,
          error: error.message,
        }
        setCompletedBuilds(prev => [...prev, errorResult])
      })

      setBuildQueue([])
      setIsActive(false)
      setBuildProgress(0)

      // Disparar webhook de batch completado
      webhookManager.triggerWebhook('batch.completed', {
        data: {
          successful: results.length,
          errors: errors.length,
          total: results.length + errors.length,
        },
      }).catch(error => {
        console.error('Error triggering webhook:', error)
      })

      toast.success(`Batch completado: ${results.length} exitosos, ${errors.length} errores`, {
        icon: '🎉',
        duration: 5000,
      })

      logger.info('Batch construction completed', {
        successful: results.length,
        errors: errors.length,
      })
    } catch (error) {
      logger.error('Error en construcción batch', error)
      toast.error('Error en construcción batch')
      setIsActive(false)
    }
  }, [buildQueue, batchConcurrency, optimizationCorePath, onModelCreated, logger, webhookManager])

  // Iniciar construcción proactiva
  const startProactive = useCallback(() => {
    if (buildQueue.length === 0) {
      toast.error('No hay modelos en la cola para construir')
      return
    }

    if (batchMode) {
      startBatchBuild()
      return
    }

    setIsActive(true)
    logger.info('Iniciando construcción proactiva', { queueLength: buildQueue.length })
    
    // Disparar webhook de inicio
    if (buildQueue.length > 0) {
      webhookManager.triggerWebhook('model.started', {
        modelName: buildQueue[0].description,
        data: { queueLength: buildQueue.length },
      }).catch(error => {
        console.error('Error triggering webhook:', error)
      })
    }
    
    if (notificationsEnabled && buildQueue.length > 0) {
      notificationManager.notifyBuildStarted(buildQueue[0].description)
    }
    processNextModel()
    toast.success('Construcción proactiva iniciada', {
      icon: '🚀',
    })
  }, [buildQueue.length, processNextModel, notificationsEnabled, notificationManager, batchMode, startBatchBuild, logger, webhookManager])

  // Pausar construcción
  const pauseProactive = useCallback(() => {
    setIsActive(false)
    setIsBuilding(false)
    toast('Construcción pausada', {
      icon: '⏸️',
    })
  }, [])

  // Detener construcción
  const stopProactive = useCallback(() => {
    logger.info('Construcción detenida por usuario', {
      queueLength: buildQueue.length,
      isBuilding,
    })
    setIsActive(false)
    setIsBuilding(false)
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current)
      pollingIntervalRef.current = null
    }
    setBuildQueue([])
    setCurrentBuild(null)
    setBuildProgress(0)
    buildStartTimeRef.current = null
    toast('Construcción detenida', {
      icon: '🛑',
    })
  }, [buildQueue.length, isBuilding, logger])

  // Limpiar completados
  const clearCompleted = useCallback(() => {
    setCompletedBuilds([])
    toast('Historial limpiado', { icon: '🗑️' })
  }, [])

  // Exportar cola
  const exportQueue = useCallback(() => {
    const data = {
      queue: buildQueue,
      timestamp: Date.now(),
    }
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `model-queue-${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(url)
    toast('Cola exportada', { icon: '💾' })
  }, [buildQueue])

  // Importar cola
  const importQueue = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = (e) => {
      try {
        const data = JSON.parse(e.target?.result as string)
        if (data.queue && Array.isArray(data.queue)) {
          setBuildQueue(prev => [...prev, ...data.queue])
          toast('Cola importada exitosamente', { icon: '📥' })
        }
      } catch (error) {
        toast.error('Error al importar cola', { icon: '❌' })
      }
    }
    reader.readAsText(file)
  }, [])

  // Preview de configuración
  const showConfigPreview = useCallback((item: QueueItem) => {
    try {
      // Verificar caché primero
      const cacheKey = `spec-${item.description.substring(0, 50)}`
      let spec = smartCache.get<any>(cacheKey)
      
      if (!spec) {
        spec = adaptiveAnalyze(item.description)
        // Guardar en caché
        smartCache.set(cacheKey, spec, 300000) // 5 minutos
      }
      
      // Validación avanzada
      const validation = validator.validate(spec, item.description)
      if (validation.errors.length > 0) {
        const errorMessages = validation.errors.map(e => e.message).join(', ')
        toast.error(`Validación: ${errorMessages}`, { duration: 5000 })
        logger.warn('Validación fallida en preview', {
          errors: validation.errors,
          description: item.description,
        })
      } else if (validation.warnings.length > 0) {
        const warningMessages = validation.warnings.map(w => w.message).join(', ')
        toast(`Advertencias: ${warningMessages}`, { icon: '⚠️', duration: 5000 })
      }
      
      const config = adaptToOptimizationCore(spec, `truthgpt-${item.description.substring(0, 30)}`, item.description)
      setPreviewItem(item)
      setPreviewConfig(config)
      setShowPreview(true)
      
      logger.info('Preview de configuración generado', {
        description: item.description,
        validation: {
          isValid: validation.isValid,
          errors: validation.errors.length,
          warnings: validation.warnings.length,
        },
      })
    } catch (error) {
      logger.error('Error generating preview', error)
      toast.error('Error al generar preview de configuración')
    }
  }, [validator, logger, smartCache])

  // Confirmar preview y agregar a cola
  const confirmPreview = useCallback(() => {
    if (previewItem) {
      // El item ya está en la cola, solo cerrar preview
      setShowPreview(false)
      setPreviewItem(null)
      setPreviewConfig(null)
      toast.success('Modelo listo para construir', { icon: '✅' })
    }
  }, [previewItem])

  // Keyboard shortcuts
  useEffect(() => {
    shortcutsManager.register({
      keys: ['Ctrl', 'Enter'],
      description: 'Iniciar construcción',
      ctrlKey: true,
      action: () => {
        if (!isActive && buildQueue.length > 0) {
          startProactive()
        }
      },
    })

    shortcutsManager.register({
      keys: ['Ctrl', 'P'],
      description: 'Pausar construcción',
      ctrlKey: true,
      action: () => {
        if (isActive) {
          pauseProactive()
        }
      },
    })

    shortcutsManager.register({
      keys: ['Ctrl', 'S'],
      description: 'Detener construcción',
      ctrlKey: true,
      action: () => {
        if (isActive || buildQueue.length > 0) {
          stopProactive()
        }
      },
    })

    shortcutsManager.register({
      keys: ['Ctrl', 'T'],
      description: 'Toggle plantillas',
      ctrlKey: true,
      action: () => {
        setShowTemplates(!showTemplates)
      },
    })

    shortcutsManager.register({
      keys: ['Ctrl', '?'],
      description: 'Mostrar ayuda',
      ctrlKey: true,
      action: () => {
        setShowHelp(!showHelp)
      },
    })

    const handleKeyDown = (e: KeyboardEvent) => {
      shortcutsManager.handleKeyDown(e)
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => {
      window.removeEventListener('keydown', handleKeyDown)
      shortcutsManager.clear()
    }
  }, [isActive, buildQueue.length, showTemplates, shortcutsManager, startProactive, pauseProactive, stopProactive])

  // Backup automático
  useEffect(() => {
    const stopBackup = backupManager.startAutoBackup(
      () => buildQueue,
      () => completedBuilds,
      () => ({
        autoMode,
        batchMode,
        batchConcurrency,
        notificationsEnabled,
      })
    )

    return () => {
      if (stopBackup) {
        stopBackup()
      } else {
        backupManager.stopAutoBackup()
      }
    }
  }, [buildQueue, completedBuilds, autoMode, batchMode, batchConcurrency, notificationsEnabled, backupManager])

  // Exportar reporte
  const exportReport = useCallback((format: 'json' | 'csv' | 'html') => {
    try {
      const report = reportGenerator.generateReport(
        completedBuilds,
        buildQueue,
        'Reporte de Modelos TruthGPT'
      )

      if (format === 'json') {
        reportGenerator.exportJSON(report)
      } else if (format === 'csv') {
        reportGenerator.exportCSV(completedBuilds)
      } else if (format === 'html') {
        reportGenerator.exportHTML(report)
      }

      logger.info('Reporte exportado', { format })
      toast.success(`Reporte exportado como ${format.toUpperCase()}`, { icon: '📄' })
    } catch (error) {
      logger.error('Error exportando reporte', error)
      toast.error('Error al exportar reporte')
    }
  }, [completedBuilds, buildQueue, reportGenerator, logger])

  // Restaurar desde backup
  const restoreFromBackup = useCallback(async () => {
    const latestBackup = backupManager.getLatestBackup()
    if (!latestBackup) {
      toast.error('No hay backups disponibles')
      return
    }

    if (window.confirm('¿Restaurar desde el backup más reciente? Esto reemplazará la cola y modelos actuales.')) {
      const restored = backupManager.restoreFromBackup(latestBackup)
      setBuildQueue(restored.queue)
      setCompletedBuilds(restored.completedModels)
      if (restored.settings) {
        setAutoMode(restored.settings.autoMode || false)
        setBatchMode(restored.settings.batchMode || false)
        setBatchConcurrency(restored.settings.batchConcurrency || 3)
        setNotificationsEnabled(restored.settings.notificationsEnabled || false)
      }
      logger.info('Sistema restaurado desde backup', { timestamp: latestBackup.timestamp })
      toast.success('Sistema restaurado desde backup', { icon: '🔄' })
    }
  }, [backupManager, logger])

  // Cleanup al desmontar
  useEffect(() => {
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current)
      }
      shortcutsManager.clear()
      backupManager.stopAutoBackup()
    }
  }, [shortcutsManager, backupManager])

  // Cargar tema al montar
  useEffect(() => {
    const theme = themeManager.getTheme()
    setCurrentTheme(theme)
  }, [themeManager])

  // Toggle tema
  const toggleTheme = useCallback(() => {
    const effective = themeManager.getEffectiveTheme()
    themeManager.setTheme(effective === 'dark' ? 'light' : 'dark')
    setCurrentTheme(themeManager.getTheme())
  }, [themeManager])

  // Copiar ID de modelo
  const copyModelId = useCallback((modelId: string) => {
    navigator.clipboard.writeText(modelId).then(() => {
      setCopiedId(modelId)
      toast.success('ID copiado al portapapeles', { icon: '📋' })
      setTimeout(() => setCopiedId(null), 2000)
    }).catch(() => {
      toast.error('Error al copiar ID')
    })
  }, [])

  // Compartir modelo
  const shareModel = useCallback((model: ProactiveBuildResult) => {
    const shareData = {
      title: `Modelo TruthGPT: ${model.modelName}`,
      text: `Modelo ${model.modelName} - ${model.description}`,
      url: window.location.href,
    }

    if (navigator.share) {
      navigator.share(shareData).catch(() => {
        // Usuario canceló o error
      })
    } else {
      // Fallback: copiar al portapapeles
      copyModelId(model.modelId)
    }
  }, [copyModelId])

  // Registrar comandos rápidos y shortcuts mejorados
  useEffect(() => {
    // Comandos rápidos
    quickCommands.registerCommand({
      id: 'start-build',
      name: 'Iniciar Construcción',
      description: 'Inicia la construcción proactiva',
      shortcut: 'Ctrl+Enter',
      icon: '🚀',
      category: 'build',
      action: () => {
        if (!isActive && buildQueue.length > 0) {
          startProactive()
        }
      },
    })

    quickCommands.registerCommand({
      id: 'pause-build',
      name: 'Pausar Construcción',
      description: 'Pausa la construcción actual',
      shortcut: 'Ctrl+P',
      icon: '⏸️',
      category: 'build',
      action: () => {
        if (isActive) {
          pauseProactive()
        }
      },
    })

    quickCommands.registerCommand({
      id: 'toggle-stats',
      name: 'Toggle Estadísticas',
      description: 'Muestra/oculta estadísticas',
      shortcut: 'Ctrl+S',
      icon: '📊',
      category: 'view',
      action: () => {
        setShowStats(!showStats)
      },
    })

    // Shortcuts mejorados
    enhancedShortcuts.registerShortcut({
      id: 'help',
      keys: ['Ctrl', '?'],
      description: 'Mostrar ayuda',
      category: 'general',
      action: () => {
        setShowHelp(!showHelp)
      },
    })

    enhancedShortcuts.registerShortcut({
      id: 'toggle-favorites',
      keys: ['Ctrl', 'F'],
      description: 'Toggle favoritos',
      category: 'view',
      action: () => {
        setShowFavorites(!showFavorites)
      },
    })

    enhancedShortcuts.registerShortcut({
      id: 'toggle-history',
      keys: ['Ctrl', 'H'],
      description: 'Toggle historial',
      category: 'view',
      action: () => {
        setShowSmartHistory(!showSmartHistory)
      },
    })

    enhancedShortcuts.registerShortcut({
      id: 'toggle-commands',
      keys: ['Ctrl', 'K'],
      description: 'Toggle comandos rápidos',
      category: 'view',
      action: () => {
        setShowQuickCommands(!showQuickCommands)
      },
    })

    return () => {
      quickCommands.clear()
      enhancedShortcuts.clear()
    }
  }, [quickCommands, enhancedShortcuts, isActive, buildQueue.length, showStats, showFavorites, showSmartHistory, showQuickCommands, startProactive, pauseProactive])

  // Modo automático: construir automáticamente cuando se agrega a la cola
  useEffect(() => {
    if (autoMode && buildQueue.length > 0 && !isBuilding && !isActive) {
      startProactive()
    }
  }, [autoMode, buildQueue.length, isBuilding, isActive, startProactive])

  // Manejar submit
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    addToQueue(input)
    setInput('')
  }

  return (
    <div className="bg-slate-800/50 backdrop-blur-sm rounded-lg border border-slate-700 p-6 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-purple-600/20 rounded-lg">
            <Zap className="w-5 h-5 text-purple-400" />
          </div>
          <div>
            <h3 className="text-lg font-bold text-white">Constructor Proactivo</h3>
            <p className="text-sm text-slate-400">
              Construye modelos continuamente adaptados a optimization_core
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={toggleTheme}
            className="p-2 bg-slate-700/50 hover:bg-slate-700 rounded-lg transition-colors text-slate-300"
            title={`Cambiar a tema ${themeManager.getEffectiveTheme() === 'dark' ? 'claro' : 'oscuro'}`}
          >
            {themeManager.getEffectiveTheme() === 'dark' ? (
              <Sun className="w-4 h-4" />
            ) : (
              <Moon className="w-4 h-4" />
            )}
          </button>
          <button
            onClick={enableNotifications}
            className={`p-2 rounded-lg transition-colors ${
              notificationsEnabled
                ? 'bg-green-600/20 text-green-400'
                : 'bg-slate-700/50 text-slate-400 hover:bg-slate-700'
            }`}
            title={notificationsEnabled ? 'Notificaciones activas' : 'Habilitar notificaciones'}
          >
            {notificationsEnabled ? <Bell className="w-4 h-4" /> : <BellOff className="w-4 h-4" />}
          </button>
          <button
            onClick={() => setShowWebhooks(!showWebhooks)}
            className="p-2 bg-slate-700/50 hover:bg-slate-700 rounded-lg transition-colors text-slate-300"
            title="Configurar webhooks"
          >
            <Webhook className="w-4 h-4" />
          </button>
          <button
            onClick={() => setShowABTesting(!showABTesting)}
            className="p-2 bg-slate-700/50 hover:bg-slate-700 rounded-lg transition-colors text-slate-300"
            title="A/B Testing de configuraciones"
          >
            <TestTube className="w-4 h-4" />
          </button>
          <button
            onClick={() => setShowCustomTemplates(!showCustomTemplates)}
            className="p-2 bg-slate-700/50 hover:bg-slate-700 rounded-lg transition-colors text-slate-300"
            title="Plantillas personalizadas"
          >
            <BookOpen className="w-4 h-4" />
          </button>
          <button
            onClick={() => setShowAdvancedStats(!showAdvancedStats)}
            className="p-2 bg-slate-700/50 hover:bg-slate-700 rounded-lg transition-colors text-slate-300"
            title="Estadísticas avanzadas"
          >
            <BarChart2 className="w-4 h-4" />
          </button>
          <div className="relative">
            <button
              onClick={() => setShowExportMenu(!showExportMenu)}
              className="p-2 bg-slate-700/50 hover:bg-slate-700 rounded-lg transition-colors text-slate-300"
              title="Exportar modelos"
            >
              <Download className="w-4 h-4" />
            </button>
            {showExportMenu && (
              <div className="absolute right-0 top-full mt-2 bg-slate-800 border border-slate-700 rounded-lg shadow-xl z-50 min-w-[200px]">
                <div className="p-2">
                  <button
                    onClick={async () => {
                      const blob = await modelExporter.exportModels(completedBuilds, { format: 'json' })
                      modelExporter.downloadFile(blob, `models-${Date.now()}.json`)
                      setShowExportMenu(false)
                      toast.success('Exportado como JSON', { icon: '📥' })
                    }}
                    className="w-full text-left px-3 py-2 hover:bg-slate-700 rounded text-sm text-slate-300"
                  >
                    📄 JSON
                  </button>
                  <button
                    onClick={async () => {
                      const blob = await modelExporter.exportModels(completedBuilds, { format: 'csv' })
                      modelExporter.downloadFile(blob, `models-${Date.now()}.csv`)
                      setShowExportMenu(false)
                      toast.success('Exportado como CSV', { icon: '📊' })
                    }}
                    className="w-full text-left px-3 py-2 hover:bg-slate-700 rounded text-sm text-slate-300"
                  >
                    📊 CSV
                  </button>
                  <button
                    onClick={async () => {
                      const blob = await modelExporter.exportModels(completedBuilds, { format: 'yaml' })
                      modelExporter.downloadFile(blob, `models-${Date.now()}.yaml`)
                      setShowExportMenu(false)
                      toast.success('Exportado como YAML', { icon: '📝' })
                    }}
                    className="w-full text-left px-3 py-2 hover:bg-slate-700 rounded text-sm text-slate-300"
                  >
                    📝 YAML
                  </button>
                  <button
                    onClick={async () => {
                      const blob = await modelExporter.exportModels(completedBuilds, { format: 'markdown' })
                      modelExporter.downloadFile(blob, `models-${Date.now()}.md`)
                      setShowExportMenu(false)
                      toast.success('Exportado como Markdown', { icon: '📄' })
                    }}
                    className="w-full text-left px-3 py-2 hover:bg-slate-700 rounded text-sm text-slate-300"
                  >
                    📄 Markdown
                  </button>
                  <button
                    onClick={async () => {
                      const blob = await modelExporter.exportModels(completedBuilds, { format: 'html' })
                      modelExporter.downloadFile(blob, `models-${Date.now()}.html`)
                      setShowExportMenu(false)
                      toast.success('Exportado como HTML', { icon: '🌐' })
                    }}
                    className="w-full text-left px-3 py-2 hover:bg-slate-700 rounded text-sm text-slate-300"
                  >
                    🌐 HTML
                  </button>
                </div>
              </div>
            )}
          </div>
          <button
            onClick={() => setShowFavorites(!showFavorites)}
            className={`p-2 rounded-lg transition-colors ${
              showFavorites
                ? 'bg-red-600/20 text-red-400'
                : 'bg-slate-700/50 text-slate-300 hover:bg-slate-700'
            }`}
            title="Favoritos"
          >
            <Heart className={`w-4 h-4 ${showFavorites ? 'fill-current' : ''}`} />
          </button>
          <button
            onClick={() => setShowSmartHistory(!showSmartHistory)}
            className="p-2 bg-slate-700/50 hover:bg-slate-700 rounded-lg transition-colors text-slate-300 relative"
            title="Historial Inteligente"
          >
            <History className="w-4 h-4" />
            {smartHistory.getAllModels().length > 0 && (
              <span className="absolute -top-1 -right-1 w-2 h-2 bg-purple-500 rounded-full"></span>
            )}
          </button>
          <button
            onClick={() => setShowQuickCommands(!showQuickCommands)}
            className="p-2 bg-slate-700/50 hover:bg-slate-700 rounded-lg transition-colors text-slate-300"
            title="Comandos Rápidos (Ctrl+K)"
          >
            <Command className="w-4 h-4" />
          </button>
          <button
            onClick={() => setShowHelp(!showHelp)}
            className="p-2 bg-slate-700/50 hover:bg-slate-700 rounded-lg transition-colors text-slate-300"
            title="Ayuda Contextual (Ctrl+?)"
          >
            <HelpCircle className="w-4 h-4" />
          </button>
          <button
            onClick={() => setShowRealtimeMetrics(!showRealtimeMetrics)}
            className="p-2 bg-slate-700/50 hover:bg-slate-700 rounded-lg transition-colors text-slate-300 relative"
            title="Métricas en Tiempo Real"
          >
            <Gauge className="w-4 h-4" />
            {realtimeData && realtimeData.buildsPerMinute > 0 && (
              <span className="absolute -top-1 -right-1 w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
            )}
          </button>
          <button
            onClick={() => setShowAlerts(!showAlerts)}
            className="p-2 bg-slate-700/50 hover:bg-slate-700 rounded-lg transition-colors text-slate-300 relative"
            title="Alertas Inteligentes"
          >
            <AlertTriangle className="w-4 h-4" />
            {intelligentAlerts.getUnacknowledgedAlerts().length > 0 && (
              <span className="absolute -top-1 -right-1 w-2 h-2 bg-red-500 rounded-full animate-pulse"></span>
            )}
          </button>
          <button
            onClick={() => setShowTemplates(!showTemplates)}
            className="p-2 bg-slate-700/50 hover:bg-slate-700 rounded-lg transition-colors text-slate-300"
            title="Plantillas de modelos"
          >
            <Sparkles className="w-4 h-4" />
          </button>
          <label className="flex items-center gap-2 text-sm text-slate-300">
            <input
              type="checkbox"
              checked={autoMode}
              onChange={(e) => setAutoMode(e.target.checked)}
              className="rounded"
            />
            <span>Auto</span>
          </label>
          <label className="flex items-center gap-2 text-sm text-slate-300">
            <input
              type="checkbox"
              checked={batchMode}
              onChange={(e) => setBatchMode(e.target.checked)}
              className="rounded"
            />
            <span>Batch</span>
          </label>
          {batchMode && (
            <div className="flex items-center gap-2">
              <Cpu className="w-4 h-4 text-slate-400" />
              <input
                type="number"
                min="1"
                max="10"
                value={batchConcurrency}
                onChange={(e) => setBatchConcurrency(Math.max(1, Math.min(10, parseInt(e.target.value) || 1)))}
                className="w-16 px-2 py-1 bg-slate-700/50 border border-slate-600 rounded text-white text-xs"
              />
            </div>
          )}
        </div>
      </div>

      {/* Estado y Estadísticas */}
      <div className="space-y-3">
        <div className="flex items-center gap-4 p-3 bg-slate-700/30 rounded-lg flex-wrap">
          <div className="flex items-center gap-2">
            {isActive ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin text-green-400" />
                <span className="text-sm text-green-400 font-medium">Activo</span>
              </>
            ) : (
              <>
                <Square className="w-4 h-4 text-slate-400" />
                <span className="text-sm text-slate-400">Inactivo</span>
              </>
            )}
          </div>
          <div className="text-sm text-slate-400">
            Cola: <span className="font-medium text-white">{buildQueue.length}</span>
          </div>
          <div className="text-sm text-slate-400">
            Completados: <span className="font-medium text-white">{stats.total}</span>
          </div>
          <div className="text-sm text-slate-400">
            Éxito: <span className="font-medium text-green-400">{stats.successful}</span>
          </div>
          <div className="text-sm text-slate-400">
            Fallidos: <span className="font-medium text-red-400">{stats.failed}</span>
          </div>
          {stats.total > 0 && (
            <div className="text-sm text-slate-400">
              Tasa: <span className="font-medium text-purple-400">{stats.successRate.toFixed(1)}%</span>
            </div>
          )}
          <div className="ml-auto flex items-center gap-1">
            <button
              onClick={() => setShowStats(!showStats)}
              className="flex items-center gap-1 px-2 py-1 text-xs bg-slate-600/50 hover:bg-slate-600 rounded transition-colors"
            >
              <BarChart3 className="w-3 h-3" />
              <span>Estadísticas</span>
            </button>
            {completedBuilds.length > 0 && (
              <>
                <button
                  onClick={() => setShowMetricsChart(!showMetricsChart)}
                  className="flex items-center gap-1 px-2 py-1 text-xs bg-slate-600/50 hover:bg-slate-600 rounded transition-colors"
                >
                  <TrendingUp className="w-3 h-3" />
                  <span>Gráficos</span>
                </button>
                <button
                  onClick={() => setShowAdvancedDashboard(!showAdvancedDashboard)}
                  className="flex items-center gap-1 px-2 py-1 text-xs bg-slate-600/50 hover:bg-slate-600 rounded transition-colors"
                >
                  <BarChart3 className="w-3 h-3" />
                  <span>Dashboard</span>
                </button>
              </>
            )}
          </div>
        </div>

        {/* Barra de Progreso */}
        {isBuilding && currentBuild && (
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs text-slate-400">
              <span className="truncate max-w-md">{currentBuild.description}</span>
              <span>{buildProgress}%</span>
            </div>
            <div className="w-full bg-slate-700 rounded-full h-2 overflow-hidden">
              <motion.div
                className="h-full bg-gradient-to-r from-purple-600 to-pink-600"
                initial={{ width: 0 }}
                animate={{ width: `${buildProgress}%` }}
                transition={{ duration: 0.3 }}
              />
            </div>
          </div>
        )}

        {/* Panel de Estadísticas */}
        {showStats && stats.total > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="p-3 bg-slate-700/30 rounded-lg space-y-2"
          >
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div>
                <span className="text-slate-400">Promedio:</span>
                <span className="ml-2 text-white font-medium">
                  {Math.round(stats.avgDuration / 1000)}s
                </span>
              </div>
              <div>
                <span className="text-slate-400">Tasa de éxito:</span>
                <span className="ml-2 text-purple-400 font-medium">
                  {stats.successRate.toFixed(1)}%
                </span>
              </div>
            </div>
          </motion.div>
        )}
      </div>

      {/* Controles */}
      <div className="flex items-center gap-2 flex-wrap">
        {!isActive ? (
          <button
            onClick={startProactive}
            disabled={buildQueue.length === 0}
            className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg transition-colors text-white font-medium"
          >
            <Play className="w-4 h-4" />
            <span>Iniciar</span>
          </button>
        ) : (
          <button
            onClick={pauseProactive}
            className="flex items-center gap-2 px-4 py-2 bg-yellow-600 hover:bg-yellow-700 rounded-lg transition-colors text-white font-medium"
          >
            <Pause className="w-4 h-4" />
            <span>Pausar</span>
          </button>
        )}
        <button
          onClick={stopProactive}
          disabled={!isActive && buildQueue.length === 0}
          className="flex items-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg transition-colors text-white font-medium"
        >
          <Square className="w-4 h-4" />
          <span>Detener</span>
        </button>
        <div className="flex items-center gap-1 ml-auto">
          <button
            onClick={exportQueue}
            disabled={buildQueue.length === 0}
            className="p-2 bg-slate-700/50 hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg transition-colors"
            title="Exportar cola"
          >
            <Download className="w-4 h-4 text-slate-300" />
          </button>
          <label className="p-2 bg-slate-700/50 hover:bg-slate-700 rounded-lg transition-colors cursor-pointer" title="Importar cola">
            <Upload className="w-4 h-4 text-slate-300" />
            <input type="file" accept=".json" onChange={importQueue} className="hidden" />
          </label>
          {completedBuilds.length > 0 && (
            <>
              <button
                onClick={() => setShowComparator(true)}
                disabled={completedBuilds.length < 2}
                className="p-2 bg-slate-700/50 hover:bg-slate-700 disabled:opacity-30 disabled:cursor-not-allowed rounded-lg transition-colors"
                title="Comparar modelos"
              >
                <GitCompare className="w-4 h-4 text-slate-300" />
              </button>
              <button
                onClick={() => setShowLogs(!showLogs)}
                className="p-2 bg-slate-700/50 hover:bg-slate-700 rounded-lg transition-colors"
                title="Ver logs"
              >
                <LogsIcon className="w-4 h-4 text-slate-300" />
              </button>
              <div className="relative group">
                <button
                  className="p-2 bg-slate-700/50 hover:bg-slate-700 rounded-lg transition-colors"
                  title="Exportar reporte"
                >
                  <FileDown className="w-4 h-4 text-slate-300" />
                </button>
                <div className="absolute right-0 top-full mt-1 bg-slate-700 rounded-lg shadow-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-50 min-w-[120px]">
                  <button
                    onClick={() => exportReport('json')}
                    className="w-full px-3 py-2 text-left text-sm text-slate-300 hover:bg-slate-600 rounded-t-lg flex items-center gap-2"
                  >
                    <FileText className="w-3 h-3" />
                    JSON
                  </button>
                  <button
                    onClick={() => exportReport('csv')}
                    className="w-full px-3 py-2 text-left text-sm text-slate-300 hover:bg-slate-600 flex items-center gap-2"
                  >
                    <FileSpreadsheet className="w-3 h-3" />
                    CSV
                  </button>
                  <button
                    onClick={() => exportReport('html')}
                    className="w-full px-3 py-2 text-left text-sm text-slate-300 hover:bg-slate-600 rounded-b-lg flex items-center gap-2"
                  >
                    <FileText className="w-3 h-3" />
                    HTML/PDF
                  </button>
                </div>
              </div>
              <button
                onClick={restoreFromBackup}
                className="p-2 bg-slate-700/50 hover:bg-slate-700 rounded-lg transition-colors"
                title="Restaurar desde backup"
              >
                <RotateCcw className="w-4 h-4 text-slate-300" />
              </button>
              <button
                onClick={clearCompleted}
                className="p-2 bg-slate-700/50 hover:bg-slate-700 rounded-lg transition-colors"
                title="Limpiar completados"
              >
                <Trash2 className="w-4 h-4 text-slate-300" />
              </button>
            </>
          )}
        </div>
      </div>

      {/* Lista de Cola */}
      {buildQueue.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm font-semibold text-slate-300">Cola de Modelos</h4>
          <div className="space-y-2 max-h-40 overflow-y-auto">
            {buildQueue.map((item, index) => (
              <motion.div
                key={item.id}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                className="flex items-center gap-2 p-2 bg-slate-700/30 rounded-lg border border-slate-600"
              >
                <span className="text-xs text-slate-400 w-6">{index + 1}</span>
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-white truncate">{item.description}</p>
                  <div className="flex items-center gap-2 mt-1">
                    {item.priority > 0 && (
                      <span className="text-xs text-purple-400">⭐ {item.priority}</span>
                    )}
                    {item.category && (
                      <span className="text-xs px-2 py-0.5 bg-purple-600/20 text-purple-400 rounded">
                        {item.category}
                      </span>
                    )}
                    {item.tags && item.tags.length > 0 && (
                      <div className="flex items-center gap-1">
                        {item.tags.slice(0, 2).map((tag) => (
                          <span key={tag} className="text-xs px-2 py-0.5 bg-slate-600/50 text-slate-300 rounded">
                            {tag}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-1">
                  <button
                    onClick={() => moveInQueue(item.id, 'up')}
                    disabled={index === 0}
                    className="p-1 hover:bg-slate-600 disabled:opacity-30 disabled:cursor-not-allowed rounded transition-colors"
                    title="Mover arriba"
                  >
                    <ArrowUp className="w-3 h-3 text-slate-300" />
                  </button>
                  <button
                    onClick={() => moveInQueue(item.id, 'down')}
                    disabled={index === buildQueue.length - 1}
                    className="p-1 hover:bg-slate-600 disabled:opacity-30 disabled:cursor-not-allowed rounded transition-colors"
                    title="Mover abajo"
                  >
                    <ArrowDown className="w-3 h-3 text-slate-300" />
                  </button>
                  <button
                    onClick={() => showConfigPreview(item)}
                    className="p-1 hover:bg-blue-600/20 rounded transition-colors"
                    title="Preview de configuración"
                  >
                    <Eye className="w-3 h-3 text-blue-400" />
                  </button>
                  <button
                    onClick={() => removeFromQueue(item.id)}
                    className="p-1 hover:bg-red-600/20 rounded transition-colors"
                    title="Eliminar"
                  >
                    <Trash2 className="w-3 h-3 text-red-400" />
                  </button>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      )}

      {/* Input para agregar modelos */}
      <form onSubmit={handleSubmit} className="space-y-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Describe un modelo para agregar a la cola..."
          className="w-full bg-slate-700/50 border border-slate-600 rounded-lg px-4 py-2 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
        />
        <button
          type="submit"
          className="w-full px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg transition-colors text-white font-medium"
        >
          Agregar a Cola
        </button>
      </form>

      {/* Panel de Plantillas */}
      <AnimatePresence>
        {showTemplates && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="bg-slate-700/30 rounded-lg p-4 space-y-3"
          >
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-semibold text-slate-300">Plantillas de Modelos</h4>
              <button
                onClick={() => setShowTemplates(false)}
                className="p-1 hover:bg-slate-600 rounded transition-colors"
              >
                <X className="w-4 h-4 text-slate-400" />
              </button>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2 max-h-60 overflow-y-auto">
              {MODEL_TEMPLATES.map((template) => (
                <button
                  key={template.id}
                  onClick={() => addTemplateToQueue(template)}
                  className="p-3 bg-slate-600/30 hover:bg-slate-600/50 rounded-lg text-left transition-colors border border-slate-600"
                >
                  <div className="flex items-start justify-between mb-1">
                    <span className="text-sm font-medium text-white">{template.name}</span>
                    {template.priority > 5 && (
                      <span className="text-xs text-purple-400">⭐</span>
                    )}
                  </div>
                  <p className="text-xs text-slate-400 mb-2">{template.description}</p>
                  <div className="flex items-center gap-1 flex-wrap">
                    <span className="text-xs px-2 py-0.5 bg-purple-600/20 text-purple-400 rounded">
                      {template.category}
                    </span>
                    {template.tags.slice(0, 2).map((tag) => (
                      <span key={tag} className="text-xs px-2 py-0.5 bg-slate-600/50 text-slate-300 rounded">
                        {tag}
                      </span>
                    ))}
                  </div>
                </button>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Dashboard Avanzado */}
      <AnimatePresence>
        {showAdvancedDashboard && completedBuilds.length > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="bg-slate-700/30 rounded-lg p-4 space-y-3"
          >
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-semibold text-slate-300">Dashboard Avanzado</h4>
              <button
                onClick={() => setShowAdvancedDashboard(false)}
                className="p-1 hover:bg-slate-600 rounded transition-colors"
              >
                <X className="w-4 h-4 text-slate-400" />
              </button>
            </div>
            <AdvancedDashboard
              models={completedBuilds}
              queueLength={buildQueue.length}
              isActive={isActive}
              stats={stats}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Gráficos de Métricas */}
      <AnimatePresence>
        {showMetricsChart && chartData.length > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="bg-slate-700/30 rounded-lg p-4 space-y-3"
          >
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-semibold text-slate-300">Métricas de Rendimiento</h4>
              <button
                onClick={() => setShowMetricsChart(false)}
                className="p-1 hover:bg-slate-600 rounded transition-colors"
              >
                <X className="w-4 h-4 text-slate-400" />
              </button>
            </div>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
                <XAxis dataKey="name" stroke="#94a3b8" fontSize={12} />
                <YAxis stroke="#94a3b8" fontSize={12} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: '8px' }}
                />
                <Legend />
                <Bar dataKey="duration" fill="#8b5cf6" name="Duración (s)" />
              </BarChart>
            </ResponsiveContainer>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Búsqueda y Filtros */}
      {completedBuilds.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-slate-400" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Buscar modelos..."
                className="w-full bg-slate-700/50 border border-slate-600 rounded-lg pl-10 pr-4 py-2 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-purple-500 text-sm"
              />
              {searchQuery && (
                <button
                  onClick={() => setSearchQuery('')}
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 text-slate-400 hover:text-white"
                >
                  <X className="w-4 h-4" />
                </button>
              )}
            </div>
            <div className="flex items-center gap-1 bg-slate-700/50 rounded-lg p-1">
              <button
                onClick={() => setFilterStatus('all')}
                className={`px-3 py-1 rounded text-xs transition-colors ${
                  filterStatus === 'all'
                    ? 'bg-purple-600 text-white'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                Todos
              </button>
              <button
                onClick={() => setFilterStatus('completed')}
                className={`px-3 py-1 rounded text-xs transition-colors ${
                  filterStatus === 'completed'
                    ? 'bg-green-600 text-white'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                Éxito
              </button>
              <button
                onClick={() => setFilterStatus('failed')}
                className={`px-3 py-1 rounded text-xs transition-colors ${
                  filterStatus === 'failed'
                    ? 'bg-red-600 text-white'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                Fallidos
              </button>
            </div>
          </div>
          {(searchQuery || filterStatus !== 'all') && (
            <div className="text-xs text-slate-400">
              Mostrando {filteredCompleted.length} de {completedBuilds.length} modelos
            </div>
          )}
        </div>
      )}

      {/* Botón para mostrar gráficos */}
      {completedBuilds.length > 0 && !showMetricsChart && (
        <button
          onClick={() => setShowMetricsChart(true)}
          className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-slate-700/50 hover:bg-slate-700 rounded-lg transition-colors text-slate-300 text-sm"
        >
          <BarChart3 className="w-4 h-4" />
          <span>Mostrar Gráficos de Métricas</span>
        </button>
      )}

      {/* Lista de modelos completados */}
      {filteredCompleted.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm font-semibold text-slate-300">
            Modelos Completados {filteredCompleted.length !== completedBuilds.length && `(${filteredCompleted.length})`}
          </h4>
          <div className="space-y-2 max-h-60 overflow-y-auto">
            <AnimatePresence>
              {filteredCompleted.map((build) => (
                <motion.div
                  key={build.modelId}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="flex items-center justify-between p-3 bg-slate-700/30 rounded-lg border border-slate-600"
                >
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      {build.status === 'completed' ? (
                        <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0" />
                      ) : (
                        <XCircle className="w-4 h-4 text-red-400 flex-shrink-0" />
                      )}
                      <span className="text-sm font-medium text-white truncate">
                        {build.modelName}
                      </span>
                    </div>
                    <p className="text-xs text-slate-400 truncate">{build.description}</p>
                    {build.duration && (
                      <p className="text-xs text-slate-500 mt-1">
                        Duración: {Math.round(build.duration / 1000)}s
                      </p>
                    )}
                    {build.error && (
                      <p className="text-xs text-red-400 mt-1">{build.error}</p>
                    )}
                    {/* Versiones del modelo */}
                    {modelVersioning.getVersions(build.modelName).length > 0 && (
                      <div className="flex items-center gap-1 mt-1">
                        <GitBranch className="w-3 h-3 text-slate-500" />
                        <span className="text-xs text-slate-500">
                          {modelVersioning.getVersions(build.modelName).length} versión(es)
                        </span>
                      </div>
                    )}
                  </div>
                  <div className="flex items-center gap-1">
                    <button
                      onClick={() => copyModelId(build.modelId)}
                      className="p-1 hover:bg-slate-600 rounded transition-colors"
                      title="Copiar ID"
                    >
                      {copiedId === build.modelId ? (
                        <Check className="w-3 h-3 text-green-400" />
                      ) : (
                        <Copy className="w-3 h-3 text-slate-400" />
                      )}
                    </button>
                    <button
                      onClick={() => shareModel(build)}
                      className="p-1 hover:bg-slate-600 rounded transition-colors"
                      title="Compartir modelo"
                    >
                      <Share2 className="w-3 h-3 text-slate-400" />
                    </button>
                    <button
                      onClick={() => {
                        if (favoritesManager.isFavorite(build.modelId)) {
                          favoritesManager.removeFavorite(build.modelId)
                          toast('Eliminado de favoritos', { icon: '💔' })
                        } else {
                          favoritesManager.addFavorite(build)
                          toast('Agregado a favoritos', { icon: '❤️' })
                        }
                      }}
                      className={`p-1 hover:bg-slate-600 rounded transition-colors ${
                        favoritesManager.isFavorite(build.modelId) ? 'text-red-400' : 'text-slate-400'
                      }`}
                      title={favoritesManager.isFavorite(build.modelId) ? 'Quitar de favoritos' : 'Agregar a favoritos'}
                    >
                      <Heart className={`w-3 h-3 ${favoritesManager.isFavorite(build.modelId) ? 'fill-current' : ''}`} />
                    </button>
                    {modelVersioning.getVersions(build.modelName).length > 0 && (
                      <button
                        onClick={() => setShowVersioning(true)}
                        className="p-1 hover:bg-slate-600 rounded transition-colors"
                        title="Ver versiones"
                      >
                        <GitBranch className="w-3 h-3 text-purple-400" />
                      </button>
                    )}
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </div>
      )}

      {/* Preview de Configuración */}
      {showPreview && previewItem && previewConfig && (
        <ConfigPreview
          config={previewConfig}
          modelName={`truthgpt-${previewItem.description.substring(0, 30)}`}
          description={previewItem.description}
          onClose={() => {
            setShowPreview(false)
            setPreviewItem(null)
            setPreviewConfig(null)
          }}
          onConfirm={confirmPreview}
        />
      )}

      {/* Panel de Ayuda */}
      <AnimatePresence>
        {showHelp && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="bg-slate-700/30 rounded-lg p-4 space-y-3"
          >
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-semibold text-slate-300">Atajos de Teclado</h4>
              <button
                onClick={() => setShowHelp(false)}
                className="p-1 hover:bg-slate-600 rounded transition-colors"
              >
                <X className="w-4 h-4 text-slate-400" />
              </button>
            </div>
            <div className="space-y-2 text-xs">
              <div className="flex items-center justify-between p-2 bg-slate-600/30 rounded">
                <span className="text-slate-300">Ctrl + Enter</span>
                <span className="text-slate-400">Iniciar construcción</span>
              </div>
              <div className="flex items-center justify-between p-2 bg-slate-600/30 rounded">
                <span className="text-slate-300">Ctrl + P</span>
                <span className="text-slate-400">Pausar construcción</span>
              </div>
              <div className="flex items-center justify-between p-2 bg-slate-600/30 rounded">
                <span className="text-slate-300">Ctrl + S</span>
                <span className="text-slate-400">Detener construcción</span>
              </div>
              <div className="flex items-center justify-between p-2 bg-slate-600/30 rounded">
                <span className="text-slate-300">Ctrl + T</span>
                <span className="text-slate-400">Toggle plantillas</span>
              </div>
              <div className="flex items-center justify-between p-2 bg-slate-600/30 rounded">
                <span className="text-slate-300">Ctrl + ?</span>
                <span className="text-slate-400">Mostrar/ocultar ayuda</span>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Comparador de Modelos */}
      {showComparator && completedBuilds.length >= 2 && (
        <ModelComparator
          models={completedBuilds}
          onClose={() => setShowComparator(false)}
        />
      )}

      {/* Configuración de Webhooks */}
      {showWebhooks && (
        <WebhookConfigPanel onClose={() => setShowWebhooks(false)} />
      )}

      {/* A/B Testing Panel */}
      {showABTesting && (
        <ABTestingPanel onClose={() => setShowABTesting(false)} />
      )}

      {/* Custom Templates Panel */}
      {showCustomTemplates && (
        <CustomTemplatesPanel
          onClose={() => setShowCustomTemplates(false)}
          onSelectTemplate={(template) => {
            // Agregar template a la cola
            addToQueue(template.example, template.spec.parameters?.priority || 0, {
              id: template.id,
              name: template.name,
              description: template.description,
              category: template.category,
              tags: template.tags,
              priority: template.spec.parameters?.priority || 0,
            } as any)
            setShowCustomTemplates(false)
          }}
        />
      )}

      {/* Advanced Statistics Panel */}
      <AnimatePresence>
        {showAdvancedStats && advancedStatsData && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="bg-slate-700/30 rounded-lg p-4 space-y-4 mb-4"
          >
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
                <SparklesIcon className="w-4 h-4 text-purple-400" />
                Estadísticas Avanzadas
              </h4>
              <button
                onClick={() => setShowAdvancedStats(false)}
                className="p-1 hover:bg-slate-600 rounded transition-colors"
              >
                <X className="w-4 h-4 text-slate-400" />
              </button>
            </div>
            
            {/* Overview */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div className="bg-slate-600/30 rounded p-3">
                <div className="text-xs text-slate-400">Mediana Duración</div>
                <div className="text-lg font-bold text-white">
                  {Math.round(advancedStatsData.overview.medianDuration / 1000)}s
                </div>
              </div>
              <div className="bg-slate-600/30 rounded p-3">
                <div className="text-xs text-slate-400">Mejor Tiempo</div>
                <div className="text-lg font-bold text-white">
                  {advancedStatsData.patterns.bestTimeOfDay}
                </div>
              </div>
              <div className="bg-slate-600/30 rounded p-3">
                <div className="text-xs text-slate-400">Mejor Día</div>
                <div className="text-lg font-bold text-white">
                  {advancedStatsData.patterns.bestDayOfWeek}
                </div>
              </div>
              <div className="bg-slate-600/30 rounded p-3">
                <div className="text-xs text-slate-400">Próximo Est.</div>
                <div className="text-lg font-bold text-white">
                  {Math.round(advancedStatsData.predictions.nextBuildEstimate / 1000)}s
                </div>
              </div>
            </div>

            {/* Performance Percentiles */}
            {advancedStatsData.performance.percentile25 > 0 && (
              <div className="bg-slate-600/30 rounded p-3">
                <div className="text-xs text-slate-400 mb-2">Percentiles de Duración</div>
                <div className="grid grid-cols-4 gap-2 text-xs">
                  <div>
                    <span className="text-slate-400">P25:</span>
                    <span className="text-white ml-1">{Math.round(advancedStatsData.performance.percentile25 / 1000)}s</span>
                  </div>
                  <div>
                    <span className="text-slate-400">P50:</span>
                    <span className="text-white ml-1">{Math.round(advancedStatsData.performance.percentile50 / 1000)}s</span>
                  </div>
                  <div>
                    <span className="text-slate-400">P75:</span>
                    <span className="text-white ml-1">{Math.round(advancedStatsData.performance.percentile75 / 1000)}s</span>
                  </div>
                  <div>
                    <span className="text-slate-400">P95:</span>
                    <span className="text-white ml-1">{Math.round(advancedStatsData.performance.percentile95 / 1000)}s</span>
                  </div>
                </div>
              </div>
            )}

            {/* Common Errors */}
            {advancedStatsData.patterns.commonErrors.length > 0 && (
              <div className="bg-slate-600/30 rounded p-3">
                <div className="text-xs text-slate-400 mb-2">Errores Comunes</div>
                <div className="space-y-1">
                  {advancedStatsData.patterns.commonErrors.slice(0, 3).map((error, idx) => (
                    <div key={idx} className="text-xs text-red-400 truncate">
                      {error.count}x {error.error.substring(0, 50)}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Smart History Panel */}
      <AnimatePresence>
        {showSmartHistory && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="bg-slate-700/30 rounded-lg p-4 space-y-3 mb-4"
          >
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
                <SearchIcon className="w-4 h-4 text-purple-400" />
                Historial Inteligente
              </h4>
              <button
                onClick={() => setShowSmartHistory(false)}
                className="p-1 hover:bg-slate-600 rounded transition-colors"
              >
                <X className="w-4 h-4 text-slate-400" />
              </button>
            </div>
            <div className="relative">
              <SearchIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-slate-400" />
              <input
                type="text"
                placeholder="Buscar en historial..."
                value={historySearchQuery}
                onChange={(e) => setHistorySearchQuery(e.target.value)}
                className="w-full bg-slate-600/50 border border-slate-600 rounded-lg px-10 py-2 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-purple-500 text-sm"
              />
            </div>
            <motion.div
              variants={staggerContainer}
              initial="hidden"
              animate="visible"
              className="space-y-2 max-h-60 overflow-y-auto"
            >
              {smartHistory.search({ query: historySearchQuery, limit: 20 }).map((model, index) => (
                <motion.div
                  key={model.modelId}
                  variants={staggerItem}
                  custom={index}
                  className="bg-slate-600/30 rounded p-3"
                >
                  <div className="text-sm font-medium text-white">{model.modelName}</div>
                  <div className="text-xs text-slate-400 truncate">{model.description}</div>
                  <div className="flex items-center gap-2 mt-1 text-xs text-slate-500">
                    <span>{model.status === 'completed' ? '✅' : '❌'}</span>
                    {model.duration && (
                      <span>{Math.round(model.duration / 1000)}s</span>
                    )}
                    <span>{new Date(model.endTime || model.startTime || Date.now()).toLocaleDateString()}</span>
                  </div>
                </motion.div>
              ))}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Quick Commands Panel */}
      <AnimatePresence>
        {showQuickCommands && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="bg-slate-700/30 rounded-lg p-4 space-y-3 mb-4"
          >
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
                <Zap className="w-4 h-4 text-yellow-400" />
                Comandos Rápidos
              </h4>
              <button
                onClick={() => setShowQuickCommands(false)}
                className="p-1 hover:bg-slate-600 rounded transition-colors"
              >
                <X className="w-4 h-4 text-slate-400" />
              </button>
            </div>
            <motion.div
              variants={staggerContainer}
              initial="hidden"
              animate="visible"
              className="grid grid-cols-1 md:grid-cols-2 gap-2"
            >
              {quickCommands.getAllCommands().map((cmd, index) => (
                <motion.button
                  key={cmd.id}
                  variants={staggerItem}
                  custom={index}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => quickCommands.executeCommand(cmd.id)}
                  className="text-left p-3 bg-slate-600/30 hover:bg-slate-600/50 rounded-lg border border-slate-600 transition-all"
                >
                  <div className="flex items-center gap-2 mb-1">
                    {cmd.icon && <span>{cmd.icon}</span>}
                    <span className="text-sm font-medium text-white">{cmd.name}</span>
                    {cmd.shortcut && (
                      <span className="text-xs text-slate-500 ml-auto font-mono">{cmd.shortcut}</span>
                    )}
                  </div>
                  <div className="text-xs text-slate-400">{cmd.description}</div>
                </motion.button>
              ))}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Real-time Metrics Panel */}
      <AnimatePresence>
        {showRealtimeMetrics && realtimeData && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="bg-slate-700/30 rounded-lg p-4 space-y-3 mb-4"
          >
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
                <ActivityIcon className="w-4 h-4 text-green-400" />
                Métricas en Tiempo Real
              </h4>
              <button
                onClick={() => setShowRealtimeMetrics(false)}
                className="p-1 hover:bg-slate-600 rounded transition-colors"
              >
                <X className="w-4 h-4 text-slate-400" />
              </button>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <motion.div
                variants={scaleInBounce}
                initial="hidden"
                animate="visible"
                className="bg-slate-600/30 rounded p-3"
              >
                <div className="text-xs text-slate-400">Builds/min</div>
                <div className="text-lg font-bold text-white">
                  {Math.round(realtimeData.buildsPerMinute)}
                </div>
                <div className="flex items-center gap-1 mt-1">
                  <TrendUp className="w-3 h-3 text-green-400" />
                  <span className="text-xs text-green-400">Live</span>
                </div>
              </motion.div>
              <motion.div
                variants={scaleInBounce}
                initial="hidden"
                animate="visible"
                className="bg-slate-600/30 rounded p-3"
              >
                <div className="text-xs text-slate-400">Tasa de Éxito</div>
                <div className="text-lg font-bold text-white">
                  {Math.round(realtimeData.successRate)}%
                </div>
                <div className="w-full bg-slate-700 rounded-full h-1 mt-2">
                  <motion.div
                    className="bg-green-500 h-1 rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${realtimeData.successRate}%` }}
                    transition={{ duration: 0.5 }}
                  />
                </div>
              </motion.div>
              <motion.div
                variants={scaleInBounce}
                initial="hidden"
                animate="visible"
                className="bg-slate-600/30 rounded p-3"
              >
                <div className="text-xs text-slate-400">Duración Prom.</div>
                <div className="text-lg font-bold text-white">
                  {Math.round(realtimeData.avgDuration / 1000)}s
                </div>
              </motion.div>
              <motion.div
                variants={scaleInBounce}
                initial="hidden"
                animate="visible"
                className="bg-slate-600/30 rounded p-3"
              >
                <div className="text-xs text-slate-400">Cola</div>
                <div className="text-lg font-bold text-white">
                  {realtimeData.queueLength}
                </div>
              </motion.div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Intelligent Alerts Panel */}
      <AnimatePresence>
        {showAlerts && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="bg-slate-700/30 rounded-lg p-4 space-y-3 mb-4"
          >
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-yellow-400" />
                Alertas Inteligentes ({intelligentAlerts.getUnacknowledgedAlerts().length})
              </h4>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => intelligentAlerts.acknowledgeAll()}
                  className="text-xs text-slate-400 hover:text-slate-300"
                >
                  Reconocer todas
                </button>
                <button
                  onClick={() => setShowAlerts(false)}
                  className="p-1 hover:bg-slate-600 rounded transition-colors"
                >
                  <X className="w-4 h-4 text-slate-400" />
                </button>
              </div>
            </div>
            <div className="space-y-2 max-h-60 overflow-y-auto">
              {intelligentAlerts.getAlerts(10).length > 0 ? (
                intelligentAlerts.getAlerts(10).map((alert) => (
                  <motion.div
                    key={alert.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    className={`rounded p-3 border ${
                      alert.severity === 'critical'
                        ? 'bg-red-900/20 border-red-700'
                        : alert.severity === 'error'
                        ? 'bg-red-800/20 border-red-600'
                        : alert.severity === 'warning'
                        ? 'bg-yellow-900/20 border-yellow-700'
                        : 'bg-blue-900/20 border-blue-700'
                    }`}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <span
                            className={`text-xs px-2 py-0.5 rounded ${
                              alert.severity === 'critical' || alert.severity === 'error'
                                ? 'bg-red-600/20 text-red-400'
                                : alert.severity === 'warning'
                                ? 'bg-yellow-600/20 text-yellow-400'
                                : 'bg-blue-600/20 text-blue-400'
                            }`}
                          >
                            {alert.severity}
                          </span>
                          {!alert.acknowledged && (
                            <span className="text-xs text-slate-500">Nuevo</span>
                          )}
                        </div>
                        <p className="text-sm text-white">{alert.message}</p>
                        <p className="text-xs text-slate-400 mt-1">
                          {new Date(alert.timestamp).toLocaleTimeString()}
                        </p>
                      </div>
                      {!alert.acknowledged && (
                        <button
                          onClick={() => intelligentAlerts.acknowledge(alert.id)}
                          className="p-1 hover:bg-slate-600 rounded transition-colors"
                        >
                          <Check className="w-4 h-4 text-slate-400" />
                        </button>
                      )}
                    </div>
                  </motion.div>
                ))
              ) : (
                <div className="text-center text-slate-400 py-4 text-sm">
                  No hay alertas activas
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Contextual Help Panel */}
      <AnimatePresence>
        {showHelp && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="bg-slate-700/30 rounded-lg p-4 space-y-3 mb-4"
          >
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
                <HelpCircle className="w-4 h-4 text-blue-400" />
                Ayuda Contextual
              </h4>
              <button
                onClick={() => setShowHelp(false)}
                className="p-1 hover:bg-slate-600 rounded transition-colors"
              >
                <X className="w-4 h-4 text-slate-400" />
              </button>
            </div>
            <div className="space-y-3">
              {helpContext ? (
                contextualHelp.searchByContext({ component: helpContext }).map((topic) => (
                  <motion.div
                    key={topic.id}
                    variants={cardHover}
                    initial="rest"
                    whileHover="hover"
                    className="bg-slate-600/30 rounded p-3 border border-slate-600"
                  >
                    <h5 className="text-sm font-medium text-white mb-1">{topic.title}</h5>
                    <p className="text-xs text-slate-400 mb-2">{topic.content}</p>
                    {topic.examples && topic.examples.length > 0 && (
                      <div className="mt-2">
                        <div className="text-xs text-slate-500 mb-1">Ejemplos:</div>
                        <ul className="list-disc list-inside text-xs text-slate-400 space-y-1">
                          {topic.examples.map((example, idx) => (
                            <li key={idx}>{example}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </motion.div>
                ))
              ) : (
                contextualHelp.getMostViewedTopics(5).map((topic) => (
                  <motion.div
                    key={topic.id}
                    variants={cardHover}
                    initial="rest"
                    whileHover="hover"
                    className="bg-slate-600/30 rounded p-3 border border-slate-600 cursor-pointer"
                    onClick={() => {
                      contextualHelp.recordView(topic.id)
                      setHelpContext(topic.category)
                    }}
                  >
                    <h5 className="text-sm font-medium text-white">{topic.title}</h5>
                    <p className="text-xs text-slate-400 mt-1">{topic.content}</p>
                  </motion.div>
                ))
              )}
            </div>
            {enhancedShortcuts.getAllShortcuts().length > 0 && (
              <div className="mt-4 pt-4 border-t border-slate-600">
                <div className="text-xs text-slate-400 mb-2 flex items-center gap-2">
                  <Keyboard className="w-3 h-3" />
                  Atajos de Teclado
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs">
                  {enhancedShortcuts.getAllShortcuts().slice(0, 6).map((shortcut) => (
                    <div key={shortcut.id} className="flex items-center justify-between">
                      <span className="text-slate-300">{shortcut.description}</span>
                      <span className="text-slate-500 font-mono">
                        {enhancedShortcuts.formatKeys(shortcut.keys)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Favorites Panel */}
      <AnimatePresence>
        {showFavorites && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="bg-slate-700/30 rounded-lg p-4 space-y-3 mb-4"
          >
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
                <Heart className="w-4 h-4 text-red-400 fill-current" />
                Favoritos ({favoritesManager.getAllFavorites().length})
              </h4>
              <button
                onClick={() => setShowFavorites(false)}
                className="p-1 hover:bg-slate-600 rounded transition-colors"
              >
                <X className="w-4 h-4 text-slate-400" />
              </button>
            </div>
            <div className="space-y-2 max-h-60 overflow-y-auto">
              {favoritesManager.getAllFavorites().length > 0 ? (
                favoritesManager.getAllFavorites().map((fav) => (
                  <div key={fav.modelId} className="bg-slate-600/30 rounded p-3">
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <div className="text-sm font-medium text-white">{fav.modelName}</div>
                        <div className="text-xs text-slate-400 truncate">{fav.description}</div>
                        {fav.notes && (
                          <div className="text-xs text-slate-500 mt-1">{fav.notes}</div>
                        )}
                      </div>
                      <button
                        onClick={() => {
                          favoritesManager.removeFavorite(fav.modelId)
                          toast('Eliminado de favoritos', { icon: '💔' })
                        }}
                        className="p-1 hover:bg-red-600/20 rounded transition-colors"
                      >
                        <Heart className="w-4 h-4 text-red-400 fill-current" />
                      </button>
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-center text-slate-400 py-4 text-sm">
                  No hay favoritos aún
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Panel de Versionado */}
      <AnimatePresence>
        {showVersioning && completedBuilds.length > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="bg-slate-700/30 rounded-lg p-4 space-y-3"
          >
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-semibold text-slate-300">Versionado de Modelos</h4>
              <button
                onClick={() => setShowVersioning(false)}
                className="p-1 hover:bg-slate-600 rounded transition-colors"
              >
                <X className="w-4 h-4 text-slate-400" />
              </button>
            </div>
            <div className="space-y-2 max-h-60 overflow-y-auto">
              {Array.from(new Set(completedBuilds.map(m => m.modelName))).map(modelName => {
                const versions = modelVersioning.getVersions(modelName)
                if (versions.length === 0) return null
                
                return (
                  <div key={modelName} className="bg-slate-600/30 rounded p-3">
                    <div className="text-sm font-medium text-white mb-2">{modelName}</div>
                    <div className="space-y-1">
                      {versions.map(version => (
                        <div key={version.version} className="flex items-center justify-between text-xs">
                          <span className="text-purple-400">{version.version}</span>
                          <span className="text-slate-400">
                            {new Date(version.createdAt).toLocaleDateString()}
                          </span>
                          {version.performance && (
                            <span className="text-slate-500">
                              {Math.round(version.performance.duration / 1000)}s
                            </span>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Panel de Logs */}
      <AnimatePresence>
        {showLogs && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="bg-slate-700/30 rounded-lg p-4 space-y-3 max-h-96 overflow-hidden flex flex-col"
          >
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-semibold text-slate-300">Logs del Sistema</h4>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => {
                    const logsText = logger.exportLogs('text')
                    const blob = new Blob([logsText], { type: 'text/plain' })
                    const url = URL.createObjectURL(blob)
                    const a = document.createElement('a')
                    a.href = url
                    a.download = `logs-${Date.now()}.txt`
                    a.click()
                    URL.revokeObjectURL(url)
                  }}
                  className="p-1 hover:bg-slate-600 rounded transition-colors"
                  title="Exportar logs"
                >
                  <Download className="w-3 h-3 text-slate-400" />
                </button>
                <button
                  onClick={() => logger.clear()}
                  className="p-1 hover:bg-slate-600 rounded transition-colors"
                  title="Limpiar logs"
                >
                  <Trash2 className="w-3 h-3 text-slate-400" />
                </button>
                <button
                  onClick={() => setShowLogs(false)}
                  className="p-1 hover:bg-slate-600 rounded transition-colors"
                >
                  <X className="w-4 h-4 text-slate-400" />
                </button>
              </div>
            </div>
            <div className="flex-1 overflow-y-auto bg-slate-900/50 rounded p-3 font-mono text-xs space-y-1">
              {logger.getLogs().slice(-50).map((log, index) => {
                const date = new Date(log.timestamp).toLocaleTimeString()
                const levelColor = {
                  DEBUG: 'text-slate-500',
                  INFO: 'text-blue-400',
                  WARN: 'text-yellow-400',
                  ERROR: 'text-red-400',
                }[log.level] || 'text-slate-300'
                
                return (
                  <div key={index} className={`${levelColor} flex items-start gap-2`}>
                    <span className="text-slate-500">[{date}]</span>
                    <span className="font-semibold">[{log.level}]</span>
                    <span>{log.message}</span>
                    {log.context && (
                      <span className="text-slate-500">
                        {JSON.stringify(log.context)}
                      </span>
                    )}
                  </div>
                )
              })}
              {logger.getLogs().length === 0 && (
                <div className="text-slate-500 text-center py-4">No hay logs disponibles</div>
              )}
            </div>
            <div className="text-xs text-slate-400">
              Total: {logger.getLogs().length} logs | 
              Errores: {logger.getStats().errors} | 
              Warnings: {logger.getStats().warnings}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Botón de ayuda flotante */}
      <button
        onClick={() => setShowHelp(!showHelp)}
        className="fixed bottom-6 right-6 p-3 bg-purple-600 hover:bg-purple-700 rounded-full shadow-lg transition-colors z-40"
        title="Ayuda y atajos de teclado (Ctrl+?)"
        aria-label="Mostrar ayuda"
      >
        <span className="text-white text-lg font-bold">?</span>
      </button>
    </div>
  )
}

