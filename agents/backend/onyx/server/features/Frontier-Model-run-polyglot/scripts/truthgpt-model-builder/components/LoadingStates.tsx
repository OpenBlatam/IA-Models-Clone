/**
 * Componentes de Estados de Carga Mejorados
 * ==========================================
 * 
 * Componentes visuales avanzados para mostrar estados de carga y progreso
 */

'use client'

import { motion, AnimatePresence } from 'framer-motion'
import { 
  Loader2, 
  CheckCircle, 
  XCircle, 
  Clock, 
  Activity,
  AlertCircle,
  Zap,
  TrendingUp,
  BarChart3
} from 'lucide-react'
import { ModelStatus, ModelProgress, ModelCreationStep } from '@/lib/types/modelTypes'

// ============================================================================
// LOADING SPINNER AVANZADO
// ============================================================================

export interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg' | 'xl'
  variant?: 'default' | 'pulse' | 'dots' | 'bars'
  color?: 'primary' | 'secondary' | 'success' | 'warning' | 'error'
  label?: string
  className?: string
}

export function LoadingSpinner({
  size = 'md',
  variant = 'default',
  color = 'primary',
  label,
  className = ''
}: LoadingSpinnerProps) {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8',
    xl: 'w-12 h-12'
  }

  const colorClasses = {
    primary: 'text-blue-600 dark:text-blue-400',
    secondary: 'text-gray-600 dark:text-gray-400',
    success: 'text-green-600 dark:text-green-400',
    warning: 'text-yellow-600 dark:text-yellow-400',
    error: 'text-red-600 dark:text-red-400'
  }

  if (variant === 'pulse') {
    return (
      <div className={`flex flex-col items-center gap-2 ${className}`}>
        <motion.div
          className={`${sizeClasses[size]} ${colorClasses[color]} rounded-full bg-current`}
          animate={{
            scale: [1, 1.2, 1],
            opacity: [0.5, 1, 0.5]
          }}
          transition={{
            duration: 1.5,
            repeat: Infinity,
            ease: 'easeInOut'
          }}
        />
        {label && (
          <span className="text-xs text-gray-600 dark:text-gray-400">{label}</span>
        )}
      </div>
    )
  }

  if (variant === 'dots') {
    return (
      <div className={`flex items-center gap-1 ${className}`}>
        {[0, 1, 2].map((i) => (
          <motion.div
            key={i}
            className={`${sizeClasses.sm} ${colorClasses[color]} rounded-full bg-current`}
            animate={{
              y: [0, -8, 0]
            }}
            transition={{
              duration: 0.6,
              repeat: Infinity,
              delay: i * 0.2,
              ease: 'easeInOut'
            }}
          />
        ))}
        {label && (
          <span className="ml-2 text-xs text-gray-600 dark:text-gray-400">{label}</span>
        )}
      </div>
    )
  }

  if (variant === 'bars') {
    return (
      <div className={`flex items-end gap-1 ${className}`}>
        {[0, 1, 2, 3].map((i) => (
          <motion.div
            key={i}
            className={`w-1 ${colorClasses[color]} bg-current`}
            animate={{
              height: ['8px', '20px', '8px']
            }}
            transition={{
              duration: 0.8,
              repeat: Infinity,
              delay: i * 0.15,
              ease: 'easeInOut'
            }}
          />
        ))}
        {label && (
          <span className="ml-2 text-xs text-gray-600 dark:text-gray-400">{label}</span>
        )}
      </div>
    )
  }

  return (
    <div className={`flex flex-col items-center gap-2 ${className}`}>
      <Loader2 
        className={`${sizeClasses[size]} ${colorClasses[color]} animate-spin`}
      />
      {label && (
        <span className="text-xs text-gray-600 dark:text-gray-400">{label}</span>
      )}
    </div>
  )
}

// ============================================================================
// PROGRESS BAR AVANZADO
// ============================================================================

export interface ProgressBarProps {
  progress: number // 0-100
  label?: string
  showPercentage?: boolean
  showAnimation?: boolean
  color?: 'primary' | 'success' | 'warning' | 'error'
  size?: 'sm' | 'md' | 'lg'
  estimatedTime?: number // segundos
  className?: string
}

export function ProgressBar({
  progress,
  label,
  showPercentage = true,
  showAnimation = true,
  color = 'primary',
  size = 'md',
  estimatedTime,
  className = ''
}: ProgressBarProps) {
  const clampedProgress = Math.max(0, Math.min(100, progress))

  const colorClasses = {
    primary: 'bg-blue-600 dark:bg-blue-500',
    success: 'bg-green-600 dark:bg-green-500',
    warning: 'bg-yellow-600 dark:bg-yellow-500',
    error: 'bg-red-600 dark:bg-red-500'
  }

  const sizeClasses = {
    sm: 'h-1',
    md: 'h-2',
    lg: 'h-3'
  }

  const formatTime = (seconds: number): string => {
    if (seconds < 60) return `${Math.round(seconds)}s`
    const mins = Math.floor(seconds / 60)
    const secs = Math.round(seconds % 60)
    return `${mins}m ${secs}s`
  }

  return (
    <div className={`w-full ${className}`}>
      {(label || showPercentage) && (
        <div className="flex justify-between items-center mb-1">
          {label && (
            <span className="text-sm text-gray-700 dark:text-gray-300">{label}</span>
          )}
          <div className="flex items-center gap-2">
            {showPercentage && (
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                {clampedProgress.toFixed(0)}%
              </span>
            )}
            {estimatedTime && estimatedTime > 0 && (
              <span className="text-xs text-gray-500 dark:text-gray-400 flex items-center gap-1">
                <Clock className="w-3 h-3" />
                {formatTime(estimatedTime)}
              </span>
            )}
          </div>
        </div>
      )}
      <div className={`w-full ${sizeClasses[size]} bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden`}>
        <motion.div
          className={`${colorClasses[color]} h-full rounded-full`}
          initial={showAnimation ? { width: 0 } : { width: `${clampedProgress}%` }}
          animate={{ width: `${clampedProgress}%` }}
          transition={{
            duration: showAnimation ? 0.5 : 0,
            ease: 'easeOut'
          }}
        />
      </div>
    </div>
  )
}

// ============================================================================
// STATUS BADGE
// ============================================================================

export interface StatusBadgeProps {
  status: ModelStatus
  size?: 'sm' | 'md' | 'lg'
  showIcon?: boolean
  className?: string
}

const statusConfig: Record<ModelStatus, { 
  label: string
  color: string
  icon: typeof Loader2
}> = {
  idle: { label: 'Inactivo', color: 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200', icon: Clock },
  validating: { label: 'Validando', color: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200', icon: Activity },
  creating: { label: 'Creando', color: 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200', icon: Zap },
  compiling: { label: 'Compilando', color: 'bg-indigo-100 text-indigo-800 dark:bg-indigo-900 dark:text-indigo-200', icon: Activity },
  training: { label: 'Entrenando', color: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200', icon: TrendingUp },
  evaluating: { label: 'Evaluando', color: 'bg-cyan-100 text-cyan-800 dark:bg-cyan-900 dark:text-cyan-200', icon: BarChart3 },
  predicting: { label: 'Prediciendo', color: 'bg-teal-100 text-teal-800 dark:bg-teal-900 dark:text-teal-200', icon: Activity },
  completed: { label: 'Completado', color: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200', icon: CheckCircle },
  failed: { label: 'Fallido', color: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200', icon: XCircle },
  cancelled: { label: 'Cancelado', color: 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200', icon: XCircle },
  paused: { label: 'Pausado', color: 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200', icon: Clock }
}

export function StatusBadge({
  status,
  size = 'md',
  showIcon = true,
  className = ''
}: StatusBadgeProps) {
  const config = statusConfig[status]
  const Icon = config.icon

  const sizeClasses = {
    sm: 'text-xs px-2 py-0.5',
    md: 'text-sm px-2.5 py-1',
    lg: 'text-base px-3 py-1.5'
  }

  const iconSizes = {
    sm: 'w-3 h-3',
    md: 'w-4 h-4',
    lg: 'w-5 h-5'
  }

  const isAnimated = status === 'creating' || status === 'compiling' || 
                     status === 'training' || status === 'validating' ||
                     status === 'evaluating' || status === 'predicting'

  return (
    <motion.span
      className={`inline-flex items-center gap-1.5 rounded-full font-medium ${config.color} ${sizeClasses[size]} ${className}`}
      animate={isAnimated ? {
        scale: [1, 1.05, 1]
      } : {}}
      transition={{
        duration: 2,
        repeat: isAnimated ? Infinity : 0,
        ease: 'easeInOut'
      }}
    >
      {showIcon && (
        <Icon 
          className={`${iconSizes[size]} ${isAnimated && Icon === Loader2 ? 'animate-spin' : ''}`}
        />
      )}
      {config.label}
    </motion.span>
  )
}

// ============================================================================
// MODEL PROGRESS CARD
// ============================================================================

export interface ModelProgressCardProps {
  progress: ModelProgress
  onCancel?: () => void
  className?: string
}

export function ModelProgressCard({
  progress,
  onCancel,
  className = ''
}: ModelProgressCardProps) {
  const stepLabels: Record<ModelCreationStep, string> = {
    validation: 'Validando descripción',
    layer_parsing: 'Analizando capas',
    model_creation: 'Creando modelo',
    compilation: 'Compilando modelo',
    initialization: 'Inicializando',
    ready: 'Listo'
  }

  const formatTime = (seconds: number): string => {
    if (seconds < 60) return `${Math.round(seconds)}s`
    const mins = Math.floor(seconds / 60)
    const secs = Math.round(seconds % 60)
    return `${mins}m ${secs}s`
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className={`bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 p-4 ${className}`}
    >
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <StatusBadge status="creating" size="sm" />
            <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
              {stepLabels[progress.step]}
            </span>
          </div>
          {onCancel && (
            <button
              onClick={onCancel}
              className="text-xs text-red-600 dark:text-red-400 hover:text-red-700 dark:hover:text-red-300"
            >
              Cancelar
            </button>
          )}
        </div>

        <ProgressBar
          progress={progress.progress}
          showPercentage
          estimatedTime={progress.estimatedTimeRemaining}
          color="primary"
          size="md"
        />

        {progress.message && (
          <p className="text-xs text-gray-600 dark:text-gray-400">
            {progress.message}
          </p>
        )}

        <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
          <span>Iniciado: {progress.startedAt.toLocaleTimeString()}</span>
          {progress.estimatedTimeRemaining && progress.estimatedTimeRemaining > 0 && (
            <span className="flex items-center gap-1">
              <Clock className="w-3 h-3" />
              {formatTime(progress.estimatedTimeRemaining)} restantes
            </span>
          )}
        </div>
      </div>
    </motion.div>
  )
}

// ============================================================================
// LOADING OVERLAY
// ============================================================================

export interface LoadingOverlayProps {
  isVisible: boolean
  message?: string
  progress?: number
  onCancel?: () => void
  className?: string
}

export function LoadingOverlay({
  isVisible,
  message = 'Cargando...',
  progress,
  onCancel,
  className = ''
}: LoadingOverlayProps) {
  return (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className={`fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm ${className}`}
        >
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.9, opacity: 0 }}
            className="bg-white dark:bg-gray-800 rounded-lg shadow-xl p-6 max-w-sm w-full mx-4"
          >
            <div className="space-y-4">
              <div className="flex justify-center">
                <LoadingSpinner size="lg" variant="pulse" />
              </div>
              
              <div className="text-center">
                <p className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-2">
                  {message}
                </p>
                {progress !== undefined && (
                  <ProgressBar
                    progress={progress}
                    showPercentage
                    size="sm"
                    className="mt-2"
                  />
                )}
              </div>

              {onCancel && (
                <div className="flex justify-center">
                  <button
                    onClick={onCancel}
                    className="text-xs text-red-600 dark:text-red-400 hover:text-red-700 dark:hover:text-red-300"
                  >
                    Cancelar
                  </button>
                </div>
              )}
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

// ============================================================================
// SKELETON LOADER
// ============================================================================

export interface SkeletonLoaderProps {
  variant?: 'text' | 'circular' | 'rectangular' | 'card'
  width?: string | number
  height?: string | number
  className?: string
  count?: number
}

export function SkeletonLoader({
  variant = 'text',
  width,
  height,
  className = '',
  count = 1
}: SkeletonLoaderProps) {
  const baseClasses = 'bg-gray-200 dark:bg-gray-700 rounded animate-pulse'

  const variantClasses = {
    text: 'h-4',
    circular: 'rounded-full',
    rectangular: '',
    card: 'h-32'
  }

  const style: React.CSSProperties = {}
  if (width) style.width = typeof width === 'number' ? `${width}px` : width
  if (height) style.height = typeof height === 'number' ? `${height}px` : height

  return (
    <>
      {Array.from({ length: count }).map((_, i) => (
        <motion.div
          key={i}
          className={`${baseClasses} ${variantClasses[variant]} ${className}`}
          style={style}
          animate={{
            opacity: [0.5, 1, 0.5]
          }}
          transition={{
            duration: 1.5,
            repeat: Infinity,
            delay: i * 0.1,
            ease: 'easeInOut'
          }}
        />
      ))}
    </>
  )
}







