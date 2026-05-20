'use client'

import { motion } from 'framer-motion'
import { Github, CheckCircle, AlertCircle, Loader2, Clock } from 'lucide-react'
import { format } from 'date-fns'
import ProgressIndicator from './ProgressIndicator'

interface Model {
  id: string
  name: string
  description: string
  status: 'creating' | 'completed' | 'failed'
  githubUrl: string | null
  createdAt: Date
  progress?: number
  currentStep?: string
  spec?: {
    type: string
    architecture: string
  }
}

interface ModelStatusProps {
  model: Model
}

export default function ModelStatus({ model }: ModelStatusProps) {
  const getStatusIcon = () => {
    switch (model.status) {
      case 'creating':
        return <Loader2 className="w-5 h-5 animate-spin text-blue-400" />
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-400" />
      case 'failed':
        return <AlertCircle className="w-5 h-5 text-red-400" />
      default:
        return <Clock className="w-5 h-5 text-slate-400" />
    }
  }

  const getStatusText = () => {
    switch (model.status) {
      case 'creating':
        return 'Creando modelo...'
      case 'completed':
        return 'Modelo completado'
      case 'failed':
        return 'Error al crear modelo'
      default:
        return 'Pendiente'
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      className="border-t border-slate-700 bg-slate-800/30 p-4 space-y-3"
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3 flex-1">
          {getStatusIcon()}
          <div className="flex-1">
            <p className="text-sm font-medium text-white">{model.name}</p>
            <p className="text-xs text-slate-400">
              {getStatusText()} • {format(model.createdAt, 'dd MMM yyyy, HH:mm')}
            </p>
            {model.spec && (
              <p className="text-xs text-slate-500 mt-1">
                {model.spec.type} • {model.spec.architecture}
              </p>
            )}
          </div>
        </div>
        {model.githubUrl && (
          <a
            href={model.githubUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 bg-slate-700 hover:bg-slate-600 text-white px-4 py-2 rounded-lg transition-colors text-sm"
          >
            <Github className="w-4 h-4" />
            Ver en GitHub
          </a>
        )}
      </div>
      
      {model.status === 'creating' && model.progress !== undefined && model.currentStep && (
        <ProgressIndicator
          progress={model.progress}
          currentStep={model.currentStep}
          status={model.status}
        />
      )}
    </motion.div>
  )
}

