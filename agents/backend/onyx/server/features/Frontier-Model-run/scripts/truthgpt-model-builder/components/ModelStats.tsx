'use client'

import { motion } from 'framer-motion'
import { TrendingUp, CheckCircle, Clock, AlertCircle, Zap } from 'lucide-react'
import { Model } from '@/store/modelStore'

interface ModelStatsProps {
  models: Model[]
}

export default function ModelStats({ models }: ModelStatsProps) {
  const stats = {
    total: models.length,
    completed: models.filter(m => m.status === 'completed').length,
    creating: models.filter(m => m.status === 'creating').length,
    failed: models.filter(m => m.status === 'failed').length,
    successRate: models.length > 0 
      ? (models.filter(m => m.status === 'completed').length / models.length) * 100 
      : 0,
  }

  const typeDistribution = models.reduce((acc, model) => {
    const type = model.spec?.type || 'custom'
    acc[type] = (acc[type] || 0) + 1
    return acc
  }, {} as Record<string, number>)

  if (models.length === 0) {
    return null
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6"
    >
      <div className="bg-slate-800/50 rounded-lg border border-slate-700 p-4">
        <div className="flex items-center gap-2 mb-2">
          <Zap className="w-5 h-5 text-purple-400" />
          <p className="text-xs text-slate-400">Total</p>
        </div>
        <p className="text-2xl font-bold text-white">{stats.total}</p>
      </div>

      <div className="bg-slate-800/50 rounded-lg border border-slate-700 p-4">
        <div className="flex items-center gap-2 mb-2">
          <CheckCircle className="w-5 h-5 text-green-400" />
          <p className="text-xs text-slate-400">Completados</p>
        </div>
        <p className="text-2xl font-bold text-white">{stats.completed}</p>
      </div>

      <div className="bg-slate-800/50 rounded-lg border border-slate-700 p-4">
        <div className="flex items-center gap-2 mb-2">
          <Clock className="w-5 h-5 text-blue-400" />
          <p className="text-xs text-slate-400">En creación</p>
        </div>
        <p className="text-2xl font-bold text-white">{stats.creating}</p>
      </div>

      <div className="bg-slate-800/50 rounded-lg border border-slate-700 p-4">
        <div className="flex items-center gap-2 mb-2">
          <AlertCircle className="w-5 h-5 text-red-400" />
          <p className="text-xs text-slate-400">Errores</p>
        </div>
        <p className="text-2xl font-bold text-white">{stats.failed}</p>
      </div>

      <div className="bg-slate-800/50 rounded-lg border border-slate-700 p-4">
        <div className="flex items-center gap-2 mb-2">
          <TrendingUp className="w-5 h-5 text-yellow-400" />
          <p className="text-xs text-slate-400">Tasa éxito</p>
        </div>
        <p className="text-2xl font-bold text-white">{stats.successRate.toFixed(0)}%</p>
      </div>
    </motion.div>
  )
}


