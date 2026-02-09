'use client'

import { motion } from 'framer-motion'
import { TrendingUp, Clock, HardDrive, Target, CheckCircle, AlertTriangle } from 'lucide-react'

interface OptimizationResult {
  improvements: string[]
  performance: {
    estimatedTrainingTime: number
    estimatedMemoryUsage: number
    estimatedAccuracy: number
  }
}

interface OptimizationReportProps {
  optimization: OptimizationResult
}

export default function OptimizationReport({ optimization }: OptimizationReportProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-slate-800/50 rounded-lg border border-slate-700 p-6 space-y-4"
    >
      <div className="flex items-center gap-2 mb-4">
        <TrendingUp className="w-6 h-6 text-green-400" />
        <h3 className="text-xl font-bold text-white">Optimización del Modelo</h3>
      </div>

      {/* Performance Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="bg-slate-700/50 rounded-lg p-4 border border-slate-600">
          <div className="flex items-center gap-2 mb-2">
            <Clock className="w-5 h-5 text-blue-400" />
            <span className="text-sm text-slate-400">Tiempo Estimado</span>
          </div>
          <p className="text-2xl font-bold text-white">
            {optimization.performance.estimatedTrainingTime} min
          </p>
        </div>

        <div className="bg-slate-700/50 rounded-lg p-4 border border-slate-600">
          <div className="flex items-center gap-2 mb-2">
            <HardDrive className="w-5 h-5 text-purple-400" />
            <span className="text-sm text-slate-400">Memoria Estimada</span>
          </div>
          <p className="text-2xl font-bold text-white">
            {optimization.performance.estimatedMemoryUsage} MB
          </p>
        </div>

        <div className="bg-slate-700/50 rounded-lg p-4 border border-slate-600">
          <div className="flex items-center gap-2 mb-2">
            <Target className="w-5 h-5 text-green-400" />
            <span className="text-sm text-slate-400">Precisión Estimada</span>
          </div>
          <p className="text-2xl font-bold text-white">
            {optimization.performance.estimatedAccuracy}%
          </p>
        </div>
      </div>

      {/* Improvements */}
      {optimization.improvements.length > 0 && (
        <div>
          <h4 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
            <CheckCircle className="w-4 h-4 text-green-400" />
            Mejoras Aplicadas
          </h4>
          <ul className="space-y-2">
            {optimization.improvements.map((improvement, index) => (
              <motion.li
                key={index}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className="flex items-start gap-2 text-sm text-slate-300"
              >
                <span className="text-green-400 mt-1">✓</span>
                <span>{improvement}</span>
              </motion.li>
            ))}
          </ul>
        </div>
      )}

      {optimization.improvements.length === 0 && (
        <div className="flex items-center gap-2 p-3 bg-blue-500/10 border border-blue-500/30 rounded-lg">
          <AlertTriangle className="w-5 h-5 text-blue-400" />
          <p className="text-sm text-slate-300">
            El modelo ya está optimizado. No se necesitaron cambios adicionales.
          </p>
        </div>
      )}
    </motion.div>
  )
}


