'use client'

import { useMemo } from 'react'
import { motion } from 'framer-motion'
import { 
  TrendingUp, Clock, CheckCircle, XCircle, Zap, Activity, 
  BarChart3, Target, Gauge, AlertTriangle
} from 'lucide-react'
import { ProactiveBuildResult } from './ProactiveModelBuilder'
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

interface AdvancedDashboardProps {
  models: ProactiveBuildResult[]
  queueLength: number
  isActive: boolean
  stats: {
    total: number
    successful: number
    failed: number
    avgDuration: number
    successRate: number
  }
}

export default function AdvancedDashboard({
  models,
  queueLength,
  isActive,
  stats,
}: AdvancedDashboardProps) {
  // Datos para gráfico de tendencias
  const trendData = useMemo(() => {
    const last20 = models.slice(-20).reverse()
    return last20.map((model, index) => ({
      name: `#${models.length - index}`,
      duration: model.duration ? Math.round(model.duration / 1000) : 0,
      success: model.status === 'completed' ? 1 : 0,
      cumulative: models.length - index,
    }))
  }, [models])

  // Distribución de tiempo
  const timeDistribution = useMemo(() => {
    const durations = models
      .filter(m => m.duration)
      .map(m => Math.round(m.duration! / 1000))
    
    const ranges = {
      fast: durations.filter(d => d < 30).length,
      medium: durations.filter(d => d >= 30 && d < 60).length,
      slow: durations.filter(d => d >= 60).length,
    }

    return ranges
  }, [models])

  // Tasa de éxito por hora (simulado)
  const hourlySuccess = useMemo(() => {
    const hourly: Record<number, { total: number; success: number }> = {}
    
    models.forEach(model => {
      if (model.startTime) {
        const hour = new Date(model.startTime).getHours()
        if (!hourly[hour]) {
          hourly[hour] = { total: 0, success: 0 }
        }
        hourly[hour].total++
        if (model.status === 'completed') {
          hourly[hour].success++
        }
      }
    })

    return Object.entries(hourly).map(([hour, data]) => ({
      hour: `${hour}:00`,
      successRate: (data.success / data.total) * 100,
      total: data.total,
    }))
  }, [models])

  // Predicción de tiempo para cola
  const estimatedTimeForQueue = useMemo(() => {
    if (stats.avgDuration === 0 || queueLength === 0) return 0
    return Math.round((stats.avgDuration * queueLength) / 1000) // en segundos
  }, [stats.avgDuration, queueLength])

  return (
    <div className="space-y-4">
      {/* Métricas Principales */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-gradient-to-br from-purple-600/20 to-pink-600/20 rounded-lg p-4 border border-purple-600/30"
        >
          <div className="flex items-center justify-between mb-2">
            <Zap className="w-5 h-5 text-purple-400" />
            <span className="text-xs text-slate-400">Activo</span>
          </div>
          <div className="text-2xl font-bold text-white">
            {isActive ? 'Sí' : 'No'}
          </div>
          <div className="text-xs text-slate-400 mt-1">
            {queueLength} en cola
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-gradient-to-br from-green-600/20 to-emerald-600/20 rounded-lg p-4 border border-green-600/30"
        >
          <div className="flex items-center justify-between mb-2">
            <CheckCircle className="w-5 h-5 text-green-400" />
            <span className="text-xs text-slate-400">Éxito</span>
          </div>
          <div className="text-2xl font-bold text-white">
            {stats.successful}
          </div>
          <div className="text-xs text-slate-400 mt-1">
            {stats.successRate.toFixed(1)}% tasa
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-gradient-to-br from-blue-600/20 to-cyan-600/20 rounded-lg p-4 border border-blue-600/30"
        >
          <div className="flex items-center justify-between mb-2">
            <Clock className="w-5 h-5 text-blue-400" />
            <span className="text-xs text-slate-400">Promedio</span>
          </div>
          <div className="text-2xl font-bold text-white">
            {Math.round(stats.avgDuration / 1000)}s
          </div>
          <div className="text-xs text-slate-400 mt-1">
            Por modelo
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-gradient-to-br from-orange-600/20 to-red-600/20 rounded-lg p-4 border border-orange-600/30"
        >
          <div className="flex items-center justify-between mb-2">
            <Target className="w-5 h-5 text-orange-400" />
            <span className="text-xs text-slate-400">Estimado</span>
          </div>
          <div className="text-2xl font-bold text-white">
            {estimatedTimeForQueue > 60 
              ? `${Math.round(estimatedTimeForQueue / 60)}m`
              : `${estimatedTimeForQueue}s`
            }
          </div>
          <div className="text-xs text-slate-400 mt-1">
            Para cola completa
          </div>
        </motion.div>
      </div>

      {/* Gráfico de Tendencias */}
      {trendData.length > 0 && (
        <div className="bg-slate-700/30 rounded-lg p-4">
          <h4 className="text-sm font-semibold text-slate-300 mb-4 flex items-center gap-2">
            <TrendingUp className="w-4 h-4" />
            Tendencias de Construcción
          </h4>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={trendData}>
              <defs>
                <linearGradient id="colorDuration" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
              <XAxis dataKey="name" stroke="#94a3b8" fontSize={10} />
              <YAxis stroke="#94a3b8" fontSize={10} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: '8px' }}
              />
              <Area
                type="monotone"
                dataKey="duration"
                stroke="#8b5cf6"
                fillOpacity={1}
                fill="url(#colorDuration)"
                name="Duración (s)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Distribución de Tiempo */}
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-slate-700/30 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <Gauge className="w-4 h-4 text-green-400" />
            <span className="text-xs text-slate-400">Rápido (&lt;30s)</span>
          </div>
          <div className="text-xl font-bold text-green-400">
            {timeDistribution.fast}
          </div>
        </div>
        <div className="bg-slate-700/30 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <Activity className="w-4 h-4 text-yellow-400" />
            <span className="text-xs text-slate-400">Medio (30-60s)</span>
          </div>
          <div className="text-xl font-bold text-yellow-400">
            {timeDistribution.medium}
          </div>
        </div>
        <div className="bg-slate-700/30 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <AlertTriangle className="w-4 h-4 text-red-400" />
            <span className="text-xs text-slate-400">Lento (&gt;60s)</span>
          </div>
          <div className="text-xl font-bold text-red-400">
            {timeDistribution.slow}
          </div>
        </div>
      </div>

      {/* Tasa de Éxito por Hora */}
      {hourlySuccess.length > 0 && (
        <div className="bg-slate-700/30 rounded-lg p-4">
          <h4 className="text-sm font-semibold text-slate-300 mb-4">
            Tasa de Éxito por Hora
          </h4>
          <ResponsiveContainer width="100%" height={150}>
            <BarChart data={hourlySuccess}>
              <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
              <XAxis dataKey="hour" stroke="#94a3b8" fontSize={10} />
              <YAxis stroke="#94a3b8" fontSize={10} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: '8px' }}
              />
              <Bar dataKey="successRate" fill="#10b981" name="Tasa de Éxito (%)" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}










