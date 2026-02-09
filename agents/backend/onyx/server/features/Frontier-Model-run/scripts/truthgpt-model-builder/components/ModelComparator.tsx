'use client'

import { useState, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { X, BarChart3, TrendingUp, Clock, CheckCircle, XCircle, Download } from 'lucide-react'
import { ProactiveBuildResult } from './ProactiveModelBuilder'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts'

interface ModelComparatorProps {
  models: ProactiveBuildResult[]
  onClose: () => void
}

export default function ModelComparator({ models, onClose }: ModelComparatorProps) {
  const [selectedMetric, setSelectedMetric] = useState<'duration' | 'status'>('duration')

  // Preparar datos para comparación
  const comparisonData = useMemo(() => {
    return models.map((model, index) => ({
      name: model.modelName.substring(0, 20),
      fullName: model.modelName,
      duration: model.duration ? Math.round(model.duration / 1000) : 0,
      status: model.status === 'completed' ? 1 : 0,
      success: model.status === 'completed' ? 100 : 0,
      index: index + 1,
    }))
  }, [models])

  // Datos para radar chart
  const radarData = useMemo(() => {
    return models.map((model) => ({
      model: model.modelName.substring(0, 15),
      duration: model.duration ? Math.min(Math.round(model.duration / 1000) / 10, 100) : 0,
      success: model.status === 'completed' ? 100 : 0,
      speed: model.duration ? Math.max(100 - Math.round(model.duration / 1000), 0) : 0,
    }))
  }, [models])

  // Estadísticas comparativas
  const stats = useMemo(() => {
    const durations = models.filter(m => m.duration).map(m => m.duration!)
    const successful = models.filter(m => m.status === 'completed').length
    
    return {
      avgDuration: durations.length > 0 
        ? Math.round(durations.reduce((a, b) => a + b, 0) / durations.length / 1000)
        : 0,
      minDuration: durations.length > 0 
        ? Math.round(Math.min(...durations) / 1000)
        : 0,
      maxDuration: durations.length > 0 
        ? Math.round(Math.max(...durations) / 1000)
        : 0,
      successRate: (successful / models.length) * 100,
      total: models.length,
      successful,
    }
  }, [models])

  // Exportar comparación
  const exportComparison = () => {
    const data = {
      models: comparisonData,
      stats,
      timestamp: Date.now(),
    }
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `model-comparison-${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  if (models.length < 2) {
    return null
  }

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.95 }}
        className="bg-slate-800 rounded-lg border border-slate-700 w-full max-w-6xl max-h-[90vh] overflow-hidden flex flex-col"
      >
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-slate-700">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-purple-600/20 rounded-lg">
              <BarChart3 className="w-5 h-5 text-purple-400" />
            </div>
            <div>
              <h3 className="text-lg font-bold text-white">Comparación de Modelos</h3>
              <p className="text-sm text-slate-400">{models.length} modelos comparados</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={exportComparison}
              className="p-2 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors"
              title="Exportar comparación"
            >
              <Download className="w-4 h-4 text-slate-300" />
            </button>
            <button
              onClick={onClose}
              className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
            >
              <X className="w-5 h-5 text-slate-400" />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Estadísticas Resumen */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-slate-700/30 rounded-lg p-4">
              <div className="text-xs text-slate-400 mb-1">Promedio</div>
              <div className="text-2xl font-bold text-white">{stats.avgDuration}s</div>
              <div className="text-xs text-slate-500">Duración promedio</div>
            </div>
            <div className="bg-slate-700/30 rounded-lg p-4">
              <div className="text-xs text-slate-400 mb-1">Rápido</div>
              <div className="text-2xl font-bold text-green-400">{stats.minDuration}s</div>
              <div className="text-xs text-slate-500">Más rápido</div>
            </div>
            <div className="bg-slate-700/30 rounded-lg p-4">
              <div className="text-xs text-slate-400 mb-1">Lento</div>
              <div className="text-2xl font-bold text-red-400">{stats.maxDuration}s</div>
              <div className="text-xs text-slate-500">Más lento</div>
            </div>
            <div className="bg-slate-700/30 rounded-lg p-4">
              <div className="text-xs text-slate-400 mb-1">Tasa Éxito</div>
              <div className="text-2xl font-bold text-purple-400">{stats.successRate.toFixed(1)}%</div>
              <div className="text-xs text-slate-500">{stats.successful}/{stats.total} exitosos</div>
            </div>
          </div>

          {/* Gráfico de Barras */}
          <div>
            <div className="flex items-center justify-between mb-4">
              <h4 className="text-sm font-semibold text-slate-300">Comparación de Duración</h4>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setSelectedMetric('duration')}
                  className={`px-3 py-1 rounded text-xs transition-colors ${
                    selectedMetric === 'duration'
                      ? 'bg-purple-600 text-white'
                      : 'bg-slate-700 text-slate-400 hover:text-white'
                  }`}
                >
                  Duración
                </button>
                <button
                  onClick={() => setSelectedMetric('status')}
                  className={`px-3 py-1 rounded text-xs transition-colors ${
                    selectedMetric === 'status'
                      ? 'bg-purple-600 text-white'
                      : 'bg-slate-700 text-slate-400 hover:text-white'
                  }`}
                >
                  Estado
                </button>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={comparisonData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
                <XAxis dataKey="name" stroke="#94a3b8" fontSize={12} />
                <YAxis stroke="#94a3b8" fontSize={12} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: '8px' }}
                />
                <Legend />
                <Bar 
                  dataKey={selectedMetric === 'duration' ? 'duration' : 'success'} 
                  fill={selectedMetric === 'duration' ? '#8b5cf6' : '#10b981'}
                  name={selectedMetric === 'duration' ? 'Duración (s)' : 'Éxito (%)'}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Radar Chart */}
          <div>
            <h4 className="text-sm font-semibold text-slate-300 mb-4">Comparación Multidimensional</h4>
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="#475569" />
                <PolarAngleAxis dataKey="model" stroke="#94a3b8" fontSize={10} />
                <PolarRadiusAxis angle={90} domain={[0, 100]} stroke="#94a3b8" />
                <Radar name="Duración" dataKey="duration" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.6} />
                <Radar name="Éxito" dataKey="success" stroke="#10b981" fill="#10b981" fillOpacity={0.6} />
                <Legend />
              </RadarChart>
            </ResponsiveContainer>
          </div>

          {/* Tabla Detallada */}
          <div>
            <h4 className="text-sm font-semibold text-slate-300 mb-4">Detalles por Modelo</h4>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-700">
                    <th className="text-left p-3 text-slate-400">Modelo</th>
                    <th className="text-left p-3 text-slate-400">Estado</th>
                    <th className="text-right p-3 text-slate-400">Duración</th>
                    <th className="text-left p-3 text-slate-400">Descripción</th>
                  </tr>
                </thead>
                <tbody>
                  {models.map((model, index) => (
                    <tr key={model.modelId} className="border-b border-slate-700/50 hover:bg-slate-700/20">
                      <td className="p-3 text-white font-medium">{model.modelName}</td>
                      <td className="p-3">
                        {model.status === 'completed' ? (
                          <span className="flex items-center gap-1 text-green-400">
                            <CheckCircle className="w-4 h-4" />
                            Completado
                          </span>
                        ) : (
                          <span className="flex items-center gap-1 text-red-400">
                            <XCircle className="w-4 h-4" />
                            Fallido
                          </span>
                        )}
                      </td>
                      <td className="p-3 text-right text-slate-300">
                        {model.duration ? `${Math.round(model.duration / 1000)}s` : '-'}
                      </td>
                      <td className="p-3 text-slate-400 text-xs truncate max-w-xs">
                        {model.description}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  )
}
