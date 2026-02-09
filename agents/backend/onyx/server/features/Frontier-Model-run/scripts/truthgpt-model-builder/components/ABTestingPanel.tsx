'use client'

import { useState, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { X, Plus, Trash2, Play, Pause, CheckCircle, BarChart3, TrendingUp } from 'lucide-react'
import { toast } from 'react-hot-toast'
import { getABTesting, ABTestConfig } from '@/lib/ab-testing'
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

interface ABTestingPanelProps {
  onClose: () => void
}

export default function ABTestingPanel({ onClose }: ABTestingPanelProps) {
  const abTesting = getABTesting()
  const [tests, setTests] = useState(abTesting.getAllTests())
  const [showCreateForm, setShowCreateForm] = useState(false)
  const [selectedTest, setSelectedTest] = useState<ABTestConfig | null>(null)

  const refreshTests = () => {
    setTests(abTesting.getAllTests())
  }

  const stats = useMemo(() => {
    if (!selectedTest) return null
    return abTesting.getComparisonStats(selectedTest.id)
  }, [selectedTest, abTesting])

  const chartData = useMemo(() => {
    if (!stats) return []
    return [
      {
        name: 'Variant A',
        'Tasa de Éxito': stats.variantA.successRate * 100,
        'Duración Promedio': stats.variantA.avgDuration / 1000,
        'Builds': stats.variantA.builds,
      },
      {
        name: 'Variant B',
        'Tasa de Éxito': stats.variantB.successRate * 100,
        'Duración Promedio': stats.variantB.avgDuration / 1000,
        'Builds': stats.variantB.builds,
      },
    ]
  }, [stats])

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.95 }}
        className="bg-slate-800 rounded-lg border border-slate-700 w-full max-w-4xl max-h-[90vh] overflow-hidden flex flex-col"
      >
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-slate-700">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-purple-600/20 rounded-lg">
              <BarChart3 className="w-5 h-5 text-purple-400" />
            </div>
            <div>
              <h3 className="text-lg font-bold text-white">A/B Testing</h3>
              <p className="text-sm text-slate-400">Compara configuraciones de modelos</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
          >
            <X className="w-5 h-5 text-slate-400" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {/* Tests List */}
          {tests.length > 0 ? (
            <div className="space-y-3">
              {tests.map((test) => (
                <div
                  key={test.id}
                  onClick={() => setSelectedTest(test)}
                  className={`bg-slate-700/30 rounded-lg p-4 border cursor-pointer transition-all ${
                    selectedTest?.id === test.id
                      ? 'border-purple-500 bg-purple-900/20'
                      : 'border-slate-600 hover:border-slate-500'
                  }`}
                >
                  <div className="flex items-start justify-between mb-2">
                    <div>
                      <h4 className="text-sm font-medium text-white">{test.name}</h4>
                      <p className="text-xs text-slate-400 mt-1">{test.description}</p>
                    </div>
                    <div className="flex items-center gap-2">
                      <span
                        className={`text-xs px-2 py-1 rounded ${
                          test.status === 'active'
                            ? 'bg-green-600/20 text-green-400'
                            : test.status === 'paused'
                            ? 'bg-yellow-600/20 text-yellow-400'
                            : 'bg-slate-600/20 text-slate-400'
                        }`}
                      >
                        {test.status}
                      </span>
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          abTesting.deleteTest(test.id)
                          refreshTests()
                          if (selectedTest?.id === test.id) {
                            setSelectedTest(null)
                          }
                          toast('Test eliminado', { icon: '🗑️' })
                        }}
                        className="p-1 hover:bg-red-600/20 rounded transition-colors"
                      >
                        <Trash2 className="w-4 h-4 text-red-400" />
                      </button>
                    </div>
                  </div>
                  {test.metrics && (
                    <div className="grid grid-cols-2 gap-4 mt-3 text-xs">
                      <div>
                        <span className="text-slate-400">Variant A:</span>
                        <span className="text-white ml-2">
                          {test.metrics.variantA.successRate * 100}% éxito
                        </span>
                      </div>
                      <div>
                        <span className="text-slate-400">Variant B:</span>
                        <span className="text-white ml-2">
                          {test.metrics.variantB.successRate * 100}% éxito
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center text-slate-400 py-8">
              <BarChart3 className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p>No hay tests A/B configurados</p>
            </div>
          )}

          {/* Stats for Selected Test */}
          {selectedTest && stats && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-slate-700/30 rounded-lg p-4 border border-slate-600 mt-4"
            >
              <h4 className="text-sm font-semibold text-white mb-4">Comparación de Variantes</h4>
              
              {/* Charts */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <div>
                  <h5 className="text-xs text-slate-400 mb-2">Tasa de Éxito</h5>
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="name" stroke="#9CA3AF" />
                      <YAxis stroke="#9CA3AF" />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                      />
                      <Bar dataKey="Tasa de Éxito" fill="#9333EA" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div>
                  <h5 className="text-xs text-slate-400 mb-2">Duración Promedio (s)</h5>
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="name" stroke="#9CA3AF" />
                      <YAxis stroke="#9CA3AF" />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                      />
                      <Bar dataKey="Duración Promedio" fill="#EC4899" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Winner */}
              {stats.winner && stats.winner !== 'tie' && (
                <div className="bg-purple-600/20 border border-purple-500/50 rounded-lg p-3 mt-4">
                  <div className="flex items-center gap-2">
                    <TrendingUp className="w-4 h-4 text-purple-400" />
                    <span className="text-sm font-medium text-purple-300">
                      Variante {stats.winner} está ganando
                    </span>
                    {stats.confidence !== undefined && (
                      <span className="text-xs text-purple-400 ml-auto">
                        {Math.round(stats.confidence)}% confianza
                      </span>
                    )}
                  </div>
                </div>
              )}
            </motion.div>
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-slate-700 p-4">
          <button
            onClick={() => setShowCreateForm(true)}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg transition-colors text-white font-medium"
          >
            <Plus className="w-4 h-4" />
            <span>Crear Nuevo Test A/B</span>
          </button>
        </div>
      </motion.div>
    </div>
  )
}










