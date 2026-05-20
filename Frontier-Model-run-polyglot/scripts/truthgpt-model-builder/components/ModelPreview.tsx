'use client'

import { motion, AnimatePresence } from 'framer-motion'
import { X, Brain, Layers, Zap, TrendingUp, Settings } from 'lucide-react'
import { ModelSpec } from '@/lib/modules/analysis'
import CodeBlock from './CodeBlock'
import CostEstimator from './CostEstimator'
import { cn } from '@/lib/cn'

interface ModelPreviewProps {
  spec: ModelSpec
  modelName: string
  description: string
  onClose: () => void
  onConfirm: () => void
}

export default function ModelPreview({ spec, modelName, description, onClose, onConfirm }: ModelPreviewProps) {
  const getOptimizerName = () => {
    const opt = spec.optimizer.charAt(0).toUpperCase() + spec.optimizer.slice(1)
    return opt === 'Sgd' ? 'SGD' : opt === 'Adamw' ? 'AdamW' : opt
  }

  const getLossName = () => {
    return spec.loss.split('_').map((w: string) => w.charAt(0).toUpperCase() + w.slice(1)).join('')
  }

  const previewCode = `# ${modelName}
# Tipo: ${spec.type}
# Arquitectura: ${spec.architecture}

import truthgpt as tg

model = tg.Sequential([
    ${spec.layers.map((size: number, idx: number) => 
      idx === spec.layers.length - 1
        ? `tg.layers.Dense(${size}, activation='${spec.outputActivation}')`
        : `tg.layers.Dense(${size}, activation='${spec.activation}'),`
    ).join('\n    ')}
])

model.compile(
    optimizer=tg.optimizers.${getOptimizerName()}(learning_rate=${spec.learningRate}),
    loss=tg.losses.${getLossName()}(),
    metrics=${JSON.stringify(spec.metrics)}
)`

  const specs = [
    { icon: Brain, label: 'Tipo', value: spec.type },
    { icon: Layers, label: 'Arquitectura', value: spec.architecture },
    { icon: TrendingUp, label: 'Learning Rate', value: spec.learningRate.toString() },
    { icon: Zap, label: 'Optimizador', value: spec.optimizer },
    { icon: Settings, label: 'Épocas', value: spec.epochs.toString() },
  ]

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          onClick={(e) => e.stopPropagation()}
          className="bg-slate-800 rounded-lg border border-slate-700 max-w-4xl w-full max-h-[90vh] overflow-y-auto"
        >
          {/* Header */}
          <div className="sticky top-0 bg-slate-800/95 backdrop-blur-sm border-b border-slate-700 p-6 flex items-center justify-between">
            <div>
              <h2 className="text-xl font-bold text-white mb-1">Vista Previa del Modelo</h2>
              <p className="text-sm text-slate-400">{description}</p>
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
            >
              <X className="w-5 h-5 text-slate-400" />
            </button>
          </div>

          {/* Content */}
          <div className="p-6 space-y-6">
            {/* Specifications */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              {specs.map((item, idx) => {
                const Icon = item.icon
                return (
                  <div
                    key={idx}
                    className="p-4 bg-slate-700/50 rounded-lg border border-slate-600"
                  >
                    <Icon className="w-5 h-5 text-purple-400 mb-2" />
                    <p className="text-xs text-slate-400 mb-1">{item.label}</p>
                    <p className="text-sm font-medium text-white capitalize">{item.value}</p>
                  </div>
                )
              })}
            </div>

            {/* Architecture Info */}
            <div className="bg-slate-700/30 rounded-lg p-4 border border-slate-600">
              <h3 className="text-sm font-medium text-white mb-3">Arquitectura</h3>
              <div className="space-y-2">
                {spec.layers.map((size, idx) => (
                  <div key={idx} className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-full bg-gradient-to-r from-purple-500 to-pink-500 flex items-center justify-center text-white text-xs font-bold">
                      {idx + 1}
                    </div>
                    <div className="flex-1">
                      <p className="text-sm text-slate-300">
                        Capa {idx + 1}: {size} neuronas
                      </p>
                      <p className="text-xs text-slate-500">
                        {idx === spec.layers.length - 1 ? spec.outputActivation : spec.activation} activation
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Code Preview */}
            <div>
              <h3 className="text-sm font-medium text-white mb-3">Código Generado</h3>
              <CodeBlock code={previewCode} language="python" />
            </div>

            {/* Cost Estimator */}
            <CostEstimator spec={spec} />

            {/* Actions */}
            <div className="flex gap-3 pt-4 border-t border-slate-700">
              <button
                onClick={onClose}
                className="flex-1 px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg transition-colors"
              >
                Cancelar
              </button>
              <button
                onClick={onConfirm}
                className="flex-1 px-4 py-2 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white rounded-lg transition-all font-medium"
              >
                Crear Modelo
              </button>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  )
}

