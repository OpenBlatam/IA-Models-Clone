'use client'

import { motion } from 'framer-motion'
import { X, Eye, Settings, CheckCircle } from 'lucide-react'
import { OptimizationCoreConfig } from '@/lib/optimization-core-adapter'

interface ConfigPreviewProps {
  config: OptimizationCoreConfig
  modelName: string
  description: string
  onClose: () => void
  onConfirm: () => void
}

export default function ConfigPreview({
  config,
  modelName,
  description,
  onClose,
  onConfirm,
}: ConfigPreviewProps) {
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
              <Eye className="w-5 h-5 text-purple-400" />
            </div>
            <div>
              <h3 className="text-lg font-bold text-white">Preview de Configuración</h3>
              <p className="text-sm text-slate-400">{modelName}</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
            aria-label="Cerrar preview"
          >
            <X className="w-5 h-5 text-slate-400" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Descripción */}
          <div>
            <h4 className="text-sm font-semibold text-slate-300 mb-2">Descripción</h4>
            <p className="text-sm text-slate-400 bg-slate-700/30 p-3 rounded-lg">{description}</p>
          </div>

          {/* Modelo */}
          <div>
            <h4 className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2">
              <Settings className="w-4 h-4" />
              Configuración del Modelo
            </h4>
            <div className="bg-slate-700/30 rounded-lg p-4 space-y-3">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <span className="text-xs text-slate-400">Modelo Base</span>
                  <p className="text-sm text-white font-medium">{config.model.name_or_path}</p>
                </div>
                <div>
                  <span className="text-xs text-slate-400">Gradient Checkpointing</span>
                  <p className="text-sm text-white font-medium">
                    {config.model.gradient_checkpointing ? '✅ Habilitado' : '❌ Deshabilitado'}
                  </p>
                </div>
                <div>
                  <span className="text-xs text-slate-400">Atención</span>
                  <p className="text-sm text-white font-medium">{config.model.attention.backend}</p>
                </div>
                <div>
                  <span className="text-xs text-slate-400">KV Cache</span>
                  <p className="text-sm text-white font-medium">{config.model.kv_cache.type}</p>
                </div>
                {config.model.lora && (
                  <>
                    <div>
                      <span className="text-xs text-slate-400">LoRA</span>
                      <p className="text-sm text-white font-medium">
                        {config.model.lora.enabled ? '✅ Habilitado' : '❌ Deshabilitado'}
                      </p>
                    </div>
                    {config.model.lora.enabled && (
                      <div>
                        <span className="text-xs text-slate-400">LoRA R</span>
                        <p className="text-sm text-white font-medium">{config.model.lora.r}</p>
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          </div>

          {/* Entrenamiento */}
          <div>
            <h4 className="text-sm font-semibold text-slate-300 mb-3">Configuración de Entrenamiento</h4>
            <div className="bg-slate-700/30 rounded-lg p-4 space-y-3">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <span className="text-xs text-slate-400">Épocas</span>
                  <p className="text-sm text-white font-medium">{config.training.epochs}</p>
                </div>
                <div>
                  <span className="text-xs text-slate-400">Batch Size</span>
                  <p className="text-sm text-white font-medium">{config.training.train_batch_size}</p>
                </div>
                <div>
                  <span className="text-xs text-slate-400">Learning Rate</span>
                  <p className="text-sm text-white font-medium">{config.training.learning_rate}</p>
                </div>
                <div>
                  <span className="text-xs text-slate-400">Precisión</span>
                  <p className="text-sm text-white font-medium">{config.training.mixed_precision}</p>
                </div>
                <div>
                  <span className="text-xs text-slate-400">Optimizador</span>
                  <p className="text-sm text-white font-medium">{config.optimizer.type}</p>
                </div>
                <div>
                  <span className="text-xs text-slate-400">Fused AdamW</span>
                  <p className="text-sm text-white font-medium">
                    {config.training.fused_adamw ? '✅' : '❌'}
                  </p>
                </div>
                <div>
                  <span className="text-xs text-slate-400">Torch Compile</span>
                  <p className="text-sm text-white font-medium">
                    {config.training.torch_compile ? '✅' : '❌'}
                  </p>
                </div>
                <div>
                  <span className="text-xs text-slate-400">EMA</span>
                  <p className="text-sm text-white font-medium">
                    {config.ema.enabled ? '✅' : '❌'}
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Datos */}
          <div>
            <h4 className="text-sm font-semibold text-slate-300 mb-3">Configuración de Datos</h4>
            <div className="bg-slate-700/30 rounded-lg p-4 space-y-3">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <span className="text-xs text-slate-400">Dataset</span>
                  <p className="text-sm text-white font-medium">{config.data.dataset}</p>
                </div>
                <div>
                  <span className="text-xs text-slate-400">Max Sequence Length</span>
                  <p className="text-sm text-white font-medium">{config.data.max_seq_len}</p>
                </div>
                <div>
                  <span className="text-xs text-slate-400">Collate Function</span>
                  <p className="text-sm text-white font-medium">{config.data.collate}</p>
                </div>
                <div>
                  <span className="text-xs text-slate-400">Workers</span>
                  <p className="text-sm text-white font-medium">{config.data.num_workers}</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 p-6 border-t border-slate-700 bg-slate-800/50">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors text-white text-sm"
          >
            Cancelar
          </button>
          <button
            onClick={onConfirm}
            className="px-4 py-2 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 rounded-lg transition-colors text-white font-medium text-sm flex items-center gap-2"
          >
            <CheckCircle className="w-4 h-4" />
            Confirmar y Construir
          </button>
        </div>
      </motion.div>
    </div>
  )
}










