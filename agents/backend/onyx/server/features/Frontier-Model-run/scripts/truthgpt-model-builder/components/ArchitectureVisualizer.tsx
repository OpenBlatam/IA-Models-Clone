'use client'

import { motion } from 'framer-motion'
import { ModelSpec } from '@/lib/modules/analysis'
import { cn } from '@/lib/cn'

interface ArchitectureVisualizerProps {
  spec: ModelSpec
  className?: string
}

export default function ArchitectureVisualizer({ spec, className }: ArchitectureVisualizerProps) {
  const maxLayerSize = Math.max(...spec.layers)

  return (
    <div className={cn('space-y-4', className)}>
      <h3 className="text-sm font-medium text-white mb-4">Arquitectura del Modelo</h3>
      <div className="flex items-center justify-center gap-2 flex-wrap">
        {spec.layers.map((size, idx) => {
          const width = (size / maxLayerSize) * 100
          const isLast = idx === spec.layers.length - 1
          
          return (
            <div key={idx} className="flex flex-col items-center gap-2">
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: idx * 0.1 }}
                className={cn(
                  'relative rounded-lg border-2 flex items-center justify-center transition-all',
                  isLast
                    ? 'bg-gradient-to-br from-purple-600 to-pink-600 border-purple-400'
                    : 'bg-gradient-to-br from-slate-700 to-slate-800 border-slate-600'
                )}
                style={{
                  width: `${Math.max(width, 30)}px`,
                  height: '60px',
                }}
              >
                <span className="text-xs font-bold text-white">{size}</span>
              </motion.div>
              {idx < spec.layers.length - 1 && (
                <motion.div
                  initial={{ scaleX: 0 }}
                  animate={{ scaleX: 1 }}
                  transition={{ delay: idx * 0.1 + 0.05 }}
                  className="w-0.5 h-4 bg-gradient-to-b from-purple-500 to-pink-500"
                />
              )}
              <p className="text-xs text-slate-400 mt-1">
                {isLast ? spec.outputActivation : spec.activation}
              </p>
            </div>
          )
        })}
      </div>
      <div className="flex items-center justify-center gap-4 mt-4 pt-4 border-t border-slate-700">
        <div className="text-center">
          <p className="text-xs text-slate-400 mb-1">Total Capas</p>
          <p className="text-sm font-medium text-white">{spec.layers.length}</p>
        </div>
        <div className="w-px h-8 bg-slate-700" />
        <div className="text-center">
          <p className="text-xs text-slate-400 mb-1">Neuronas</p>
          <p className="text-sm font-medium text-white">{spec.layers.reduce((a, b) => a + b, 0)}</p>
        </div>
        <div className="w-px h-8 bg-slate-700" />
        <div className="text-center">
          <p className="text-xs text-slate-400 mb-1">Dropout</p>
          <p className="text-sm font-medium text-white">{spec.useDropout ? `${(spec.dropoutRate * 100).toFixed(0)}%` : 'No'}</p>
        </div>
      </div>
    </div>
  )
}


