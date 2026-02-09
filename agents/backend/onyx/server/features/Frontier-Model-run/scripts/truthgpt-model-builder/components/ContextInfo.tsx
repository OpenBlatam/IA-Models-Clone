'use client'

import { motion } from 'framer-motion'
import { Brain, Zap, Database, TrendingUp } from 'lucide-react'

interface ContextInfoProps {
  context: {
    domain?: string
    complexity?: string
    dataSize?: string
    performance?: string
    constraints?: string[]
  }
}

export default function ContextInfo({ context }: ContextInfoProps) {
  if (!context.domain && !context.complexity && !context.dataSize) {
    return null
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      className="mt-4 p-4 bg-slate-700/30 rounded-lg border border-slate-600"
    >
      <h4 className="text-sm font-medium text-white mb-3 flex items-center gap-2">
        <Brain className="w-4 h-4 text-purple-400" />
        Contexto Detectado
      </h4>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {context.domain && (
          <div className="flex items-center gap-2">
            <Database className="w-4 h-4 text-blue-400" />
            <div>
              <p className="text-xs text-slate-400">Dominio</p>
              <p className="text-sm font-medium text-white capitalize">{context.domain}</p>
            </div>
          </div>
        )}
        {context.complexity && (
          <div className="flex items-center gap-2">
            <Brain className="w-4 h-4 text-purple-400" />
            <div>
              <p className="text-xs text-slate-400">Complejidad</p>
              <p className="text-sm font-medium text-white capitalize">{context.complexity}</p>
            </div>
          </div>
        )}
        {context.dataSize && (
          <div className="flex items-center gap-2">
            <Database className="w-4 h-4 text-green-400" />
            <div>
              <p className="text-xs text-slate-400">Tamaño Datos</p>
              <p className="text-sm font-medium text-white capitalize">{context.dataSize}</p>
            </div>
          </div>
        )}
        {context.performance && (
          <div className="flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-yellow-400" />
            <div>
              <p className="text-xs text-slate-400">Rendimiento</p>
              <p className="text-sm font-medium text-white capitalize">{context.performance}</p>
            </div>
          </div>
        )}
      </div>
      {context.constraints && context.constraints.length > 0 && (
        <div className="mt-3 pt-3 border-t border-slate-600">
          <p className="text-xs text-slate-400 mb-2">Restricciones:</p>
          <div className="flex flex-wrap gap-2">
            {context.constraints.map((constraint, idx) => (
              <span
                key={idx}
                className="px-2 py-1 bg-slate-600/50 text-xs text-slate-300 rounded"
              >
                {constraint}
              </span>
            ))}
          </div>
        </div>
      )}
    </motion.div>
  )
}


