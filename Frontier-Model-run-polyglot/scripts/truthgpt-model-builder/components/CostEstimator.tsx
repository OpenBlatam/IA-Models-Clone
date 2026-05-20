'use client'

import { motion } from 'framer-motion'
import { DollarSign, TrendingUp, Server, Database, Zap, Info } from 'lucide-react'
import { estimateCost, formatCost, type CostEstimate } from '@/lib/cost-estimator'
import { ModelSpec } from '@/lib/adaptive-analyzer'

interface CostEstimatorProps {
  spec: ModelSpec
}

export default function CostEstimator({ spec }: CostEstimatorProps) {
  const cost = estimateCost({
    architecture: spec.architecture,
    layers: spec.layers,
    epochs: spec.epochs,
    batchSize: spec.batchSize,
    optimizer: spec.optimizer,
  })

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-slate-800/50 rounded-lg border border-slate-700 p-6"
    >
      <div className="flex items-center gap-2 mb-6">
        <DollarSign className="w-6 h-6 text-green-400" />
        <h3 className="text-xl font-bold text-white">Estimación de Costos</h3>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="bg-slate-700/50 rounded-lg p-4 border border-slate-600">
          <div className="flex items-center gap-2 mb-2">
            <TrendingUp className="w-5 h-5 text-blue-400" />
            <span className="text-sm text-slate-400">Entrenamiento</span>
          </div>
          <p className="text-2xl font-bold text-white">{formatCost(cost.training.total)}</p>
          <div className="mt-2 text-xs text-slate-500 space-y-1">
            <div>Compute: {formatCost(cost.training.compute)}</div>
            <div>Storage: {formatCost(cost.training.storage)}</div>
          </div>
        </div>

        <div className="bg-slate-700/50 rounded-lg p-4 border border-slate-600">
          <div className="flex items-center gap-2 mb-2">
            <Server className="w-5 h-5 text-purple-400" />
            <span className="text-sm text-slate-400">Despliegue</span>
          </div>
          <p className="text-2xl font-bold text-white">{formatCost(cost.deployment.total)}</p>
          <div className="mt-2 text-xs text-slate-500 space-y-1">
            <div>Hosting: {formatCost(cost.deployment.hosting)}</div>
            <div>API: {formatCost(cost.deployment.api)}</div>
          </div>
        </div>

        <div className="bg-gradient-to-br from-purple-600/20 to-pink-600/20 rounded-lg p-4 border border-purple-500/30">
          <div className="flex items-center gap-2 mb-2">
            <Zap className="w-5 h-5 text-yellow-400" />
            <span className="text-sm text-slate-400">Total</span>
          </div>
          <p className="text-2xl font-bold text-white">{formatCost(cost.total)}</p>
          <p className="text-xs text-slate-400 mt-2">Por mes estimado</p>
        </div>
      </div>

      <div className="space-y-2">
        <h4 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
          <Database className="w-4 h-4 text-purple-400" />
          Desglose Detallado
        </h4>
        {cost.breakdown.map((item, index) => (
          <div
            key={index}
            className="flex items-center justify-between p-3 bg-slate-700/30 rounded-lg border border-slate-600"
          >
            <div className="flex-1">
              <p className="text-sm font-medium text-white">{item.item}</p>
              <p className="text-xs text-slate-400">{item.category}</p>
            </div>
            <div className="text-right">
              <p className="text-sm font-semibold text-white">{formatCost(item.total)}</p>
              <p className="text-xs text-slate-500">
                {item.quantity} × {formatCost(item.unitCost)}
              </p>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-4 p-3 bg-blue-500/10 border border-blue-500/30 rounded-lg flex items-start gap-2">
        <Info className="w-4 h-4 text-blue-400 flex-shrink-0 mt-0.5" />
        <p className="text-xs text-slate-300">
          Estas son estimaciones aproximadas. Los costos reales pueden variar según el proveedor
          de cloud, la región y la demanda.
        </p>
      </div>
    </motion.div>
  )
}


