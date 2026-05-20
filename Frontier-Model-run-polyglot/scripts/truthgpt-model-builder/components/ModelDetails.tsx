'use client'

import { motion } from 'framer-motion'
import { X, Github, Code, Book, Download, Share2, GitBranch } from 'lucide-react'
import { Model } from '@/store/modelStore'
import ArchitectureVisualizer from './ArchitectureVisualizer'
import PerformanceMetrics from './PerformanceMetrics'
import ExportButton from './ExportButton'
import ShareModal from './ShareModal'
import VersionHistory from './VersionHistory'
import ModelTags from './ModelTags'
import ModelCloner from './ModelCloner'
import CostEstimator from './CostEstimator'
import { useState } from 'react'

interface ModelDetailsProps {
  model: Model
  isOpen: boolean
  onClose: () => void
}

export default function ModelDetails({ model, isOpen, onClose }: ModelDetailsProps) {
  const [showShare, setShowShare] = useState(false)
  const [showCloner, setShowCloner] = useState(false)

  if (!isOpen) return null

  return (
    <>
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
          <div className="sticky top-0 bg-slate-800/95 backdrop-blur-sm border-b border-slate-700 p-6 flex items-center justify-between">
            <div>
              <h2 className="text-xl font-bold text-white mb-1">{model.name}</h2>
              <p className="text-sm text-slate-400">{model.description}</p>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setShowCloner(true)}
                className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
                title="Clonar modelo"
              >
                <GitBranch className="w-5 h-5 text-slate-300" />
              </button>
              <button
                onClick={() => setShowShare(true)}
                className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
                title="Compartir"
              >
                <Share2 className="w-5 h-5 text-slate-300" />
              </button>
              <ExportButton model={model} />
              <button
                onClick={onClose}
                className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
              >
                <X className="w-5 h-5 text-slate-400" />
              </button>
            </div>
          </div>

          <div className="p-6 space-y-6">
            {/* Status */}
            <div className="flex items-center gap-4">
              <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                model.status === 'completed' ? 'bg-green-500/20 text-green-400' :
                model.status === 'creating' ? 'bg-blue-500/20 text-blue-400' :
                'bg-red-500/20 text-red-400'
              }`}>
                {model.status}
              </span>
              {model.githubUrl && (
                <a
                  href={model.githubUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2 text-purple-400 hover:text-purple-300 text-sm"
                >
                  <Github className="w-4 h-4" />
                  Ver en GitHub
                </a>
              )}
            </div>

            {/* Architecture */}
            {model.spec && (
              <div className="p-4 bg-slate-700/30 rounded-lg border border-slate-600">
                <ArchitectureVisualizer spec={model.spec as any} />
              </div>
            )}

            {/* Tags */}
            <ModelTags modelId={model.id} />

            {/* Performance Metrics */}
            <PerformanceMetrics modelId={model.id} />

            {/* Cost Estimator */}
            {model.spec && <CostEstimator spec={model.spec as any} />}

            {/* Version History */}
            <div className="p-4 bg-slate-700/30 rounded-lg border border-slate-600">
              <h3 className="text-sm font-semibold text-white mb-4">Historial de Versiones</h3>
              <VersionHistory modelId={model.id} />
            </div>

            {/* Specifications */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="p-4 bg-slate-700/30 rounded-lg border border-slate-600">
                <p className="text-xs text-slate-400 mb-1">Tipo</p>
                <p className="text-sm font-medium text-white capitalize">{model.spec?.type || 'N/A'}</p>
              </div>
              <div className="p-4 bg-slate-700/30 rounded-lg border border-slate-600">
                <p className="text-xs text-slate-400 mb-1">Arquitectura</p>
                <p className="text-sm font-medium text-white uppercase">{model.spec?.architecture || 'N/A'}</p>
              </div>
              <div className="p-4 bg-slate-700/30 rounded-lg border border-slate-600">
                <p className="text-xs text-slate-400 mb-1">Creado</p>
                <p className="text-sm font-medium text-white">
                  {model.createdAt.toLocaleDateString()}
                </p>
              </div>
              <div className="p-4 bg-slate-700/30 rounded-lg border border-slate-600">
                <p className="text-xs text-slate-400 mb-1">Progreso</p>
                <p className="text-sm font-medium text-white">
                  {model.progress || 0}%
                </p>
              </div>
            </div>
          </div>
        </motion.div>
      </motion.div>

      {showShare && (
        <ShareModal
          model={model}
          isOpen={showShare}
          onClose={() => setShowShare(false)}
        />
      )}

      {showCloner && model.spec && (
        <ModelCloner
          model={{
            id: model.id,
            name: model.name,
            description: model.description,
            spec: model.spec as any,
          }}
          isOpen={showCloner}
          onClose={() => setShowCloner(false)}
          onClone={(clonedModel) => {
            // Handle cloned model
            console.log('Model cloned:', clonedModel)
          }}
        />
      )}
    </>
  )
}

