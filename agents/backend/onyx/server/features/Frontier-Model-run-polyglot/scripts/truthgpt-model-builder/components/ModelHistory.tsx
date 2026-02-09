'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Github, CheckCircle, Clock, AlertCircle, ExternalLink, Trash2, Share2 } from 'lucide-react'
import { format } from 'date-fns'
import { useModelStore } from '@/store/modelStore'
import { toast } from 'react-hot-toast'
import { deleteModelFromHistory } from '@/lib/storage'
import ExportButton from './ExportButton'
import SearchBar from './SearchBar'
import ShareModal from './ShareModal'
import ModelDetails from './ModelDetails'

interface ModelHistoryItem {
  id: string
  name: string
  description: string
  status: 'creating' | 'completed' | 'failed'
  githubUrl: string | null
  createdAt: Date
  spec?: {
    type: string
    architecture: string
  }
}

interface ModelHistoryProps {
  models: ModelHistoryItem[]
  onSelectModel?: (model: ModelHistoryItem) => void
}

export default function ModelHistory({ models, onSelectModel }: ModelHistoryProps) {
  const [searchQuery, setSearchQuery] = useState('')
  const [filteredModels, setFilteredModels] = useState(models)
  const [shareModel, setShareModel] = useState<ModelHistoryItem | null>(null)
  const [selectedModel, setSelectedModel] = useState<ModelHistoryItem | null>(null)
  const { setCurrentModel } = useModelStore()

  useEffect(() => {
    if (!searchQuery.trim()) {
      setFilteredModels(models)
      return
    }

    const query = searchQuery.toLowerCase()
    const filtered = models.filter(model => 
      model.name.toLowerCase().includes(query) ||
      model.description.toLowerCase().includes(query) ||
      model.spec?.type?.toLowerCase().includes(query) ||
      model.spec?.architecture?.toLowerCase().includes(query)
    )
    setFilteredModels(filtered)
  }, [searchQuery, models])

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-400" />
      case 'creating':
        return <Clock className="w-5 h-5 text-blue-400 animate-pulse" />
      case 'failed':
        return <AlertCircle className="w-5 h-5 text-red-400" />
      default:
        return <Clock className="w-5 h-5 text-slate-400" />
    }
  }

  const getStatusText = (status: string) => {
    switch (status) {
      case 'completed':
        return 'Completado'
      case 'creating':
        return 'En creación'
      case 'failed':
        return 'Error'
      default:
        return 'Pendiente'
    }
  }

  const handleModelClick = (model: ModelHistoryItem) => {
    setSelectedModel(model)
  }

  const handleDelete = (modelId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    if (confirm('¿Estás seguro de eliminar este modelo del historial?')) {
      deleteModelFromHistory(modelId)
      toast.success('Modelo eliminado del historial')
      // Refresh models
      setFilteredModels(prev => prev.filter(m => m.id !== modelId))
    }
  }

  if (models.length === 0) {
    return (
      <div className="text-center py-12 text-slate-400">
        <p className="text-lg mb-2">No hay modelos creados aún</p>
        <p className="text-sm">Crea tu primer modelo usando el chat</p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <SearchBar onSearch={setSearchQuery} />
      
      {filteredModels.length === 0 ? (
        <div className="text-center py-8 text-slate-400">
          <p>No se encontraron modelos con esa búsqueda</p>
        </div>
      ) : (
        filteredModels.map((model, index) => (
        <motion.div
          key={model.id}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.05 }}
          onClick={() => handleModelClick(model)}
          className="group p-4 bg-slate-800/50 hover:bg-slate-700/50 border border-slate-700 rounded-lg cursor-pointer transition-all"
        >
          <div className="flex items-start justify-between gap-4">
            <div className="flex items-start gap-3 flex-1">
              {getStatusIcon(model.status)}
              <div className="flex-1 min-w-0">
                <h3 className="text-sm font-medium text-white mb-1 truncate">{model.name}</h3>
                <p className="text-xs text-slate-400 mb-2 line-clamp-2">{model.description}</p>
                <div className="flex items-center gap-3 text-xs text-slate-500">
                  <span>{getStatusText(model.status)}</span>
                  <span>•</span>
                  <span>{format(model.createdAt, 'dd MMM yyyy, HH:mm')}</span>
                  {model.spec && (
                    <>
                      <span>•</span>
                      <span className="capitalize">{model.spec.type}</span>
                      <span>•</span>
                      <span className="uppercase">{model.spec.architecture}</span>
                    </>
                  )}
                </div>
              </div>
            </div>
            <div className="flex items-center gap-2 flex-shrink-0">
              <ExportButton model={model as any} />
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  setShareModel(model)
                }}
                className="p-2 hover:bg-slate-600 rounded-lg transition-colors"
                title="Compartir modelo"
              >
                <Share2 className="w-4 h-4 text-slate-400" />
              </button>
              {model.githubUrl && (
                <a
                  href={model.githubUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  onClick={(e) => e.stopPropagation()}
                  className="p-2 hover:bg-slate-600 rounded-lg transition-colors"
                  title="Ver en GitHub"
                >
                  <Github className="w-4 h-4 text-slate-400" />
                </a>
              )}
              <button
                onClick={(e) => handleDelete(model.id, e)}
                className="p-2 hover:bg-red-600/20 rounded-lg transition-colors"
                title="Eliminar del historial"
              >
                <Trash2 className="w-4 h-4 text-red-400" />
              </button>
            </div>
          </div>
        </motion.div>
        ))
      )}

      {/* Share Modal */}
      {shareModel && (
        <ShareModal
          model={shareModel as any}
          isOpen={!!shareModel}
          onClose={() => setShareModel(null)}
        />
      )}

      {/* Model Details Modal */}
      {selectedModel && (
        <ModelDetails
          model={selectedModel as any}
          isOpen={!!selectedModel}
          onClose={() => setSelectedModel(null)}
        />
      )}
    </div>
  )
}

