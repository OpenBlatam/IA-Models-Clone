'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Tag as TagIcon, Plus, X, Star } from 'lucide-react'
import {
  getAllTags,
  getModelTags,
  setModelTags,
  addTagToModel,
  removeTagFromModel,
  isFavorite,
  toggleFavorite,
  type Tag,
} from '@/lib/tags'
import { toast } from 'react-hot-toast'

interface ModelTagsProps {
  modelId: string
  onTagsChange?: (tags: string[]) => void
}

export default function ModelTags({ modelId, onTagsChange }: ModelTagsProps) {
  const [allTags, setAllTags] = useState<Tag[]>([])
  const [modelTags, setModelTagsList] = useState<string[]>([])
  const [isFav, setIsFav] = useState(false)
  const [showAddTag, setShowAddTag] = useState(false)

  useEffect(() => {
    setAllTags(getAllTags())
    setModelTagsList(getModelTags(modelId))
    setIsFav(isFavorite(modelId))
  }, [modelId])

  const handleAddTag = (tagId: string) => {
    addTagToModel(modelId, tagId)
    const updated = getModelTags(modelId)
    setModelTagsList(updated)
    onTagsChange?.(updated)
    toast.success('Etiqueta agregada')
  }

  const handleRemoveTag = (tagId: string) => {
    removeTagFromModel(modelId, tagId)
    const updated = getModelTags(modelId)
    setModelTagsList(updated)
    onTagsChange?.(updated)
    toast.success('Etiqueta eliminada')
  }

  const handleToggleFavorite = () => {
    const newState = toggleFavorite(modelId)
    setIsFav(newState)
    toast.success(newState ? 'Agregado a favoritos' : 'Eliminado de favoritos')
  }

  const getTagById = (tagId: string): Tag | undefined => {
    return allTags.find(t => t.id === tagId)
  }

  const availableTags = allTags.filter(t => !modelTags.includes(t.id))

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-semibold text-white flex items-center gap-2">
          <TagIcon className="w-4 h-4 text-purple-400" />
          Etiquetas
        </h4>
        <div className="flex items-center gap-2">
          <button
            onClick={handleToggleFavorite}
            className={`p-1.5 rounded-lg transition-colors ${
              isFav
                ? 'bg-yellow-500/20 text-yellow-400'
                : 'bg-slate-700/50 text-slate-400 hover:bg-slate-600'
            }`}
            title={isFav ? 'Eliminar de favoritos' : 'Agregar a favoritos'}
          >
            <Star className={`w-4 h-4 ${isFav ? 'fill-current' : ''}`} />
          </button>
          {availableTags.length > 0 && (
            <button
              onClick={() => setShowAddTag(!showAddTag)}
              className="p-1.5 bg-slate-700/50 hover:bg-slate-600 rounded-lg transition-colors"
              title="Agregar etiqueta"
            >
              <Plus className="w-4 h-4 text-slate-300" />
            </button>
          )}
        </div>
      </div>

      {modelTags.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {modelTags.map(tagId => {
            const tag = getTagById(tagId)
            if (!tag) return null

            return (
              <motion.div
                key={tagId}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm"
                style={{ backgroundColor: `${tag.color}20`, borderColor: tag.color, borderWidth: '1px' }}
              >
                <span className="text-white" style={{ color: tag.color }}>
                  {tag.name}
                </span>
                <button
                  onClick={() => handleRemoveTag(tagId)}
                  className="hover:bg-black/20 rounded-full p-0.5 transition-colors"
                >
                  <X className="w-3 h-3" style={{ color: tag.color }} />
                </button>
              </motion.div>
            )
          })}
        </div>
      )}

      {showAddTag && availableTags.length > 0 && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          className="bg-slate-700/50 rounded-lg p-3 border border-slate-600"
        >
          <p className="text-xs text-slate-400 mb-2">Agregar etiqueta:</p>
          <div className="flex flex-wrap gap-2">
            {availableTags.map(tag => (
              <button
                key={tag.id}
                onClick={() => {
                  handleAddTag(tag.id)
                  setShowAddTag(false)
                }}
                className="px-3 py-1.5 rounded-full text-sm border transition-colors hover:opacity-80"
                style={{
                  backgroundColor: `${tag.color}20`,
                  borderColor: tag.color,
                  color: tag.color,
                }}
              >
                {tag.name}
              </button>
            ))}
          </div>
        </motion.div>
      )}

      {modelTags.length === 0 && !showAddTag && (
        <p className="text-sm text-slate-500">No hay etiquetas. Haz clic en + para agregar.</p>
      )}
    </div>
  )
}


