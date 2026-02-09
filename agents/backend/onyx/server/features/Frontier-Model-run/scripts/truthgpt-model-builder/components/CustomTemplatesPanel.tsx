'use client'

import { useState, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { X, Plus, Trash2, Edit, Star, Search, Filter, Download, Upload } from 'lucide-react'
import { toast } from 'react-hot-toast'
import { getCustomTemplates, CustomTemplate } from '@/lib/custom-templates'

interface CustomTemplatesPanelProps {
  onClose: () => void
  onSelectTemplate?: (template: CustomTemplate) => void
}

export default function CustomTemplatesPanel({ onClose, onSelectTemplate }: CustomTemplatesPanelProps) {
  const customTemplates = getCustomTemplates()
  const [templates, setTemplates] = useState(customTemplates.getAllTemplates())
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedCategory, setSelectedCategory] = useState<string>('all')
  const [showCreateForm, setShowCreateForm] = useState(false)

  const refreshTemplates = () => {
    setTemplates(customTemplates.getAllTemplates())
  }

  const categories = useMemo(() => {
    return ['all', ...customTemplates.getCategories()]
  }, [customTemplates])

  const filteredTemplates = useMemo(() => {
    let filtered = templates

    if (searchQuery) {
      filtered = customTemplates.searchTemplates(searchQuery)
    }

    if (selectedCategory !== 'all') {
      filtered = filtered.filter(t => t.category === selectedCategory)
    }

    return filtered.sort((a, b) => b.usageCount - a.usageCount)
  }, [templates, searchQuery, selectedCategory, customTemplates])

  const handleExport = (template: CustomTemplate) => {
    const json = customTemplates.exportTemplate(template.id)
    const blob = new Blob([json], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${template.name}-template.json`
    a.click()
    URL.revokeObjectURL(url)
    toast.success('Plantilla exportada', { icon: '📥' })
  }

  const handleImport = () => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = '.json'
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0]
      if (!file) return

      const reader = new FileReader()
      reader.onload = (event) => {
        try {
          const json = event.target?.result as string
          const template = customTemplates.importTemplate(json)
          if (template) {
            refreshTemplates()
            toast.success('Plantilla importada', { icon: '📤' })
          } else {
            toast.error('Error al importar plantilla')
          }
        } catch (error) {
          toast.error('Error al leer el archivo')
        }
      }
      reader.readAsText(file)
    }
    input.click()
  }

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
              <Star className="w-5 h-5 text-purple-400" />
            </div>
            <div>
              <h3 className="text-lg font-bold text-white">Plantillas Personalizadas</h3>
              <p className="text-sm text-slate-400">Crea y gestiona tus propias plantillas</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
          >
            <X className="w-5 h-5 text-slate-400" />
          </button>
        </div>

        {/* Search and Filters */}
        <div className="p-4 border-b border-slate-700 space-y-3">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-slate-400" />
            <input
              type="text"
              placeholder="Buscar plantillas..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full bg-slate-700/50 border border-slate-600 rounded-lg px-10 py-2 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-purple-500 text-sm"
            />
          </div>
          <div className="flex items-center gap-2 flex-wrap">
            {categories.map((category) => (
              <button
                key={category}
                onClick={() => setSelectedCategory(category)}
                className={`px-3 py-1 rounded-lg text-xs transition-colors ${
                  selectedCategory === category
                    ? 'bg-purple-600 text-white'
                    : 'bg-slate-700/50 text-slate-300 hover:bg-slate-700'
                }`}
              >
                {category === 'all' ? 'Todas' : category}
              </button>
            ))}
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-3">
          {filteredTemplates.length > 0 ? (
            filteredTemplates.map((template) => (
              <div
                key={template.id}
                className="bg-slate-700/30 rounded-lg p-4 border border-slate-600"
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <h4 className="text-sm font-medium text-white">{template.name}</h4>
                      {template.isPublic && (
                        <span className="text-xs px-2 py-0.5 bg-green-600/20 text-green-400 rounded">
                          Público
                        </span>
                      )}
                      <span className="text-xs text-slate-500">
                        {template.usageCount} usos
                      </span>
                    </div>
                    <p className="text-xs text-slate-400 mb-2">{template.description}</p>
                    <div className="flex flex-wrap gap-1">
                      {template.tags.map((tag) => (
                        <span
                          key={tag}
                          className="text-xs px-2 py-0.5 bg-purple-600/20 text-purple-400 rounded"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {onSelectTemplate && (
                      <button
                        onClick={() => {
                          onSelectTemplate(template)
                          customTemplates.incrementUsage(template.id)
                          refreshTemplates()
                          toast.success('Plantilla aplicada', { icon: '✅' })
                        }}
                        className="p-2 hover:bg-green-600/20 rounded transition-colors"
                        title="Usar plantilla"
                      >
                        <Star className="w-4 h-4 text-green-400" />
                      </button>
                    )}
                    <button
                      onClick={() => handleExport(template)}
                      className="p-2 hover:bg-blue-600/20 rounded transition-colors"
                      title="Exportar"
                    >
                      <Download className="w-4 h-4 text-blue-400" />
                    </button>
                    <button
                      onClick={() => {
                        if (window.confirm('¿Eliminar esta plantilla?')) {
                          customTemplates.deleteTemplate(template.id)
                          refreshTemplates()
                          toast('Plantilla eliminada', { icon: '🗑️' })
                        }
                      }}
                      className="p-2 hover:bg-red-600/20 rounded transition-colors"
                      title="Eliminar"
                    >
                      <Trash2 className="w-4 h-4 text-red-400" />
                    </button>
                  </div>
                </div>
              </div>
            ))
          ) : (
            <div className="text-center text-slate-400 py-8">
              <Star className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p>No hay plantillas personalizadas</p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-slate-700 p-4 flex items-center gap-2">
          <button
            onClick={() => setShowCreateForm(true)}
            className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg transition-colors text-white font-medium"
          >
            <Plus className="w-4 h-4" />
            <span>Crear Plantilla</span>
          </button>
          <button
            onClick={handleImport}
            className="flex items-center justify-center gap-2 px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors text-white"
          >
            <Upload className="w-4 h-4" />
            <span>Importar</span>
          </button>
        </div>
      </motion.div>
    </div>
  )
}










