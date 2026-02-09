'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Copy, X, GitBranch, Check } from 'lucide-react'
import { toast } from 'react-hot-toast'
import { useCopyToClipboard } from '@/lib/hooks/useCopyToClipboard'
import { cloneModelAsVersion, generateNextVersion } from '@/lib/versioning'
import { ModelSpec } from '@/lib/adaptive-analyzer'

interface ModelClonerProps {
  model: {
    id: string
    name: string
    description: string
    spec?: ModelSpec
  }
  isOpen: boolean
  onClose: () => void
  onClone: (clonedModel: any) => void
}

export default function ModelCloner({ model, isOpen, onClose, onClone }: ModelClonerProps) {
  const [changes, setChanges] = useState<string[]>([''])
  const [versionType, setVersionType] = useState<'major' | 'minor' | 'patch'>('minor')
  const { copy, copied } = useCopyToClipboard()

  const handleAddChange = () => {
    setChanges([...changes, ''])
  }

  const handleChangeUpdate = (index: number, value: string) => {
    const updated = [...changes]
    updated[index] = value
    setChanges(updated.filter(c => c.trim() !== ''))
  }

  const handleRemoveChange = (index: number) => {
    const updated = changes.filter((_, i) => i !== index)
    setChanges(updated.length > 0 ? updated : [''])
  }

  const handleClone = async () => {
    if (!model.spec) {
      toast.error('El modelo no tiene especificaciones para clonar')
      return
    }

    const validChanges = changes.filter(c => c.trim() !== '')
    if (validChanges.length === 0) {
      toast.error('Agrega al menos un cambio descrito')
      return
    }

    try {
      // Generate model code (simplified - in real app would fetch from API)
      const modelCode = `# Cloned from ${model.name}\n# Changes: ${validChanges.join(', ')}\n`

      const clonedVersion = cloneModelAsVersion(
        model.id,
        validChanges,
        model.spec,
        modelCode,
        versionType
      )

      toast.success(`Modelo clonado como versión ${clonedVersion.version}`)
      onClone({
        ...model,
        id: clonedVersion.modelId,
        name: `${model.name}-v${clonedVersion.version}`,
        version: clonedVersion.version,
        changes: validChanges,
      })
      onClose()

      // Reset form
      setChanges([''])
      setVersionType('minor')
    } catch (error) {
      toast.error('Error al clonar el modelo')
      console.error(error)
    }
  }

  const nextVersion = generateNextVersion(model.id, versionType)

  if (!isOpen) return null

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center p-4 z-50"
        onClick={onClose}
      >
        <motion.div
          initial={{ y: -50, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          exit={{ y: -50, opacity: 0 }}
          className="bg-slate-800 rounded-lg shadow-xl max-w-2xl w-full relative border border-slate-700"
          onClick={(e) => e.stopPropagation()}
        >
          <button
            onClick={onClose}
            className="absolute top-4 right-4 text-slate-400 hover:text-white transition-colors"
          >
            <X className="w-6 h-6" />
          </button>

          <div className="p-8">
            <div className="flex items-center gap-3 mb-6">
              <GitBranch className="w-8 h-8 text-purple-400" />
              <div>
                <h2 className="text-2xl font-bold text-white">Clonar Modelo</h2>
                <p className="text-sm text-slate-400">Crear una nueva versión basada en: {model.name}</p>
              </div>
            </div>

            <div className="space-y-6">
              {/* Version Type */}
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Tipo de Versión
                </label>
                <div className="grid grid-cols-3 gap-3">
                  {(['major', 'minor', 'patch'] as const).map(type => (
                    <button
                      key={type}
                      onClick={() => setVersionType(type)}
                      className={`p-3 rounded-lg border transition-colors ${
                        versionType === type
                          ? 'bg-purple-600 border-purple-500 text-white'
                          : 'bg-slate-700/50 border-slate-600 text-slate-300 hover:bg-slate-700'
                      }`}
                    >
                      <div className="text-sm font-semibold capitalize">{type}</div>
                      <div className="text-xs mt-1 opacity-75">v{generateNextVersion(model.id, type)}</div>
                    </button>
                  ))}
                </div>
                <p className="text-xs text-slate-500 mt-2">
                  Nueva versión: <span className="font-mono text-purple-400">v{nextVersion}</span>
                </p>
              </div>

              {/* Changes */}
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Cambios en esta versión
                </label>
                <div className="space-y-2">
                  {changes.map((change, index) => (
                    <div key={index} className="flex items-center gap-2">
                      <input
                        type="text"
                        value={change}
                        onChange={(e) => handleChangeUpdate(index, e.target.value)}
                        placeholder={`Cambio ${index + 1}...`}
                        className="flex-1 bg-slate-700/50 border border-slate-600 rounded-lg px-4 py-2 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-purple-500"
                      />
                      {changes.length > 1 && (
                        <button
                          onClick={() => handleRemoveChange(index)}
                          className="p-2 bg-red-500/20 hover:bg-red-500/30 text-red-400 rounded-lg transition-colors"
                        >
                          <X className="w-4 h-4" />
                        </button>
                      )}
                    </div>
                  ))}
                  <button
                    onClick={handleAddChange}
                    className="w-full py-2 bg-slate-700/50 hover:bg-slate-700 border border-slate-600 border-dashed rounded-lg text-slate-400 hover:text-slate-300 transition-colors text-sm"
                  >
                    + Agregar cambio
                  </button>
                </div>
              </div>

              {/* Preview */}
              <div className="bg-slate-700/30 rounded-lg p-4 border border-slate-600">
                <h4 className="text-sm font-semibold text-white mb-2">Nueva Versión</h4>
                <div className="space-y-1 text-sm">
                  <div className="flex items-center justify-between">
                    <span className="text-slate-400">Nombre:</span>
                    <span className="font-mono text-white">{model.name}-v{nextVersion}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-slate-400">Versión:</span>
                    <span className="font-mono text-purple-400">v{nextVersion}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-slate-400">Cambios:</span>
                    <span className="text-white">{changes.filter(c => c.trim()).length}</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="flex justify-end gap-3 mt-6">
              <button
                onClick={onClose}
                className="px-6 py-3 rounded-lg border border-slate-600 text-slate-300 hover:bg-slate-700 transition-colors"
              >
                Cancelar
              </button>
              <button
                onClick={handleClone}
                className="px-6 py-3 rounded-lg bg-gradient-to-r from-purple-600 to-pink-600 text-white hover:from-purple-700 hover:to-pink-700 transition-all flex items-center gap-2"
              >
                <Copy className="w-5 h-5" />
                Clonar Modelo
              </button>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  )
}


