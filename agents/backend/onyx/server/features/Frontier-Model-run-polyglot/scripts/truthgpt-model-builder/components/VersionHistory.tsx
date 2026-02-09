'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { GitBranch, Clock, Tag, ChevronRight, Copy, Check } from 'lucide-react'
import { getVersionHistory, type ModelVersion } from '@/lib/versioning'
import { format } from 'date-fns'
import { useCopyToClipboard } from '@/lib/hooks/useCopyToClipboard'
import { toast } from 'react-hot-toast'

interface VersionHistoryProps {
  modelId: string
  onSelectVersion?: (version: ModelVersion) => void
}

export default function VersionHistory({ modelId, onSelectVersion }: VersionHistoryProps) {
  const [versions, setVersions] = useState<ModelVersion[]>([])
  const [expandedVersion, setExpandedVersion] = useState<string | null>(null)
  const { copy, copied } = useCopyToClipboard()

  useEffect(() => {
    const history = getVersionHistory(modelId)
    setVersions(history.versions.sort((a, b) => 
      new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
    ))
  }, [modelId])

  const handleCopyVersionId = (versionId: string) => {
    copy(versionId)
    toast.success('ID de versión copiado')
  }

  const getVersionColor = (version: string) => {
    const [major] = version.split('.')
    if (parseInt(major) === 0) return 'text-slate-400'
    if (parseInt(major) === 1) return 'text-blue-400'
    if (parseInt(major) >= 2) return 'text-purple-400'
    return 'text-slate-400'
  }

  if (versions.length === 0) {
    return (
      <div className="text-center py-8 text-slate-400">
        <GitBranch className="w-12 h-12 mx-auto mb-4 opacity-50" />
        <p>No hay versiones guardadas</p>
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {versions.map((version, index) => (
        <motion.div
          key={version.version}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.05 }}
          className="bg-slate-700/30 rounded-lg border border-slate-600 overflow-hidden"
        >
          <div
            className="p-4 cursor-pointer hover:bg-slate-700/50 transition-colors"
            onClick={() => setExpandedVersion(
              expandedVersion === version.version ? null : version.version
            )}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3 flex-1">
                <Tag className={`w-5 h-5 ${getVersionColor(version.version)}`} />
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className={`font-mono font-semibold ${getVersionColor(version.version)}`}>
                      v{version.version}
                    </span>
                    {version.parentVersion && (
                      <span className="text-xs text-slate-500">
                        ← v{version.parentVersion}
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-3 text-xs text-slate-400">
                    <span className="flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      {format(version.createdAt, 'dd MMM yyyy, HH:mm')}
                    </span>
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        handleCopyVersionId(version.modelId)
                      }}
                      className="flex items-center gap-1 hover:text-slate-300 transition-colors"
                    >
                      {copied ? (
                        <>
                          <Check className="w-3 h-3" />
                          <span>Copiado</span>
                        </>
                      ) : (
                        <>
                          <Copy className="w-3 h-3" />
                          <span>Copiar ID</span>
                        </>
                      )}
                    </button>
                  </div>
                </div>
                <ChevronRight
                  className={`w-5 h-5 text-slate-400 transition-transform ${
                    expandedVersion === version.version ? 'rotate-90' : ''
                  }`}
                />
              </div>
            </div>
          </div>

          <AnimatePresence>
            {expandedVersion === version.version && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="border-t border-slate-600"
              >
                <div className="p-4 space-y-3">
                  {version.changes.length > 0 && (
                    <div>
                      <h4 className="text-sm font-semibold text-white mb-2">Cambios:</h4>
                      <ul className="space-y-1">
                        {version.changes.map((change, idx) => (
                          <li key={idx} className="text-sm text-slate-300 flex items-start gap-2">
                            <span className="text-purple-400 mt-1">•</span>
                            <span>{change}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                  
                  <div className="flex items-center gap-2 pt-2">
                    <button
                      onClick={() => onSelectVersion?.(version)}
                      className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg text-sm transition-colors"
                    >
                      Usar esta versión
                    </button>
                    {version.githubUrl && (
                      <a
                        href={version.githubUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg text-sm transition-colors"
                      >
                        Ver en GitHub
                      </a>
                    )}
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      ))}
    </div>
  )
}


