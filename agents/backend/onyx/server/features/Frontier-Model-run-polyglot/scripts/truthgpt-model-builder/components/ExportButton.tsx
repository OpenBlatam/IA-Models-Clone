'use client'

import { useState } from 'react'
import { Download, FileJson, FileCode, FileText } from 'lucide-react'
import { toast } from 'react-hot-toast'
import { Model } from '@/store/modelStore'

interface ExportButtonProps {
  model: Model
}

export default function ExportButton({ model }: ExportButtonProps) {
  const [isExporting, setIsExporting] = useState(false)

  const exportModel = async (format: 'json' | 'code' | 'markdown') => {
    setIsExporting(true)
    try {
      let content = ''
      let filename = ''
      let mimeType = ''

      if (format === 'json') {
        content = JSON.stringify(model, null, 2)
        filename = `${model.name}-config.json`
        mimeType = 'application/json'
      } else if (format === 'code') {
        // Generate Python code
        content = `# ${model.name}
# Description: ${model.description}
# Generated: ${model.createdAt.toISOString()}

import truthgpt as tg
import numpy as np

# Model configuration
MODEL_CONFIG = {
    "name": "${model.name}",
    "description": "${model.description}",
    "type": "${model.spec?.type || 'custom'}",
    "architecture": "${model.spec?.architecture || 'dense'}",
    "status": "${model.status}",
    "github_url": "${model.githubUrl || ''}",
}

# Create your model here
# model = tg.Sequential([...])
`
        filename = `${model.name}.py`
        mimeType = 'text/python'
      } else if (format === 'markdown') {
        content = `# ${model.name}

## Description
${model.description}

## Status
${model.status}

## Specifications
- Type: ${model.spec?.type || 'N/A'}
- Architecture: ${model.spec?.architecture || 'N/A'}

## GitHub
${model.githubUrl ? `[View on GitHub](${model.githubUrl})` : 'Not published yet'}

## Created
${model.createdAt.toISOString()}
`
        filename = `${model.name}.md`
        mimeType = 'text/markdown'
      }

      // Create blob and download
      const blob = new Blob([content], { type: mimeType })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = filename
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)

      toast.success(`Modelo exportado como ${format.toUpperCase()}`)
    } catch (error) {
      console.error('Export error:', error)
      toast.error('Error al exportar el modelo')
    } finally {
      setIsExporting(false)
    }
  }

  return (
    <div className="relative group">
      <button
        disabled={isExporting}
        className="flex items-center gap-2 px-3 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg transition-colors text-sm disabled:opacity-50"
      >
        <Download className="w-4 h-4" />
        <span>Exportar</span>
      </button>
      <div className="absolute right-0 top-full mt-2 w-48 bg-slate-800 border border-slate-700 rounded-lg shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-10">
        <div className="p-2 space-y-1">
          <button
            onClick={() => exportModel('json')}
            className="w-full flex items-center gap-2 px-3 py-2 hover:bg-slate-700 rounded text-sm text-slate-300"
          >
            <FileJson className="w-4 h-4" />
            <span>JSON</span>
          </button>
          <button
            onClick={() => exportModel('code')}
            className="w-full flex items-center gap-2 px-3 py-2 hover:bg-slate-700 rounded text-sm text-slate-300"
          >
            <FileCode className="w-4 h-4" />
            <span>Python</span>
          </button>
          <button
            onClick={() => exportModel('markdown')}
            className="w-full flex items-center gap-2 px-3 py-2 hover:bg-slate-700 rounded text-sm text-slate-300"
          >
            <FileText className="w-4 h-4" />
            <span>Markdown</span>
          </button>
        </div>
      </div>
    </div>
  )
}


