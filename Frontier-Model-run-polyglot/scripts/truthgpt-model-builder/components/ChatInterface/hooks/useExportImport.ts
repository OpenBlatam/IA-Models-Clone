/**
 * Custom hook for export and import functionality
 * Handles exporting messages in various formats and importing data
 */

import { useState, useCallback } from 'react'

export interface ExportImportState {
  exportFormats: Set<'json' | 'txt' | 'md' | 'html' | 'csv' | 'pdf' | 'xml' | 'yaml'>
  showExportMenu: boolean
  importEnabled: boolean
  exportTemplates: Map<string, { name: string, format: string, template: string }>
  isExporting: boolean
  isImporting: boolean
}

export interface ExportImportActions {
  setExportFormats: (formats: Set<string>) => void
  setShowExportMenu: (show: boolean) => void
  setImportEnabled: (enabled: boolean) => void
  exportMessages: (messages: any[], format: string, options?: any) => Promise<string>
  importMessages: (file: File) => Promise<any[]>
  addExportTemplate: (name: string, format: string, template: string) => void
  removeExportTemplate: (name: string) => void
  exportToFile: (content: string, filename: string, mimeType: string) => void
}

export function useExportImport(): ExportImportState & ExportImportActions {
  const [exportFormats, setExportFormats] = useState<Set<'json' | 'txt' | 'md' | 'html' | 'csv' | 'pdf' | 'xml' | 'yaml'>>(
    new Set(['json', 'txt', 'md'])
  )
  const [showExportMenu, setShowExportMenu] = useState(false)
  const [importEnabled, setImportEnabled] = useState(true)
  const [exportTemplates, setExportTemplates] = useState<Map<string, { name: string, format: string, template: string }>>(new Map())
  const [isExporting, setIsExporting] = useState(false)
  const [isImporting, setIsImporting] = useState(false)

  const exportMessages = useCallback(async (
    messages: any[],
    format: string,
    options: any = {}
  ): Promise<string> => {
    setIsExporting(true)
    try {
      switch (format.toLowerCase()) {
        case 'json':
          return JSON.stringify(messages, null, 2)
        
        case 'txt':
          return messages.map(msg => 
            `${msg.role.toUpperCase()}: ${msg.content}`
          ).join('\n\n')
        
        case 'md':
          return messages.map(msg => 
            `## ${msg.role.toUpperCase()}\n\n${msg.content}`
          ).join('\n\n---\n\n')
        
        case 'html':
          return `<!DOCTYPE html>
<html>
<head>
  <title>Chat Export</title>
  <style>
    body { font-family: Arial, sans-serif; padding: 20px; }
    .message { margin: 10px 0; padding: 10px; border-left: 3px solid #007bff; }
    .user { background: #f0f0f0; }
    .assistant { background: #e8f4f8; }
  </style>
</head>
<body>
  ${messages.map(msg => 
    `<div class="message ${msg.role}">
      <strong>${msg.role.toUpperCase()}:</strong>
      <p>${msg.content.replace(/\n/g, '<br>')}</p>
    </div>`
  ).join('')}
</body>
</html>`
        
        case 'csv':
          return [
            'Role,Content,Timestamp',
            ...messages.map(msg => 
              `"${msg.role}","${msg.content.replace(/"/g, '""')}","${msg.timestamp || ''}"`
            )
          ].join('\n')
        
        case 'xml':
          return `<?xml version="1.0" encoding="UTF-8"?>
<messages>
  ${messages.map(msg => 
    `<message role="${msg.role}" timestamp="${msg.timestamp || ''}">
      <content><![CDATA[${msg.content}]]></content>
    </message>`
  ).join('\n  ')}
</messages>`
        
        case 'yaml':
          return `messages:\n${messages.map((msg, i) => 
            `  - id: ${i}\n    role: ${msg.role}\n    content: |\n      ${msg.content.split('\n').map((line: string) => `      ${line}`).join('\n')}\n    timestamp: ${msg.timestamp || ''}`
          ).join('\n')}`
        
        default:
          throw new Error(`Unsupported export format: ${format}`)
      }
    } finally {
      setIsExporting(false)
    }
  }, [])

  const importMessages = useCallback(async (file: File): Promise<any[]> => {
    setIsImporting(true)
    try {
      const text = await file.text()
      const extension = file.name.split('.').pop()?.toLowerCase()

      switch (extension) {
        case 'json':
          const jsonData = JSON.parse(text)
          return Array.isArray(jsonData) ? jsonData : []
        
        case 'txt':
        case 'md':
          // Simple text import - split by lines and try to detect role
          return text.split('\n\n').map((content, i) => ({
            id: `imported-${i}`,
            role: content.startsWith('USER:') || content.startsWith('user:') ? 'user' : 'assistant',
            content: content.replace(/^(USER|ASSISTANT|user|assistant):\s*/i, ''),
            timestamp: Date.now(),
          }))
        
        case 'csv':
          const lines = text.split('\n')
          const headers = lines[0].split(',')
          return lines.slice(1).map((line, i) => {
            const values = line.split(',')
            return {
              id: `imported-${i}`,
              role: values[0]?.replace(/"/g, '') || 'user',
              content: values[1]?.replace(/"/g, '') || '',
              timestamp: values[2] ? parseInt(values[2].replace(/"/g, '')) : Date.now(),
            }
          })
        
        default:
          throw new Error(`Unsupported import format: ${extension}`)
      }
    } catch (error) {
      console.error('Error importing messages:', error)
      throw error
    } finally {
      setIsImporting(false)
    }
  }, [])

  const addExportTemplate = useCallback((name: string, format: string, template: string) => {
    setExportTemplates(prev => {
      const next = new Map(prev)
      next.set(name, { name, format, template })
      return next
    })
  }, [])

  const removeExportTemplate = useCallback((name: string) => {
    setExportTemplates(prev => {
      const next = new Map(prev)
      next.delete(name)
      return next
    })
  }, [])

  const exportToFile = useCallback((content: string, filename: string, mimeType: string) => {
    const blob = new Blob([content], { type: mimeType })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = filename
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  }, [])

  return {
    // State
    exportFormats,
    showExportMenu,
    importEnabled,
    exportTemplates,
    isExporting,
    isImporting,
    // Actions
    setExportFormats,
    setShowExportMenu,
    setImportEnabled,
    exportMessages,
    importMessages,
    addExportTemplate,
    removeExportTemplate,
    exportToFile,
  }
}




