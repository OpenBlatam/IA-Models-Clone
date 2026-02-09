'use client'

import React, { useState } from 'react'
import { Download, FileText, Code } from 'lucide-react'
import { Button, Modal } from '../ui'
import toast from 'react-hot-toast'

interface ExportButtonProps {
  data: {
    original?: string
    improved?: string
    suggestions?: Array<{
      type: string
      description: string
      line?: number
    }>
    metadata?: Record<string, unknown>
  }
  filename?: string
  variant?: 'primary' | 'outline' | 'ghost'
  size?: 'sm' | 'md' | 'lg'
}

const ExportButton: React.FC<ExportButtonProps> = ({
  data,
  filename = 'code-improvement',
  variant = 'outline',
  size = 'md',
}) => {
  const [isModalOpen, setIsModalOpen] = useState(false)

  const handleExportJSON = () => {
    const json = JSON.stringify(data, null, 2)
    const blob = new Blob([json], { type: 'application/json' })
    downloadFile(blob, `${filename}.json`)
    setIsModalOpen(false)
    toast.success('Exported as JSON!')
  }

  const handleExportMarkdown = () => {
    let markdown = `# Code Improvement Report\n\n`
    markdown += `Generated: ${new Date().toLocaleString()}\n\n`
    
    if (data.original) {
      markdown += `## Original Code\n\n\`\`\`python\n${data.original}\n\`\`\`\n\n`
    }
    
    if (data.improved) {
      markdown += `## Improved Code\n\n\`\`\`python\n${data.improved}\n\`\`\`\n\n`
    }
    
    if (data.suggestions && data.suggestions.length > 0) {
      markdown += `## Suggestions\n\n`
      data.suggestions.forEach((suggestion, index) => {
        markdown += `### ${index + 1}. ${suggestion.type}\n`
        if (suggestion.line) {
          markdown += `**Line:** ${suggestion.line}\n\n`
        }
        markdown += `${suggestion.description}\n\n`
      })
    }

    const blob = new Blob([markdown], { type: 'text/markdown' })
    downloadFile(blob, `${filename}.md`)
    setIsModalOpen(false)
    toast.success('Exported as Markdown!')
  }

  const handleExportCode = (code: string, label: string) => {
    const blob = new Blob([code], { type: 'text/plain' })
    downloadFile(blob, `${filename}-${label}.txt`)
    toast.success(`Exported ${label} code!`)
  }

  const downloadFile = (blob: Blob, filename: string) => {
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  return (
    <>
      <Button
        variant={variant}
        size={size}
        onClick={() => setIsModalOpen(true)}
      >
        <Download className="w-4 h-4 mr-2" />
        Export
      </Button>

      <Modal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        title="Export Options"
        size="md"
      >
        <div className="space-y-3">
          <Button
            variant="outline"
            className="w-full justify-start"
            onClick={handleExportJSON}
          >
            <FileText className="w-4 h-4 mr-2" />
            Export as JSON
          </Button>
          <Button
            variant="outline"
            className="w-full justify-start"
            onClick={handleExportMarkdown}
          >
            <FileText className="w-4 h-4 mr-2" />
            Export as Markdown
          </Button>
          {data.original && (
            <Button
              variant="outline"
              className="w-full justify-start"
              onClick={() => handleExportCode(data.original!, 'original')}
            >
              <Code className="w-4 h-4 mr-2" />
              Export Original Code
            </Button>
          )}
          {data.improved && (
            <Button
              variant="outline"
              className="w-full justify-start"
              onClick={() => handleExportCode(data.improved!, 'improved')}
            >
              <Code className="w-4 h-4 mr-2" />
              Export Improved Code
            </Button>
          )}
        </div>
      </Modal>
    </>
  )
}

export default ExportButton

