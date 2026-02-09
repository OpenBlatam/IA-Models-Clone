'use client'

import React, { useState } from 'react'
import { GitCompare, Copy, Download, Maximize2 } from 'lucide-react'
import { Card, Button, Badge } from '../ui'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { Modal } from '../ui'

interface CodeComparisonProps {
  original: string
  improved: string
  language?: string
  suggestions?: Array<{
    type: string
    description: string
    line?: number
    severity?: string
  }>
}

const CodeComparison: React.FC<CodeComparisonProps> = ({
  original,
  improved,
  language = 'python',
  suggestions = [],
}) => {
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [activeView, setActiveView] = useState<'split' | 'original' | 'improved'>('split')

  const handleCopy = (text: string, label: string) => {
    navigator.clipboard.writeText(text)
  }

  const handleDownload = (content: string, filename: string) => {
    const blob = new Blob([content], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const renderCode = (code: string, title: string, isLeft = true) => (
    <div className="flex-1">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <h3 className="font-semibold text-gray-900">{title}</h3>
          {suggestions.length > 0 && (
            <Badge variant="info" size="sm">
              {suggestions.length} suggestions
            </Badge>
          )}
        </div>
        <div className="flex gap-1">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => handleCopy(code, title)}
            title="Copy to clipboard"
          >
            <Copy className="w-4 h-4" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => handleDownload(code, `${title.toLowerCase().replace(' ', '-')}.txt`)}
            title="Download"
          >
            <Download className="w-4 h-4" />
          </Button>
        </div>
      </div>
      <div className="overflow-x-auto border border-gray-200 rounded-lg">
        <SyntaxHighlighter
          language={language}
          style={vscDarkPlus}
          customStyle={{
            borderRadius: '8px',
            fontSize: '14px',
            margin: 0,
            padding: '1rem',
          }}
        >
          {code}
        </SyntaxHighlighter>
      </div>
    </div>
  )

  const content = (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <GitCompare className="w-5 h-5 text-primary-600" />
          <h2 className="text-lg font-semibold text-gray-900">Code Comparison</h2>
        </div>
        <div className="flex gap-2">
          <Button
            variant={activeView === 'split' ? 'primary' : 'outline'}
            size="sm"
            onClick={() => setActiveView('split')}
          >
            Split
          </Button>
          <Button
            variant={activeView === 'original' ? 'primary' : 'outline'}
            size="sm"
            onClick={() => setActiveView('original')}
          >
            Original
          </Button>
          <Button
            variant={activeView === 'improved' ? 'primary' : 'outline'}
            size="sm"
            onClick={() => setActiveView('improved')}
          >
            Improved
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setIsFullscreen(true)}
          >
            <Maximize2 className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {activeView === 'split' && (
        <div className="grid md:grid-cols-2 gap-4">
          {renderCode(original, 'Original Code', true)}
          {renderCode(improved, 'Improved Code', false)}
        </div>
      )}

      {activeView === 'original' && (
        <div>{renderCode(original, 'Original Code')}</div>
      )}

      {activeView === 'improved' && (
        <div>{renderCode(improved, 'Improved Code')}</div>
      )}
    </div>
  )

  return (
    <>
      <Card>{content}</Card>

      {isFullscreen && (
        <Modal
          isOpen={isFullscreen}
          onClose={() => setIsFullscreen(false)}
          title="Code Comparison - Fullscreen"
          size="full"
        >
          {content}
        </Modal>
      )}
    </>
  )
}

export default CodeComparison




