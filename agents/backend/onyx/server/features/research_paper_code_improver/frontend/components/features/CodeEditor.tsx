'use client'

import React, { useState, useRef, useEffect } from 'react'
import { Copy, Download, Maximize2, Minimize2, Settings } from 'lucide-react'
import { Button, Card, Select } from '../ui'
import CodeSnippet from './CodeSnippet'
import { storage } from '@/lib/utils'
import toast from 'react-hot-toast'

interface CodeEditorProps {
  code: string
  language?: string
  onChange?: (code: string) => void
  readOnly?: boolean
  showToolbar?: boolean
  className?: string
  onSave?: (code: string) => void
}

const CodeEditor: React.FC<CodeEditorProps> = ({
  code: initialCode,
  language = 'python',
  onChange,
  readOnly = false,
  showToolbar = true,
  className = '',
  onSave,
}) => {
  const [code, setCode] = useState(initialCode)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [editorSettings, setEditorSettings] = useState({
    fontSize: 14,
    tabSize: 2,
    wordWrap: false,
    lineNumbers: true,
  })
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    setCode(initialCode)
  }, [initialCode])

  const handleCodeChange = (newCode: string) => {
    setCode(newCode)
    onChange?.(newCode)
  }

  const handleCopy = () => {
    navigator.clipboard.writeText(code)
    toast.success('Code copied to clipboard!')
  }

  const handleDownload = () => {
    const blob = new Blob([code], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `code.${language}`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
    toast.success('Code downloaded!')
  }

  const handleSave = () => {
    onSave?.(code)
    toast.success('Code saved!')
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Tab' && !readOnly) {
      e.preventDefault()
      const textarea = e.currentTarget
      const start = textarea.selectionStart
      const end = textarea.selectionEnd
      const newCode =
        code.substring(0, start) +
        ' '.repeat(editorSettings.tabSize) +
        code.substring(end)
      handleCodeChange(newCode)
      setTimeout(() => {
        textarea.selectionStart = textarea.selectionEnd =
          start + editorSettings.tabSize
      }, 0)
    }

    if ((e.metaKey || e.ctrlKey) && e.key === 's') {
      e.preventDefault()
      if (onSave) {
        handleSave()
      }
    }
  }

  const editorContent = (
    <div className="relative">
      {showToolbar && (
        <div className="flex items-center justify-between p-2 bg-gray-50 border-b border-gray-200">
          <div className="flex items-center gap-2">
            <Select
              value={language}
              onChange={(e) => {
                // Language change would be handled by parent
              }}
              options={[
                { value: 'python', label: 'Python' },
                { value: 'javascript', label: 'JavaScript' },
                { value: 'typescript', label: 'TypeScript' },
                { value: 'java', label: 'Java' },
                { value: 'cpp', label: 'C++' },
                { value: 'go', label: 'Go' },
                { value: 'rust', label: 'Rust' },
              ]}
              disabled={readOnly}
            />
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowSettings(!showSettings)}
              title="Settings"
            >
              <Settings className="w-4 h-4" />
            </Button>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" onClick={handleCopy}>
              <Copy className="w-4 h-4 mr-2" />
              Copy
            </Button>
            <Button variant="ghost" size="sm" onClick={handleDownload}>
              <Download className="w-4 h-4 mr-2" />
              Download
            </Button>
            {onSave && !readOnly && (
              <Button variant="primary" size="sm" onClick={handleSave}>
                Save
              </Button>
            )}
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsFullscreen(!isFullscreen)}
            >
              {isFullscreen ? (
                <>
                  <Minimize2 className="w-4 h-4 mr-2" />
                  Exit
                </>
              ) : (
                <>
                  <Maximize2 className="w-4 h-4 mr-2" />
                  Fullscreen
                </>
              )}
            </Button>
          </div>
        </div>
      )}

      {showSettings && (
        <div className="p-4 bg-gray-50 border-b border-gray-200 space-y-3">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Font Size
              </label>
              <input
                type="number"
                min="10"
                max="24"
                value={editorSettings.fontSize}
                onChange={(e) =>
                  setEditorSettings({
                    ...editorSettings,
                    fontSize: parseInt(e.target.value) || 14,
                  })
                }
                className="w-full px-3 py-2 border border-gray-300 rounded-lg"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Tab Size
              </label>
              <input
                type="number"
                min="1"
                max="8"
                value={editorSettings.tabSize}
                onChange={(e) =>
                  setEditorSettings({
                    ...editorSettings,
                    tabSize: parseInt(e.target.value) || 2,
                  })
                }
                className="w-full px-3 py-2 border border-gray-300 rounded-lg"
              />
            </div>
          </div>
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="wordWrap"
              checked={editorSettings.wordWrap}
              onChange={(e) =>
                setEditorSettings({
                  ...editorSettings,
                  wordWrap: e.target.checked,
                })
              }
              className="rounded border-gray-300 text-primary-600"
            />
            <label htmlFor="wordWrap" className="text-sm text-gray-700">
              Word Wrap
            </label>
          </div>
        </div>
      )}

      {readOnly ? (
        <CodeSnippet
          code={code}
          language={language}
          showLineNumbers={editorSettings.lineNumbers}
        />
      ) : (
        <textarea
          ref={textareaRef}
          value={code}
          onChange={(e) => handleCodeChange(e.target.value)}
          onKeyDown={handleKeyDown}
          className={`w-full h-full min-h-[400px] p-4 font-mono text-sm border-0 focus:outline-none resize-none ${className}`}
          style={{
            fontSize: `${editorSettings.fontSize}px`,
            tabSize: editorSettings.tabSize,
            whiteSpace: editorSettings.wordWrap ? 'pre-wrap' : 'pre',
          }}
          spellCheck={false}
          readOnly={readOnly}
        />
      )}
    </div>
  )

  if (isFullscreen) {
    return (
      <div className="fixed inset-0 z-50 bg-white flex flex-col">
        {editorContent}
      </div>
    )
  }

  return <Card padding="none">{editorContent}</Card>
}

export default CodeEditor

