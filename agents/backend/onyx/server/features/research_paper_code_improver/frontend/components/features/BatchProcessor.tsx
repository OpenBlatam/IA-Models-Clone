'use client'

import React, { useState } from 'react'
import { FileCode, CheckCircle, XCircle, Loader } from 'lucide-react'
import { Card, Button, ProgressBar, Badge, Modal } from '../ui'
import FileUploader from './FileUploader'
import { useImproveCodeText } from '@/hooks'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'
import toast from 'react-hot-toast'

interface BatchFile {
  id: string
  name: string
  content: string
  language?: string
  status: 'pending' | 'processing' | 'completed' | 'error'
  result?: {
    improved: string
    suggestions: Array<{ type: string; description: string }>
  }
  error?: string
}

interface BatchProcessorProps {
  onComplete?: (results: BatchFile[]) => void
}

const BatchProcessor: React.FC<BatchProcessorProps> = ({ onComplete }) => {
  const [files, setFiles] = useState<BatchFile[]>([])
  const [isProcessing, setIsProcessing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [selectedFile, setSelectedFile] = useState<BatchFile | null>(null)
  const improveCodeMutation = useImproveCodeText()

  const handleFileUpload = async (file: File) => {
    const content = await file.text()
    const language = file.name.split('.').pop() || 'python'
    
    const newFile: BatchFile = {
      id: Date.now().toString() + Math.random(),
      name: file.name,
      content,
      language,
      status: 'pending',
    }
    
    setFiles([...files, newFile])
  }

  const handleProcessAll = async () => {
    if (files.length === 0) {
      toast.error('No files to process')
      return
    }

    setIsProcessing(true)
    setProgress(0)

    const results: BatchFile[] = []

    for (let i = 0; i < files.length; i++) {
      const file = files[i]
      
      // Update status to processing
      setFiles((prev) =>
        prev.map((f) =>
          f.id === file.id ? { ...f, status: 'processing' } : f
        )
      )

      try {
        const response = await improveCodeMutation.mutateAsync({
          code: file.content,
          modelId: undefined,
        })

        const updatedFile: BatchFile = {
          ...file,
          status: 'completed',
          result: {
            improved: response.improved_code,
            suggestions: response.suggestions,
          },
        }

        results.push(updatedFile)
        setFiles((prev) =>
          prev.map((f) => (f.id === file.id ? updatedFile : f))
        )
      } catch (error) {
        const errorFile: BatchFile = {
          ...file,
          status: 'error',
          error: error instanceof Error ? error.message : 'Processing failed',
        }
        results.push(errorFile)
        setFiles((prev) =>
          prev.map((f) => (f.id === file.id ? errorFile : f))
        )
      }

      setProgress(((i + 1) / files.length) * 100)
    }

    setIsProcessing(false)
    onComplete?.(results)
    toast.success(`Processed ${results.filter((f) => f.status === 'completed').length} of ${files.length} files`)
  }

  const handleRemoveFile = (id: string) => {
    setFiles(files.filter((f) => f.id !== id))
  }

  const completedCount = files.filter((f) => f.status === 'completed').length
  const errorCount = files.filter((f) => f.status === 'error').length
  const processingCount = files.filter((f) => f.status === 'processing').length

  return (
    <div className="space-y-6">
      <Card>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-gray-900">
              Batch Code Processing
            </h2>
            <Badge variant="info">
              {files.length} file{files.length !== 1 ? 's' : ''}
            </Badge>
          </div>

          <FileUploader
            onUpload={handleFileUpload}
            accept={{ 'text/plain': ['.txt', '.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs', '.php', '.rb'] }}
            maxSize={10 * 1024 * 1024}
            maxFiles={10}
            label="Upload Code Files"
            helperText="Upload multiple code files for batch processing"
          />

          {files.length > 0 && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex gap-2">
                  <Badge variant="success">{completedCount} completed</Badge>
                  {processingCount > 0 && (
                    <Badge variant="info">{processingCount} processing</Badge>
                  )}
                  {errorCount > 0 && (
                    <Badge variant="error">{errorCount} errors</Badge>
                  )}
                </div>
                <Button
                  onClick={handleProcessAll}
                  disabled={isProcessing || files.length === 0}
                  isLoading={isProcessing}
                >
                  <FileCode className="w-4 h-4 mr-2" />
                  Process All
                </Button>
              </div>

              {isProcessing && (
                <ProgressBar
                  value={progress}
                  label="Processing Files"
                  showPercentage
                />
              )}

              <div className="space-y-2 max-h-64 overflow-y-auto">
                {files.map((file) => (
                  <div
                    key={file.id}
                    className="flex items-center justify-between p-3 border border-gray-200 rounded-lg hover:border-primary-300 transition-colors"
                  >
                    <div className="flex items-center gap-3 flex-1 min-w-0">
                      {file.status === 'processing' && (
                        <Loader className="w-4 h-4 text-primary-600 animate-spin" />
                      )}
                      {file.status === 'completed' && (
                        <CheckCircle className="w-4 h-4 text-green-600" />
                      )}
                      {file.status === 'error' && (
                        <XCircle className="w-4 h-4 text-red-600" />
                      )}
                      {file.status === 'pending' && (
                        <FileCode className="w-4 h-4 text-gray-400" />
                      )}
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-gray-900 truncate">
                          {file.name}
                        </p>
                        {file.language && (
                          <p className="text-xs text-gray-500">{file.language}</p>
                        )}
                      </div>
                      <Badge
                        variant={
                          file.status === 'completed'
                            ? 'success'
                            : file.status === 'error'
                            ? 'error'
                            : file.status === 'processing'
                            ? 'info'
                            : 'default'
                        }
                        size="sm"
                      >
                        {file.status}
                      </Badge>
                    </div>
                    <div className="flex gap-2 ml-4">
                      {file.status === 'completed' && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => setSelectedFile(file)}
                        >
                          View
                        </Button>
                      )}
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleRemoveFile(file.id)}
                        disabled={isProcessing}
                      >
                        <XCircle className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </Card>

      {selectedFile && selectedFile.result && (
        <Modal
          isOpen={!!selectedFile}
          onClose={() => setSelectedFile(null)}
          title={`Results: ${selectedFile.name}`}
          size="xl"
        >
          <div className="space-y-4">
            <div>
              <h3 className="font-semibold mb-2">Improved Code</h3>
              <div className="overflow-x-auto">
                <SyntaxHighlighter
                  language={selectedFile.language || 'python'}
                  style={vscDarkPlus}
                  customStyle={{ borderRadius: '8px', fontSize: '14px' }}
                >
                  {selectedFile.result.improved}
                </SyntaxHighlighter>
              </div>
            </div>
            {selectedFile.result.suggestions.length > 0 && (
              <div>
                <h3 className="font-semibold mb-2">Suggestions</h3>
                <div className="space-y-2">
                  {selectedFile.result.suggestions.map((suggestion, index) => (
                    <div
                      key={index}
                      className="p-3 bg-gray-50 rounded-lg border border-gray-200"
                    >
                      <p className="text-sm text-gray-700">
                        {suggestion.description}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </Modal>
      )}
    </div>
  )
}

export default BatchProcessor

