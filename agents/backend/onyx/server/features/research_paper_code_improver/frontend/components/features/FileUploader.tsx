'use client'

import React, { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, File, X, CheckCircle, AlertCircle } from 'lucide-react'
import { Button, Card, ProgressBar, Badge } from '../ui'
import { clsx } from 'clsx'

interface FileUploaderProps {
  onUpload: (file: File) => Promise<void>
  accept?: Record<string, string[]>
  maxSize?: number
  maxFiles?: number
  label?: string
  helperText?: string
  className?: string
}

const FileUploader: React.FC<FileUploaderProps> = ({
  onUpload,
  accept = { 'application/pdf': ['.pdf'] },
  maxSize = 50 * 1024 * 1024, // 50MB default
  maxFiles = 1,
  label,
  helperText,
  className,
}) => {
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [error, setError] = useState<string | null>(null)

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      if (acceptedFiles.length === 0) return

      const file = acceptedFiles[0]
      setError(null)
      setUploading(true)
      setProgress(0)

      // Simulate progress
      const progressInterval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 90) {
            clearInterval(progressInterval)
            return 90
          }
          return prev + 10
        })
      }, 100)

      try {
        await onUpload(file)
        setProgress(100)
        setUploadedFile(file)
        setTimeout(() => {
          setUploadedFile(null)
          setProgress(0)
        }, 2000)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Upload failed')
      } finally {
        clearInterval(progressInterval)
        setUploading(false)
      }
    },
    [onUpload]
  )

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept,
    maxSize,
    maxFiles,
    disabled: uploading,
  })

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i]
  }

  return (
    <div className={clsx('w-full', className)}>
      {label && (
        <label className="block text-sm font-medium text-gray-700 mb-2">
          {label}
        </label>
      )}

      <Card>
        <div
          {...getRootProps()}
          className={clsx(
            'border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors',
            isDragActive
              ? 'border-primary-500 bg-primary-50'
              : 'border-gray-300 hover:border-primary-400',
            uploading && 'opacity-50 cursor-not-allowed'
          )}
        >
          <input {...getInputProps()} />
          {uploading ? (
            <div className="space-y-4">
              <div className="flex justify-center">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600" />
              </div>
              <div className="space-y-2">
                <p className="text-sm font-medium text-gray-700">Uploading...</p>
                <ProgressBar value={progress} showPercentage />
              </div>
            </div>
          ) : uploadedFile ? (
            <div className="space-y-2">
              <CheckCircle className="w-12 h-12 mx-auto text-green-600" />
              <p className="text-sm font-medium text-green-700">
                {uploadedFile.name} uploaded successfully!
              </p>
            </div>
          ) : error ? (
            <div className="space-y-2">
              <AlertCircle className="w-12 h-12 mx-auto text-red-600" />
              <p className="text-sm font-medium text-red-700">{error}</p>
              <Button
                variant="outline"
                size="sm"
                onClick={(e) => {
                  e.stopPropagation()
                  setError(null)
                }}
              >
                Try Again
              </Button>
            </div>
          ) : (
            <div className="space-y-4">
              <Upload className="w-12 h-12 mx-auto text-gray-400" />
              <div>
                <p className="text-sm font-medium text-gray-700">
                  {isDragActive
                    ? 'Drop the file here'
                    : 'Drag and drop a file here, or click to select'}
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  Max size: {formatFileSize(maxSize)} • Max files: {maxFiles}
                </p>
              </div>
            </div>
          )}
        </div>
      </Card>

      {helperText && !error && (
        <p className="mt-2 text-xs text-gray-500">{helperText}</p>
      )}
    </div>
  )
}

export default FileUploader




